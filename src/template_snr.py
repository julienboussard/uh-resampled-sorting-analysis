import numpy as np
from tqdm.auto import tqdm
import torch


from dartsort.util.waveform_util import make_channel_index
from dartsort.util.spikeio import read_waveforms_channel_index
from dartsort.util.drift_util import get_waveforms_on_static_channels, get_spike_pitch_shifts, registered_geometry
from dartsort.templates import TemplateData

def compute_template_SNR(
    sorting, # UHD sorting
    recording,
    me,
    chunk_time_ranges_s,
    geom,
    geom_ultra,
    localization_results, # these are the UHD localizations (same locs across all patterns)
    template_config=None,
    template_data_list=None,
    zero_radius=200, # Templates get set to 0 outside of this radius
    n_spikes_computation=2000,
    tsvd=None,
    return_denoising_tsvd=True,
    template_save_dir=None,
    overwrite=False,
    template_npz_filename="template_data.npz",
    n_jobs=0,
    device=None,
    trough_offset_samples=20,
    spike_length_samples=40,
    slice_snr = [1000, 3000],
    sampling_rate=30_000,
    ):
    
    """
    This function:
     - computes registered templates over chunks (to get templates over time)
     - computes smoothed subchunk templates over time
     - for each unit: 
         - read n_spikes_computation spikes fr in chunk or chunks 
         - read n_spikes_computation random spikes 
         - register all spikes
         - compute scalar product with unit/random wfs
         - take ratio of both
    """
    
    # Get registered templates per chunks
    if template_data_list is None:
        res = TemplateData.from_config_multiple_chunks_linear(
            recording,
            chunk_time_ranges_s,
            sorting,
            template_config,
            save_folder=template_save_dir,
            overwrite=overwrite,
            motion_est=me,
            save_npz_name=template_npz_filename,
            n_jobs=n_jobs, 
            device=device,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            denoising_tsvd=tsvd,
            return_denoising_tsvd=return_denoising_tsvd,
        )
        if return_denoising_tsvd:
            template_data_list, tsvd = res
        else:
            template_data_list = res
    
    # For all units: 
    # Read waveforms and register them

    registered_geom = registered_geometry(geom, me)
    # Is this needed or is it ok to use spikeio.read_full_waveforms? 
    channel_index = make_channel_index(geom, radius = 2*geom.max())
    unit_ids = np.unique(sorting.labels)
    unit_ids = unit_ids[unit_ids>-1]
    n_units = len(unit_ids)

    templates_SNR = np.zeros(n_units)

    print("computing main chans")

    main_channels = ((geom_ultra[sorting.channels][None] - geom[:, None])**2).sum(2).argmin(0)
    print("computing SNR")

    idx_time = np.flatnonzero(np.logical_and(sorting.times_seconds > slice_snr[0], sorting.times_seconds < slice_snr[1]))
    
    for unit in tqdm(unit_ids):
        
        idx_unit = idx_time[np.flatnonzero(sorting.labels[idx_time] == unit)]
        n_spikes_unit = len(idx_unit)
        if n_spikes_unit>n_spikes_computation:
            idx_unit = idx_unit[np.random.choice(n_spikes_unit, n_spikes_computation, replace=False)]

        if len(idx_unit):
            # Is this needed or is it ok to use spikeio.read_full_waveforms? 
            wfs = read_waveforms_channel_index(
                    recording,
                    sorting.times_samples[idx_unit],
                    channel_index,
                    main_channels[idx_unit],
                    trough_offset_samples=trough_offset_samples,
                    spike_length_samples=spike_length_samples,
            )
            
            n_pitches_shift = get_spike_pitch_shifts(localization_results[idx_unit, 2], geom, times_s = sorting.times_seconds[idx_unit], motion_est=me)
    
            wfs = get_waveforms_on_static_channels(
                wfs,
                geom,
                main_channels=main_channels[idx_unit],
                channel_index=channel_index,
                target_channels=None, #here, all channels
                n_pitches_shift=n_pitches_shift,
                registered_geom=registered_geom,
            )

            main_channels_random = np.random.choice(main_channels[idx_unit], n_spikes_computation, replace=True)
    
            random_times = np.random.choice(np.arange(slice_snr[0]*sampling_rate, slice_snr[1]*sampling_rate), n_spikes_computation, replace=False)
    
            wfs_random = read_waveforms_channel_index(
                    recording,
                    random_times,
                    channel_index,
                    main_channels_random,
                    trough_offset_samples=trough_offset_samples,
                    spike_length_samples=spike_length_samples,
            )
            
            n_pitches_shift = get_spike_pitch_shifts(geom[main_channels_random, 1], geom, times_s = random_times//sampling_rate, motion_est=me)
    
            wfs_random = get_waveforms_on_static_channels(
                wfs_random,
                geom,
                main_channels=main_channels_random,
                channel_index=channel_index,
                target_channels=None, #here, all channels
                n_pitches_shift=n_pitches_shift,
                registered_geom=registered_geom,
            )
    
            cmp_unit = 0
            cmp_random = 0
    
            dotprod_unit = 0
            dotprod_random = 0
            
            for j, chunk_range in tqdm(enumerate(chunk_time_ranges_s)):
                temp_chunk_unit = template_data_list[j].templates[unit].flatten()
                # mc = temp_chunk_unit.ptp(0).argmax()
                # far_chans = np.flatnonzero(np.sqrt(((registered_geom - registered_geom[mc])**2).sum(1))>200)
                # temp_chunk_unit[:, far_chans] = 0
                # temp_chunk_unit = temp_chunk_unit.flatten()
                
                # keep only close channels - radius 200? turn off channels low amp? 
                # used radius 200 in previous version, doesn't really make a diff? 
                idx_chunk = np.flatnonzero(np.logical_and(sorting.times_seconds[idx_unit] > chunk_range[0], 
                                                          sorting.times_seconds[idx_unit] < chunk_range[1]))
                
                if len(idx_chunk):
                    n_good_chans = (~np.isnan(wfs[idx_chunk][:, 0])).sum(1)
                    if np.any(n_good_chans):
                        dotprodall = np.einsum('ji, i -> j', np.nan_to_num(wfs[idx_chunk], copy=False).reshape((len(idx_chunk), -1)), temp_chunk_unit).sum()/n_good_chans
                        dotprod_unit += dotprodall.sum()
                        
                        cmp_unit+=len(idx_chunk)
                    
                idx_chunk = np.flatnonzero(np.logical_and(random_times/sampling_rate > chunk_range[0], 
                                                          random_times/sampling_rate < chunk_range[1]))
                if len(idx_chunk):
                    n_good_chans = (~np.isnan(wfs_random[idx_chunk][:, 0])).sum(1)
                    if np.any(n_good_chans):
                        dotprodall = np.einsum('ji, i -> j',  np.nan_to_num(wfs_random[idx_chunk], copy=False).reshape((len(idx_chunk), -1)), temp_chunk_unit).sum()/n_good_chans
                        dotprod_random += np.abs(dotprodall).sum()
                        cmp_random += len(idx_chunk)
    
            templates_SNR[unit] = (dotprod_unit*cmp_random) / (dotprod_random*cmp_unit)
        else:
            templates_SNR[unit] = np.nan

    return templates_SNR
