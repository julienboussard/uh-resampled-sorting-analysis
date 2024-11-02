import numpy as np
from tqdm.auto import tqdm
import torch

from dartsort.transform.temporal_pca import BaseTemporalPCA
from dartsort.transform.single_channel_denoiser import SingleChannelWaveformDenoiser

from dartsort.util.waveform_util import make_channel_index
from dartsort.util.spikeio import read_waveforms_channel_index
from dartsort.localize.localize_torch import localize_amplitude_vectors


def relocalize_after_clustering(recording, 
                                geom, 
                                times_samples, 
                                channels, # UHD max channels 
                                rec_sd,
                                geom_ultra,
                                loc_radius = 100, 
                                n_spikes_fit_tpca = 10_000,
                                batch_size=1024, 
                                model="dipole",
                                device=None,
                                ):
    
    """
    This function takes spikes detected at time `times_samples` on channels `channels`
    It then denoises the spikes (using temporal PCA + neural network), gets their amplitude, and localizes the spikes. 

    For consistency across the different patterns, it takes the max channels from the UHD pattern and assigns the closest 
    channel in the resampled geometry as the max channel.
    """
    
    channel_index = make_channel_index(geom,loc_radius)
    c = channel_index.shape[1]
    main_channels = ((geom_ultra[channels][None] - geom[:, None])**2).sum(2).argmin(0)
    denoiser = SingleChannelWaveformDenoiser(channel_index).to(device)

    n_spikes = len(times_samples)
    print("fitting pca")
    idx_pca = np.random.choice(n_spikes, n_spikes_fit_tpca, replace=False)

    wfs = read_waveforms_channel_index(
        recording,
        times_samples[idx_pca],
        channel_index,
        main_channels[idx_pca])

    _, T, C = wfs.shape
    wfs = denoiser(torch.tensor(wfs, device=device), max_channels=main_channels[idx_pca])
 
    pca_temporal = BaseTemporalPCA(channel_index, geom=geom)
    pca_temporal.fit(wfs.detach().cpu(), main_channels[idx_pca])

    # Run denoiser + PCA 
    amp_vector = np.zeros(n_spikes)
    loc_vector = np.zeros((n_spikes, 3))
    n_chunks = n_spikes // batch_size + 1

    print("relocalizing")
    for k in tqdm(range(n_chunks)):
        batch_ids = np.arange(k*batch_size, min((k+1)*batch_size, len(times_samples)))
        wfs = read_waveforms_channel_index(
            recording,
            times_samples[batch_ids],
            channel_index,
            main_channels[batch_ids])
    
        wfs = denoiser(torch.tensor(wfs, device=device), max_channels=main_channels[batch_ids]).detach().cpu()
        # transform + reverse
        wfs = pca_temporal._transform_in_probe(wfs.permute(0, 2, 1).reshape((-1, T)))
        wfs = pca_temporal._inverse_transform_in_probe(wfs).reshape((-1, c, T)).permute(0, 2, 1)
        dict_result=localize_amplitude_vectors(
            torch.tensor(wfs.max(1).values - wfs.min(1).values, device=device),
            geom,
            main_channels=main_channels[batch_ids],
            channel_index=channel_index,
            model=model, # try both? which one make more sense 
        )
        
        loc_vector[batch_ids, 0] = dict_result['x'].detach().cpu().numpy()
        loc_vector[batch_ids, 1] = dict_result['y'].detach().cpu().numpy()
        loc_vector[batch_ids, 2] = dict_result['z_abs'].detach().cpu().numpy()

        wfs = wfs.detach().cpu().numpy()
        amp_vector[batch_ids] = np.nanmax(wfs.ptp(1), axis=1)
        
    return loc_vector, amp_vector*rec_sd

def spread(
    loc_vector, 
    labels,
):
    """
    This function takes locations (or amplitudes), subtract the mean over time and compute the MAD
    """
    unit_ids = np.unique(labels)
    unit_ids = unit_ids[unit_ids>-1]
    n_units = len(unit_ids)

    z_spread = np.zeros(n_units)
    x_spread = np.zeros(n_units)

    for unit in unit_ids:
        # Neew to subtract mean -check what's the best method for that
        x_spread[unit] = loc_vector[labels==unit, 0]/0.675
        z_spread[unit] = loc_vector[labels==unit, 2]/0.675
    return x_spread, z_spread


