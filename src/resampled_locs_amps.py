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
                                # extract_radius = 200, 
                                loc_radius = 100, 
                                n_spikes_fit_tpca = 10_000,
                                batch_size=1024, 
                                model="dipole",
                                #pca_rank=8,
                                # loc_workers=4, dtype=np.float32,
                                ):
    
    """
    This function takes spikes detected at time `times_samples` on channels `channels`
    It then denoises the spikes (using temporal PCA + neural network), gets their amplitude, and localizes the spikes 
    """
    
    channel_index = make_channel_index(geom,loc_radius)
    c = channel_index.shape[1]
    main_channels = ((geom_ultra[channels][None] - geom[:, None])**2).sum(2).argmin(0)
    denoiser = SingleChannelWaveformDenoiser(channel_index)

    n_spikes = len(times_samples)
    print("fitting pca")
    idx_pca = np.random.choice(n_spikes, n_spikes_fit_tpca, replace=False)

    wfs = read_waveforms_channel_index(
        recording,
        times_samples[idx_pca],
        channel_index,
        main_channels[idx_pca])

    _, T, C = wfs.shape
    wfs = denoiser(torch.tensor(wfs), max_channels=main_channels[idx_pca])
 
    pca_temporal = BaseTemporalPCA(channel_index, geom=geom)
    pca_temporal.fit(wfs, main_channels[idx_pca])

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
    
        wfs = denoiser(torch.tensor(wfs), max_channels=main_channels[batch_ids])
        # transform + reverse
        wfs = pca_temporal._transform_in_probe(wfs.permute(0, 2, 1).reshape((-1, T)))
        wfs = pca_temporal._inverse_transform_in_probe(wfs).reshape((-1, c, T)).permute(0, 2, 1).detach().cpu().numpy()
        amp_vector[batch_ids] = np.nanmax(wfs.ptp(1), axis=1)
    
        dict_result=localize_amplitude_vectors(
            wfs.ptp(1),
            geom,
            main_channels=main_channels[batch_ids],
            channel_index=channel_index,
            model=model, # try both? which one make more sense 
        )      
        loc_vector[batch_ids, 0] = dict_result['x']
        loc_vector[batch_ids, 1] = dict_result['y']
        loc_vector[batch_ids, 2] = dict_result['z_abs']
        
    return loc_vector, amp_vector*rec_sd

