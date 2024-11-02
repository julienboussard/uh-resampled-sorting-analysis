# uhd-resampled-sorting-analysis
This repository contains code used for analyzing differences between ultra high density Neuropixels recordings and resampled (NP1.0, NP2.0, Large-dense) patterns.

It relies on the dartsort repository (uhd_np branch) available here https://github.com/cwindolf/dartsort.git

src.resampled_locs_amps contains functions that estimate the locations and amplitudes of spikes in UHD and resampled recordings, as well as a function to compute the per-unit location spread. 

src.template_snr contains a function to compute the templates SNR in UHD and resampled recordings.

notebooks contain 2 notebooks showing how to use the code to get all the data (amplitudes, locations and template SNRs) as well as computing statistical tests. 
