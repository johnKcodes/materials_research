import numpy as np
import matplotlib.pyplot as plt 
import os
import re
import sys
import glob
import pandas as pd

path = 'Data/'

# read in spectra csvs
spectra_listing = glob.glob('Data/*/*spectra.csv')
spectra_listing.sort()
spectra = []

for f in spectra_listing:

    # load spectrum file
    spectrum_wl = np.loadtxt(f, delimiter = ',') # spectrum + wavelength
    wavelengths = spectrum_wl[:,0]
    spectrum = spectrum_wl[:,1:] # only spectrum

    # reading data from file name and
    # text wrangling for integration time
    fn_tokens = re.split('_', f)
    tray_number = fn_tokens[-2]
    integration_time = fn_tokens[-3]
    
    # if integration time unretrievable, just do 1
    integration_time = float(integration_time[:-1])*1000
    print(integration_time)

    # divide by integration time
    spectrum = spectrum/integration_time

    spectrum_width = spectrum.shape[1]

    # generate sequences for well volume and well number
    well_vol = np.tile(np.repeat(np.array([900,800,400]),12),8)
    well_num = np.concatenate([np.tile(np.arange(1,13),3) + (i)*12 for i in np.arange(8)])

    # peak wavelength calculation
    peak_intensities = np.max(spectrum, axis = 0)
    peak_wavelengths = wavelengths[np.argmax(spectrum, axis = 0)]

    # read spectra into a dataframe as a col of vectors
    spectrum_df = pd.DataFrame({
        'well_num': well_num,
        'well_vol': well_vol,
        'peak_wavelength': peak_wavelengths,
        'peak_intensity': peak_intensities,
        'spectrum': spectrum.T.tolist()})

    spectra.append(spectrum_df)

# read in experimental parameter files (the complement of the spectra csvs)
exp_param_listing = list(set(glob.glob('Data/*/*.csv')) - set(spectra_listing))
exp_param_listing.sort()

# well volumes to cross join with exp_params csvs
well_vols = pd.DataFrame({'well_vol': [900, 800, 400], 'key': 1})

# init empty list for individual dfs
exp_param_parts = []

# similar loop for experimental params as for the spectra
for f in exp_param_listing:
    exp_params = pd.read_csv(f, delimiter = ',').iloc[:,:-3]
    # this merge is a cross-join to expand each row to 3 rows (1 per dot in the well).
    exp_params['key'] = 1
    exp_params['well_num'] = np.arange(1, exp_params.shape[0] + 1)
    exp_params_pivot = pd.merge(exp_params, well_vols, on ='key').drop(columns = "key")
    exp_param_parts.append(exp_params_pivot)

# join each experimental param df to each spectra df
joined = []
for i,j in zip(exp_param_parts, spectra):
    joined_file = pd.merge(i, j, on = ['well_num', 'well_vol'])
    joined.append(joined_file)

# concatenate results
result = pd.concat(joined)
result.to_csv('spectra_experiment_df.csv')
