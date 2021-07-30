import numpy as np
import matplotlib.pyplot as plt 
import os
import re
import sys
import glob
import pandas as pd
import pickle

path = 'Data/'

### Part 1: Formatting spectra files and scaling

# read in spectra csvs
spectra_listing = glob.glob('Data/*/*spectra.csv')
spectra_listing.sort()
spectra = []
spectra_mats = []

for f in spectra_listing:

    # load spectrum file
    spectrum_wl = np.loadtxt(f, delimiter = ',') # spectrum + wavelength
    wavelengths = spectrum_wl[:,0]

    with open('wavelengths.npy', 'wb') as fn:
        np.save(fn, wavelengths)

    spectrum = spectrum_wl[:,1:] # only spectrum

    # reading data from file name and
    # text wrangling for integration time
    #tokens= filename words
    fn_tokens = re.split('_', f)
    tray_number = fn_tokens[-2]
    integration_time = fn_tokens[-3]
    
    integration_time = float(integration_time[:-1])*1000
    
    # divide by integration time for scaling all the files
    spectrum = spectrum/integration_time

    spectrum_width = spectrum.shape[1]

    ## Need to add info about drop volume in rows to take the experimental csv from
    ## 96 rows (wells) to 288 rows (drops) to match the spectra files

    # generate sequences for drop volume and well number to match experimental csv to spectra
    drop_vol = np.tile(np.repeat(np.array([900,800,400]),12),8)
    well_num = np.concatenate([np.tile(np.arange(1,13),3) + (i)*12 for i in np.arange(8)])

    # peak wavelength calculation
    peak_intensities = np.max(spectrum, axis = 0)
    peak_wavelengths = wavelengths[np.argmax(spectrum, axis = 0)]

    # creating a dataframe for the current spectra file
    spectrum_df = pd.DataFrame({
        'well_num': well_num,
        'drop_vol': drop_vol,
        'peak_wavelength': peak_wavelengths,
        'peak_intensity': peak_intensities,
    # read each spectra file into a dataframe as a col of vectors
        'spectrum': spectrum.T.tolist()})

    spectra_mats.append(spectrum)

    spectra.append(spectrum_df) # yields all spectra dataframes in a list

#####################################################################
### Part 2: Adding experimental params to create the main dataframe

# read in experimental csv files (the complement of the spectra csvs)
exp_param_listing = list(set(glob.glob('Data/*/*.csv')) - set(spectra_listing))
exp_param_listing.sort()

# drop volumes to cross join with exp_params csvs
drop_vol = pd.DataFrame({'drop_vol': [900, 800, 400], 'key': 1})

# initialize empty list for individual dfs for each experimental csv
exp_param_parts = []

# similar loop for experimental params as for the spectra
for f in exp_param_listing:
    exp_params = pd.read_csv(f, delimiter = ',').iloc[:,:-3]
    ## This is the merge to expand each row to 3 rows (1 per drop in the well). 
    ## It is a cross-join.
    exp_params['key'] = 1
    exp_params['well_num'] = np.arange(1, exp_params.shape[0] + 1)
    # reshaping dataframe, adding drop volume to experimental param file:
    exp_params_pivot = pd.merge(exp_params, drop_vol, on ='key').drop(columns = "key")
    exp_param_parts.append(exp_params_pivot)

# initialize empty list to store the joined experimental and spectra dfs
joined = []
# join each experimental df to each spectra df
for i,j in zip(exp_param_parts, spectra):
    joined_file = pd.merge(i, j, on = ['well_num', 'drop_vol'])
    joined.append(joined_file)

# concatenate results
result = pd.concat(joined)
result.to_csv('Master_Spectra_DF.csv')
# write result to pickle file for
result.to_pickle("Master_Spectra_DF.pkl")

with open('spectra_mat.npy', 'wb') as f:
    np.save(f, np.concatenate([m.T for m in spectra_mats]))

with open('wavelengths.npy', 'wb') as f:
    np.save(f, wavelengths)
