# This file contains functions for reading in x-ray diffraction data and initial processing
# 1. read data (contains many try/except blocks to account for variable file naming conventions)
# 2.
import pandas
import numpy as np
from numpy import linalg as LA
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io
import imageio
from scipy.fftpack import fft
from scipy import signal
from PIL import Image

################################################################################
# Experiment Parameters
################################################################################

DAQ_hertz = 25000 # EMG DAQ Hz
T = 1.0/200 # Pilatus frame rate, s
detector_hertz = 200 # Hz
# Bragg's law:  d10 = 1000*lambda_*Sdd/(pixel_size*S10)
lambda_ = .1033
pixel_size = 172 # micrometers. http://www.bio.aps.anl.gov/techniques/FD-HOWTO.html
Sdd = 2006.6 # millimeters

################################################################################
# Function to read data
################################################################################

def read_bg_eq_dc_EMG(Drive_path, day, trial):
    Drive_path = '/Volumes/Argonne2017/Argonne2017-images-data'

    # bg (background) import
    bg = pandas.DataFrame()

    bg_path_trial = Drive_path + '/' + day + '/' + trial + '/qf_results/bg/background_sum_' + trial + '.csv'
    bg_path_notrial = Drive_path + '/' + day + '/' + trial + '/qf_results/bg/background_sum' + '.csv'
    try:
        bg = pandas.read_csv(bg_path_trial)
    except:
        try:
            bg = pandas.read_csv(bg_path_notrial)
        except:
            print('bg import failed')

    # eq (equitorial) import
    eq = pandas.DataFrame()

    eq_path = Drive_path + '/' + day + '/' + trial + '/qf_results/eq_results_bx/summary_eq_' + trial + '_bx15.csv'

    try:
        eq = pandas.read_csv(eq_path)
    except:
        print('eq import failed')

    # dc (diffraction centroids; for meridional data) import
    dc = pandas.DataFrame()

    dc_path_notrial = Drive_path + '/' + day + '/' + trial + '/qf_results/dc_summary/summary_1f_M3_M6' + '.csv'
    dc_path_trial = Drive_path + '/' + day + '/' + trial + '/qf_results/dc_summary/summary_1f_M3_M6_' + trial + '.csv'
    dc_path_notrial_manual = Drive_path + '/' + day + '/' + trial + '/qf_results/manual/dc_summary/summary_1f_M3_M6' + '.csv'
    dc_path_trial_manual = Drive_path + '/' + day + '/' + trial + '/qf_results/manual/dc_summary/summary_1f_M3_M6_' + trial + '.csv'

    try:
        dc = pandas.read_csv(dc_path_notrial)
    except:
        try:
            dc = pandas.read_csv(dc_path_trial)
        except:
            try:
                dc = pandas.read_csv(dc_path_notrial_manual)
            except:
                try:
                    dc = pandas.read_csv(dc_path_trial_manual)
                except:
                    print('dc import failed')

    # EMG import
    EMG = pandas.DataFrame()
    try:
        EMG_path = Drive_path  + '/DAQ_files_Argonnedec2017/' + trial + '.mat'
        EMG = scipy.io.loadmat(EMG_path)
        EMG = EMG['Indata']
    except:
        print('EMG import failed')
    return bg, eq, dc, EMG

################################################################################
# EMG filters
################################################################################

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    # Flips the EMG signal upside down if the greatest polarization is in the minus rather than plus direction
    if y.max() < (-1*y).max():
        y = -1*y
    return y

################################################################################
# PreProcessing
# Includes steps for:
# 1. creating a tif number column that can be used to join dataframes
# 2. reducing data to only the columns needed
# 3. Ensuring that each tif has only one row of data associated with it
# 4. Ensuring that data marked as invalid is represented as a NaN
################################################################################

def Preprocess_bg(bg):
    # Extract tif number
    first_split = pandas.DataFrame((bg['Name'].str.split('_').str[3]))
    tif_num = pandas.DataFrame((first_split['Name'].str.split('.').str[0]).astype(str).astype(int))
    bg['tif_num'] = tif_num
    bg = bg.sort_values(by = ['tif_num']).reset_index(drop = 'True')

    # Select only the columns that you need
    bg = bg[['Sum', 'tif_num']]

    return bg





# Note on processing of equitorial data
# The version of Musclex we used returned the equitorial data with multiple rows for each tiff
# The number of rows was determined by the number of peaks tracked, and the left and right
# sides of the detector. Since we must track 3 peaks for the Voigt model to be properly determined,
# there are 6 rows per tiff image; this needs to be flattened into a single row, with the data
# instead represented in 6 columns. The portion of the code that handles this is bounded by ***

def Preprocess_eq(eq):
    # Extract tif number
    try:
        tif_num = pandas.DataFrame((eq['Filename'].str.split('_').str[3]).astype(int))
    except:
        tif_num = pandas.DataFrame((eq['filename'].str.split('_').str[3]).astype(int))

    eq['tif_num'] = tif_num

    eq = eq.sort_values(by = ['tif_num']).reset_index(drop = 'True')

    # Select only the columns that you need
    eq = eq[['tif_num', 'Side', 'Distance From Center', 'Area', 'S10','Fitting error']]

    # **************************************************************************

    # Create a peak label column that will specify L_10, L_11 and L_20, and right respectively
    indices = []
    peak_label = []
    fitting_er = []
    for i in eq.tif_num.drop_duplicates(): # loop over tifs
        temp_i = eq.where(eq.tif_num == i).dropna(how = 'all') # subset to one tif number
        if len(temp_i.index) == 1:
            peak_label.append(['NoPeak'])
            indices.append(np.array(temp_i.index))
            fitting_er.append([np.nan])

        # Subset for left peaks
        temp = temp_i.where(temp_i.Side == 'left').dropna(how = 'all').sort_values(by = 'Distance From Center')
        indices.append(np.array(temp.index)) # Add indices sorted in increasing order of peak distance
        # Conditional block that determines how many peaks are in the given file and appends them in correct order
        if len(temp.index) == 3:
            peak_label.append(['L_10', 'L_11', 'L_20'])
            fitting_er.append([pandas.to_numeric(temp_i['Fitting error']).sum(), pandas.to_numeric(temp_i['Fitting error']).sum(), pandas.to_numeric(temp_i['Fitting error']).sum()])
        elif len(temp.index) == 2:
            peak_label.append(['L_10', 'L_11'])
            fitting_er.append([pandas.to_numeric(temp_i['Fitting error']).sum(), pandas.to_numeric(temp_i['Fitting error']).sum()])


        # Reset value of temp to the tif number subset
        temp = temp_i
        # Subset for right peaks
        temp = temp_i.where(temp.Side == 'right').dropna(how = 'all').sort_values(by = 'Distance From Center')
        indices.append(np.array(temp.index))
        # Conditional block that determines how many peaks are in the given file and appends them in correct order
        if len(temp.index) == 3:
            peak_label.append(['R_10', 'R_11', 'R_20'])
            fitting_er.append([pandas.to_numeric(temp_i['Fitting error']).sum(), pandas.to_numeric(temp_i['Fitting error']).sum(), pandas.to_numeric(temp_i['Fitting error']).sum()])
        elif len(temp.index) == 2:
            peak_label.append(['R_10', 'R_11'])
            fitting_er.append([pandas.to_numeric(temp_i['Fitting error']).sum(), pandas.to_numeric(temp_i['Fitting error']).sum()])

    # **************************************************************************

    indices = np.concatenate(indices)
    peak_label = np.concatenate(peak_label)
    fitting_er = np.concatenate(fitting_er)

    # Add this column to eq by joining on the index column that we just generated in conjunction with peak label column
    eq = eq.join(pandas.DataFrame({'Indices':indices, 'peak_label':peak_label, 'fitting_er':fitting_er}).set_index('Indices'))


    # Pivot on peak label so that data is clustered first under the column in value, then on the peak label
    eq = eq.pivot(index = 'tif_num', columns = 'peak_label', values = ['Distance From Center', 'Area','fitting_er'])

    # Coalesce multiindexing into one index
    eq.columns = ["_".join((j,k)) for j,k in eq.columns]

    # More convenient naming conventions
    eq = eq.rename(columns = {'Distance From Center_L_10':'S10_L','Distance From Center_L_11':'S11_L','Distance From Center_L_20':'S20_L',
               'Distance From Center_R_10':'S10_R','Distance From Center_R_11':'S11_R','Distance From Center_R_20':'S20_R',
               'Distance From Center_NoPeak':'S10_NoPeak', 'Area_L_10':'I10_L', 'Area_L_11':'I11_L', 'Area_L_20':'I20_L',
               'Area_NoPeak': 'I_NoPeak', 'Area_R_10':'I10_R', 'Area_R_11':'I11_R', 'Area_R_20':'I20_R','fitting_er_L_10':'fitting_error'})

    eq = eq[['S10_L', 'S10_R', 'S11_L', 'S11_R', 'S20_L', 'S20_R', 'I10_L', 'I10_R', 'I11_L', 'I11_R', 'I20_L', 'I20_R','fitting_error']]

    return eq




def Preprocess_dc(dc):
    # Split filename to create a tif number column
    tif_num = pandas.DataFrame((dc['filename'].str.split('_').str[3]))
    tif_num = pandas.DataFrame((tif_num['filename'].str.split('.').str[0]).astype(str).astype(int))
    dc['tif_num'] = tif_num

    # Rename columns without spaces
    dc = dc.rename(columns = {'average M3 centroid':'M3_c', 'average M6 centroid':'M6_c',
                              '51 average centroid':'A51_c', '59 average centroid':'A59_c',
                              'average M3 intensity':'M3_i', 'average M6 intensity':'M6_i',
                              '59 average intensity':'A59_i','51 average intensity':'A51_i'})
    # subset to necessary data
    dc = dc[['tif_num','M3_c','M6_c','A51_c','A59_c','M3_i','M6_i','A59_i','A51_i']]

    # In the version of Musclex we used, data rejected by image anatators was preceded
    # by an underscore. Here these values are replaced with a NaN.
    for i in dc.columns:
        if dc[i].dtype == np.object:
            ind = dc[dc[i].str.startswith('_') == True].index
            dc.loc[ind,i] = np.nan
    return dc




def Preprocess_EMG(EMG_matrix):
    Trig = EMG_matrix[:,0]
    EMG_amplitude = EMG_matrix[:,2]

    EMG = pandas.DataFrame({'seconds':np.arange(0,len(EMG_matrix))*(1/DAQ_hertz)})
    EMG['Trig'] = Trig
    EMG['EMG_amplitude'] = EMG_amplitude

    # Cutting to have only EMG in trigger region; this is where  the detector
    # was triggered to record. Reset time to where trigger >1 to start at 0
    # seconds by substracting universally the first value in the seconds column

    EMG = EMG[EMG.Trig > 1]
    EMG.seconds = EMG.seconds - EMG.seconds.iloc[0]

    # This was old code from when we were thinking of interpolating detector data to EMG timebase.
    # Setting up EMG time base to be used later for interpolating detector data to EMG timebase
    #t = EMG.seconds

    EMG = EMG.reset_index(drop = 'True')

    # Filter the EMG signal, then find peaks and label with a binary column
    # Filter Parameters
    fs = 25000
    cutoff = 12

    EMG_data = EMG.EMG_amplitude
    EMG_data = butter_highpass_filter(EMG_data, cutoff, fs)

    # Flip the signal so that the largest amplitude peak is positive; the sign
    # of the signal is arbitrary at the recording step.

    if EMG_data.max() < (-1*EMG_data).max():
        EMG_data = -1*EMG_data

    EMG['EMG_amplitude_filter'] = EMG_data

    # Storing peak locations in a binary column

    std_dev = EMG['EMG_amplitude_filter'].std()
    thresh = 2.5*std_dev
    p, _ = signal.find_peaks(np.array(1.0*EMG['EMG_amplitude_filter']), distance = 800, height = thresh) #, np.arange(30,60,5)) # np.arange(60,80))

    # Create a column that numbers peaks, and is zero else
    peaks = []
    indicator = 0
    for i in range(0, len(EMG.seconds)):
        for j in p:
            if i == j:
                indicator=  1
        peaks.append(indicator)
        indicator = 0
    EMG['peaks'] = peaks
    return EMG




def myround(x, base=.005):
    return base * round(x/base)




def Braggs(S):
    D = ((lambda_*Sdd)/(pixel_size*S))*1000
    return D



################################################################################
# Collect all of the preprocessing functions into one process function.
# Includes:
# 1. replace - with nans and cast all values as floats.
# 2. Conversion from pixel to nm space by Bragg's law.
# 3. Averaging sides of the detector
# 4. Replacing data that had a high fit error with a nan
# 5. Merging all the data to one data frame
# 6. Cutting off the first ISI and last ISI since these will not be guarenteed
#    to encompass a full cycle of shortening and lengthening
################################################################################
def process(Drive_path, day, trial):

    bg, eq, dc, EMG = read_bg_eq_dc_EMG(Drive_path, day, trial)

    eq = Preprocess_eq(eq)
    dc = Preprocess_dc(dc)
    bg = Preprocess_bg(bg)
    det = pandas.merge(dc.merge(bg, on = 'tif_num'), eq, on = 'tif_num')

    # Replace any rejected values that we're indicated with a - with a nan
    det.replace('-', np.nan)

    # Convert dataframe to floats
    det = det.astype(float)

    # peaks is the column where we will store a binary indicator of when an EMG spike occured
    det['peaks'] = np.zeros(len(det))

    # seconds uses tif number and the frame rate of the detectore to infer time
    det['seconds'] = det.tif_num*(1.0/detector_hertz)

    # Average left and right sides of the detector
    det['S10'] = (det.S10_L + det.S10_R)/2
    det['S11'] = (det.S11_L + det.S11_R)/2
    det['S20'] = (det.S20_L + det.S20_R)/2

    det['D10'] = Braggs(det.S10)
    det['D11'] = Braggs(det.S11)
    det['D20'] = Braggs(det.S20)

    det.M3_c = Braggs(det.M3_c)
    det.M6_c = Braggs(det.M6_c)
    det.A59_c = Braggs(det.A59_c)
    det.A51_c = Braggs(det.A51_c)

    det['I10'] = (det.I10_L + det.I10_R)/2
    det['I11'] = (det.I11_L + det.I11_R)/2
    det['I20'] = (det.I20_L + det.I20_R)/2

    det['I20_I10'] = det.I20/det.I10
    det['I20I11_I10'] = (det.I20 + det.I11)/det.I10


    det = det[['D10','D11','D20','I10', 'I20I11_I10','I20_I10','fitting_error','seconds',
               'tif_num','M3_c','M6_c','A51_c','A59_c','M3_i','M6_i','A59_i','A51_i','Sum', 'peaks']]

    ## Use EMG to insert a binary peaks column that indicates when the muscle was activated
    EMG = Preprocess_EMG(EMG)

    # Find the times at which peaks occured and round to the same time base as detector
    EMG_seconds = EMG.seconds.where(EMG.peaks > .5).dropna(how = 'all')
    EMG_seconds = myround(EMG_seconds, .005)

    # For the times when a peak occured insert a 1 in to peak label column
    ind = []
    for i in EMG_seconds:
        ind.append(det[det['seconds'] == i].index.values)

    det.peaks[np.concatenate(ind)] = 1
    det['ISI'] = det.peaks.cumsum()

    ## now that all columns are in place, replace any high fit values with NaNs
    ## Edited 10.22: This was made to replace all data with NaNs except for seconds, tif num, sum and peaks column
    data_cols = ['D10','D11','D20','I10', 'I20I11_I10','I20_I10','M3_c',
                'M6_c','A51_c','A59_c','M3_i','M6_i','A59_i','A51_i']
    det[data_cols] = det[data_cols].where(det.fitting_error < .05)

    # Cut off the first 10 images; generally 8 frames were empty
    det = det.where(det.tif_num > 9).dropna(how = 'all').reset_index(drop = 'True')

    # Cut off first ISI and last ISI, since these may contain only a partial cycle's worth of data; this is especially pertinant for the zeroeth, and could go either way for the -1th.
    det = det.where(det.ISI > det.ISI.min()).dropna(how = 'all').reset_index(drop = 'True')
    det = det.where(det.ISI < det.ISI.max()).dropna(how = 'all').reset_index(drop = 'True')

    return EMG, det
