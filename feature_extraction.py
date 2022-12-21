# Code adapted from https://github.com/jeandeducla/ML-Time-Series

# Tools to extract features from the signals. 
# The following functions help us build the feature vectors out of raw signals. 
# These functions help us extract statistical and geometrical features from 
# raw signals and jerk signals (acceleration first derivative), frequency domain 
# features from raw signals and jerk signals 
# 
# For each sample we extract the following features:
# x,y and z raw signals : mean, max, min, standard deviation, skewness, kurtosis, 
#                         interquartile range, median absolute deviation, 
#                         area under curve, area under squared curve

# x,y and z jerk signals (first derivative) : mean, max, min, standard deviation, skewness, kurtosis, 
#                         interquartile range, median absolute deviation, 
#                         area under curve, area under squared curve

# x,y and z raw signals Discrete Fourier Transform: mean, max, min, standard deviation, skewness, kurtosis, 
#                         interquartile range, median absolute deviation, 
#                         area under curve, area under squared curve, 
#                         weighted mean frequency, 
#                         5 first DFT coefficients, 
#                         5 first local maxima of DFT coefficients and their corresponding frequencies.
# x,y and z jerk signals Discrete Fourier Transform: mean, max, min, standard deviation, skewness, kurtosis, 
#                         interquartile range, median absolute deviation, 
#                         area under curve, area under squared curve, 
#                         weighted mean frequency, 
#                         5 first DFT coefficients, 
#                         5 first local maxima of DFT coefficients and their corresponding frequencies.
# x,y and z correlation coefficients

import scipy.stats as st
from scipy.fftpack import fft, fftfreq 
from scipy.signal import argrelextrema
from sklearn.metrics import f1_score
from sklearn import svm
import operator
import numpy as np

import matplotlib.pylab as plt
import itertools

def stat_area_features(x, Te=1.0):

    mean_ts = np.mean(x, axis=1).reshape(-1,1) # mean
    max_ts = np.amax(x, axis=1).reshape(-1,1) # max
    min_ts = np.amin(x, axis=1).reshape(-1,1) # min
    std_ts = np.std(x, axis=1).reshape(-1,1) # std
    skew_ts = st.skew(x, axis=1).reshape(-1,1) # skew
    kurtosis_ts = st.kurtosis(x, axis=1).reshape(-1,1) # kurtosis 
    iqr_ts = st.iqr(x, axis=1).reshape(-1,1) # interquartile rante
    mad_ts = np.median(np.sort(abs(x - np.median(x, axis=1).reshape(-1,1)),
                               axis=1), axis=1).reshape(-1,1) # median absolute deviation
    area_ts = np.trapz(x, axis=1, dx=Te).reshape(-1,1) # area under curve
    sq_area_ts = np.trapz(x ** 2, axis=1, dx=Te).reshape(-1,1) # area under curve ** 2

    return np.concatenate((mean_ts,max_ts,min_ts,std_ts,skew_ts,kurtosis_ts,
                           iqr_ts,mad_ts,area_ts,sq_area_ts), axis=1)

def frequency_domain_features(x, Te=1.0):

    # As the DFT coefficients and their corresponding frequencies are symetrical arrays
    # with respect to the middle of the array we need to know if the number of readings 
    # in x is even or odd to then split the arrays...
    if x.shape[1]%2 == 0:
        N = int(x.shape[1]/2)
    else:
        N = int(x.shape[1]/2) - 1
    xf = np.repeat(fftfreq(x.shape[1],d=Te)[:N].reshape(1,-1), x.shape[0], axis=0) # frequencies
    dft = np.abs(fft(x, axis=1))[:,:N] # DFT coefficients   
    
    # statistical and area features
    dft_features = stat_area_features(dft, Te=1.0)
    # weighted mean frequency
    dft_weighted_mean_f = np.average(xf, axis=1, weights=dft).reshape(-1,1)
    # 5 first DFT coefficients 
    dft_first_coef = dft[:,:5]    
    # 5 first local maxima of DFT coefficients and their corresponding frequencies
    dft_max_coef = np.zeros((x.shape[0],5))
    dft_max_coef_f = np.zeros((x.shape[0],5))
    for row in range(x.shape[0]):
        # finds all local maximas indexes
        extrema_ind = argrelextrema(dft[row,:], np.greater, axis=0) 
        # makes a list of tuples (DFT_i, f_i) of all the local maxima
        # and keeps the 5 biggest...
        extrema_row = sorted([(dft[row,:][j],xf[row,j]) for j in extrema_ind[0]],
                             key=operator.itemgetter(0), reverse=True)[:5] 
        for i, ext in enumerate(extrema_row):
            dft_max_coef[row,i] = ext[0]
            dft_max_coef_f[row,i] = ext[1]    
    
    return np.concatenate((dft_features,dft_weighted_mean_f,dft_first_coef,
                           dft_max_coef,dft_max_coef_f), axis=1)

def make_feature_vectors(signals_array, Te=1.0):
# This function is modified from original to include more than x, y, z dimensions,
# given that ECGs provide 15 time series altogether instead of just 3
# Its purpose is to convert the 15 x N_samples input time series
    corrs = []
    feats = np.empty((signals_array.shape[1], 0))

    # Correlations are to be iterated over patients dimension
    for patient in range(signals_array.shape[1]): 
        # Compute the correlation coefficient for each of the 15 signals
        corrs_patient = np.corrcoef(signals_array[:,:,patient])
        # Extract the flattened superior diagonal of the correlation matrix
        corr_vector = corrs_patient[np.triu_indices(corrs_patient.shape[0])]
        # Add it to the collection of patients
        corrs.append(corr_vector)
    corrs = np.array(corrs)

    # ECG leads are to be iterated over the leads dimension
    for signal_samples in signals_array:
        # Raw signals :  stat and area features
        features_t = stat_area_features(signal_samples, Te=Te)
        # Jerk signals :  stat and area features
        features_t_jerk = stat_area_features((signal_samples[:,1:] - signal_samples[:,:-1])/Te, Te=Te)    
        # Raw signals : frequency domain features 
        features_f = frequency_domain_features(signal_samples, Te=1/Te)    
        # Jerk signals : frequency domain features 
        features_f_jerk = frequency_domain_features((signal_samples[:,1:] - signal_samples[:,:-1])/Te, Te=1/Te)

        features_array = np.hstack((features_t,
                                    features_t_jerk,
                                    features_f,
                                    features_f_jerk))
        feats = np.hstack((feats, features_array))

    feats = np.concatenate((feats, corrs), axis=1)
    print('Features matrix:', feats.shape)

    return feats
