import numpy as np
from numpy import argmax, argmin, sqrt
import pandas as pd
import matplotlib.pyplot as plt
from math import log
import tqdm
# from scipy import interpolate, ndimage
# from scipy.special import expit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Photometry data file containing 5 columns: Hel.JD, I magnitude, magnitude error, seeing estimation (in pixels - 0.26"/pixel) and sky level.
df = pd.read_csv('phot.dat', sep=' ', names=['HJD', 'I-magnitude', 'Magnitude Error', 'Seeing Estimation', 'sky level'])
# Cleaning the data
# convert HJD to time in days and normalized it such that the peak of the curve will be at t = 0.
Magnitude = df['I-magnitude']
M_star = np.average(df[df['HJD'] < 2.45841 * 10 ** 6]['I-magnitude'])
M_star_error = sqrt(np.var(df[df['HJD'] < 2.45841 * 10 ** 6]['I-magnitude']))

df['I/I_star'] = 10**((df['I-magnitude'] - M_star)/-2.5)
df['I/I_star error'] = 10 ** (-(1 / 2.5) * (df['I-magnitude'] - M_star)) * log(10) * (1 / 2.5) * \
                sqrt(df['Magnitude Error'] ** 2 + M_star_error ** 2)

time_where_mag_is_max = df['HJD'][argmax(df['I/I_star'])]
print(time_where_mag_is_max)
#2019-4-26 9:54:38.02     year-month-day hour:minute:second#
df['days'] = df['HJD'] - time_where_mag_is_max


def chi_squared_fitting(df):
    
    return
