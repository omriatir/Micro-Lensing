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
#2019-4-26 9:54:38.02     year-month-day hour:minute:second#
df['days'] = df['HJD'] - time_where_mag_is_max
# the form of the fitting that we want to minimize chi_squared in is : a + bx + cx^2
# the unknown vector is : (a,b,c)

#taking only points near the peak:
smaller_df = pd.merge(df.loc[df['days'] < 25 ] , df.loc[df['days'] > -25 ])
N = len(smaller_df['days'])

def chi_square_parbola_fit(df):
    x = df['days']
    y = df['I/I_star']
    y_er = df['I/I_star error']**2
    V = np.diag(y_er)
    Vinv = np.linalg.inv(V)
    Ct = np.array([x ** 2, x, np.full(len(x), 1)], dtype='float64')
    C = np.transpose(Ct)
    a = np.linalg.multi_dot([np.linalg.inv(np.linalg.multi_dot([Ct, Vinv, C])), Ct, Vinv, y])
    asig = np.sqrt(np.diag(np.linalg.inv(np.linalg.multi_dot([Ct, Vinv, C]))))

    return a, asig


print(chi_square_parbola_fit(smaller_df))



matrix = np.zeros((3,3))
data_vector = np.array([0,0,0] , dtype=np.int64 )

for j in range(3):
    for i in range(N):
        matrix[j][0] = matrix[j][0] + smaller_df['days'][i]**j / smaller_df['I/I_star error'][i]**2
    for i in range(N):
        matrix[j][1] = matrix[j][1] + smaller_df['days'][i]**(j+1) / smaller_df['I/I_star error'][i] ** 2
    for i in range(N):
        matrix[j][2] = matrix[j][2] + smaller_df['days'][i]**(j+2) / smaller_df['I/I_star error'][i] ** 2

for j in range(3):
    for i in range(N):
        data_vector[j] = data_vector[j] + smaller_df['I/I_star'][i] * smaller_df['days'][i]**j / smaller_df['I/I_star error'][i]**2

optimize_params = np.linalg.solve(matrix,data_vector)
print(optimize_params)

smaller_df.to_csv("results.csv",index=False)

# extracted params :
T_0 =  optimize_params[1] / (-2 *optimize_params[2])
I_max = optimize_params[0] - optimize_params[1]**2 / (4*optimize_params[2])


# now we start the bootstrap method:
T_arr = np.zeros(10000)
u_min_arr = np.zeros(10000)

#for i in range(10000):

