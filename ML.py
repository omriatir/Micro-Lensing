import numpy as np
from numpy import argmax, argmin, sqrt
import pandas as pd
import matplotlib.pyplot as plt
from math import log

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
print(time_where_mag_is_max)
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

def process_fit(a, asig):
    T_0  = -a[1] / (2 * a[0])
    T_error = np.sqrt((asig[1] / (2 * a[0])) ** 2 + (a[1] * asig[0] / (2 * a[0] ** 2)) ** 2)
    I_max = a[2] - (a[1] ** 2 / (4 * a[0]))
    I_max_error = np.sqrt(
        (asig[2]) ** 2 + (2 * a[1] * asig[1] / (4 * a[0])) ** 2 + (a[1] ** 2 * asig[0] / (4 * a[0] ** 2)) ** 2)

    return T_0, T_error, I_max, I_max_error

# the initial parameters when running chi squared for real data is:
umin_theo = 0.919
umin_theo_error = 0.003
a , asig = chi_square_parbola_fit(smaller_df)
initial_params = process_fit(a,asig)
umin_initial = sqrt(-2+2/np.sqrt(1-1/initial_params[2]**2))
umin_initial_error = -(sqrt(-2+2/sqrt(1-initial_params[2]**-2))*(1-1/initial_params[2]**2)**1.5 * initial_params[2]**3)**-1 * initial_params[3]
print(umin_initial_error)
print(f"{initial_params[0] + time_where_mag_is_max}")
print(f"{initial_params[1]=}")
N_sigma = np.abs(umin_initial-umin_theo)/sqrt(umin_theo_error**2 + umin_initial_error**2)
print("Umin_initial_par is: ")
print(umin_initial)
print("N sigma is: ")
print(N_sigma)

new_x = np.linspace(-25, 25, len(smaller_df))
predict = a[0]*new_x**2+a[1]*new_x+a[2]

fig, axes = plt.subplots(2, 1, figsize=(6, 9))
axes[0].errorbar(smaller_df['days'], smaller_df['I/I_star'], yerr=smaller_df['I/I_star error'], fmt='o', ms=3)
axes[0].plot(new_x, predict, c='red')
axes[0].grid(1)
axes[0].title.set_text('Chi Squared Parabolic Fitting')
axes[0].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='$I/I_{*}$')

axes[1].errorbar(smaller_df['days'], predict-smaller_df['I/I_star'], yerr=smaller_df['I/I_star error'], fmt='o', ms=3)
axes[1].axhline(y=0, color='r', linestyle='-', zorder=5)
axes[1].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='$F(x_i)-y_i$')
axes[1].title.set_text('Chi Squared Parabolic Fitting Residuals')
axes[1].grid(1)

plt.tight_layout(h_pad=3.0)
plt.show()

# now we use bootstrap method for 10k simulations:

fits = []
for i in range(10000):
    # Sample a new dataset
    data = smaller_df.loc[np.random.choice(smaller_df.index.values, len(smaller_df))]
    # Fit a new graph
    a, asig = chi_square_parbola_fit(data)
    T_0, T_0_error, I_max, I_max_error = process_fit(a, asig)
    params = [T_0, T_0, I_max, I_max_error]
    fits.append(params)

fits_df = pd.DataFrame(fits, columns=['T_0', 'T_0_error', 'I_max', 'I_max_error'])
fits_df['u_min'] = np.sqrt(-2+2/np.sqrt(1-1/fits_df['I_max']**2))

print("The peak of the Umin histogram that we get is: ")
print(max(fits_df['u_min']))
print("The peak of the T0 histogram that we get is: ")
print(max(fits_df['T_0']+ time_where_mag_is_max))
print("The relative goodness of Umin that we got by the histogram is: ")
print(np.abs(umin_initial-max(fits_df['u_min']))/umin_initial)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(fits_df['T_0'], 50)
axes[0].grid(1)
axes[0].title.set_text('$T_0\ Histogram$')
axes[0].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='Count')

axes[1].hist(fits_df['u_min'], 50)
axes[1].grid(1)
axes[1].title.set_text('$u_{min}\ Histogram$')
axes[1].set(xlabel='$u_{min}$', ylabel='Count')

plt.show()




