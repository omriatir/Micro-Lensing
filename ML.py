import numpy as np
from numpy import array, linspace, max, min, argmax, argmin, sqrt, abs, exp, histogram, average, var, transpose, meshgrid
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from scipy.optimize import curve_fit


def chi_square_parbola_fit(df):
    x = df['days']
    y = df['I/I_star']
    y_er = df['I/I_star error'] ** 2
    V = np.diag(y_er)
    Vinv = np.linalg.inv(V)
    Ct = np.array([x ** 2, x, np.full(len(x), 1)], dtype='float64')
    C = np.transpose(Ct)
    a = np.linalg.multi_dot([np.linalg.inv(np.linalg.multi_dot([Ct, Vinv, C])), Ct, Vinv, y])
    asig = np.sqrt(np.diag(np.linalg.inv(np.linalg.multi_dot([Ct, Vinv, C]))))
    return a, asig


def process_fit(a, asig):
    T_0 = -a[1] / (2 * a[0])
    T_error = np.sqrt((asig[1] / (2 * a[0])) ** 2 + (a[1] * asig[0] / (2 * a[0] ** 2)) ** 2)
    I_max = a[2] - (a[1] ** 2 / (4 * a[0]))
    I_max_error = sqrt((asig[2]) ** 2 + (2 * a[1] * asig[1] / (4 * a[0])) ** 2
                       + (a[1] ** 2 * asig[0] / (4 * a[0] ** 2)) ** 2)
    return T_0, T_error, I_max, I_max_error


def plot_parabolic_fit_part_a(df, a):
    new_x = linspace(-25, 25, len(df))
    predict = a[0] * new_x ** 2 + a[1] * new_x + a[2]

    fig, axes = plt.subplots(2, 1, figsize=(6, 9))

    axes[0].errorbar(df['days'], df['I/I_star'], yerr=df['I/I_star error'], fmt='o', ms=3)
    axes[0].plot(new_x, predict, c='red')
    axes[0].grid(1)
    axes[0].title.set_text('Chi Squared Parabolic Fitting')
    axes[0].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='$I/I_{*}$')

    axes[1].errorbar(df['days'], predict - df['I/I_star'], yerr=df['I/I_star error'], fmt='o', ms=3)
    axes[1].axhline(y=0, color='r', linestyle='-', zorder=5)
    axes[1].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='$F(x_i)-y_i$')
    axes[1].title.set_text('Chi Squared Parabolic Fitting Residuals')
    axes[1].grid(1)

    plt.tight_layout(h_pad=3.0)
    plt.show()


def bootstrap_part_a(df, num_simulations):
    fits = []
    for i in range(num_simulations):
        # Sample a new dataset
        data = df.loc[np.random.choice(df.index.values, len(df))]
        # Fit a new graph
        a, a_sig = chi_square_parbola_fit(data)
        T_0, T_0_error, I_max, I_max_error = process_fit(a, a_sig)
        params = [T_0, T_0_error, I_max, I_max_error]
        fits.append(params)

    fits_df = pd.DataFrame(fits, columns=['T_0', 'T_0_error', 'I_max', 'I_max_error'])
    fits_df['u_min'] = sqrt(-2 + 2 / np.sqrt(1 - 1 / fits_df['I_max'] ** 2))

    print(f"The peak of the Umin histogram that we get is: \t{max(fits_df['u_min'])}")
    print(f"The peak of the T0 histogram that we get is: \t{max(fits_df['T_0'] + time_where_mag_is_max)}")
    print(
        f"The relative goodness of Umin that we got by the histogram is: \t {abs(umin_initial - max(fits_df['u_min'])) / umin_initial}")
    return fits_df


def gauss(x, *p):
    A, mu, sigma = p
    return A * exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def fit_gauss(df, param):
    x_data = df[param]
    hist, bin_edges = histogram(x_data, 50)
    n = len(hist)
    x_hist = array([(bin_edges[k + 1] + bin_edges[k]) / 2 for k in range(n)])
    y_hist = hist
    mean = sum(x_hist * y_hist) / sum(y_hist)
    sigma = sum(y_hist * (x_hist - mean) ** 2) / sum(y_hist)

    param_optimised, param_covariance_matrix = curve_fit(gauss, x_hist, y_hist, p0=[max(y_hist), mean, 0.05],
                                                         maxfev=5000)
    x_hist_2 = linspace(min(x_hist), max(x_hist), 500)
    return x_hist_2, param_optimised, param_covariance_matrix, x_data


def plot_histogram_part_a(df):
    x_hist_t0, params_t0, params_covar_t0, x_data_t0 = fit_gauss(df, 'T_0')
    x_hist_imax, params_imax, params_covar_imax, x_data_imax = fit_gauss(df, 'u_min')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(df['T_0'], 50)
    axes[0].plot(x_hist_t0, gauss(x_hist_t0, *params_t0), 'r', linewidth=3, label='Gaussian fit')
    axes[0].grid(1)
    axes[0].title.set_text('$T_0\ Histogram$')
    axes[0].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='Count')

    axes[1].hist(df['u_min'], 50)
    axes[1].plot(x_hist_imax, gauss(x_hist_imax, *params_imax), 'r', linewidth=3, label='Gaussian fit')
    axes[1].grid(1)
    axes[1].title.set_text('$u_{min}\ Histogram$')
    axes[1].set(xlabel='$u_{min}$', ylabel='Count')

    plt.show()


def u(umin, T0, tau, t):
    return sqrt(umin ** 2 + ((t - T0) / tau) ** 2)


def u_arr(umin, T0, tau):
    return array([sqrt(umin ** 2 + ((t - T0) / tau) ** 2) for t in df["days"]])


def mu(u):
    return (u ** 2 + 2) / u * sqrt(u ** 2 + 4)


def I_by_Istar(mu, f_bl):
    return mu * f_bl + 1 - f_bl


def chi_square_nonlinear_fit(df, umin_low, umin_high, n_umin, T0_low, T0_high, n_T0, tau, f_bl=1):
    umin_bands = np.linspace(umin_low, umin_high, n_umin)
    T0_bands = np.linspace(T0_low, T0_high, n_T0)
    umin_v, T0_v = np.meshgrid(umin_bands, T0_bands)
    umin_flatten = umin_v.flatten()
    T0_flatten = T0_v.flatten()
    u_values = u_arr(umin_flatten, T0_flatten, tau)
    mu_values = mu(u_values)
    I_by_Istar_valus = I_by_Istar(mu_values, f_bl)

    chi2_flatten = np.array([np.sum(((df['I/I_star'] - I) / df['I/I_star error']) ** 2) for I in I_by_Istar_valus.T])
    index_min_chi2 = chi2_flatten.argmin()
    umin_fit, T0_fit, chi2_fit = umin_flatten[index_min_chi2], T0_flatten[index_min_chi2], chi2_flatten[index_min_chi2]

    # table = pd.DataFrame(chi2_v, columns=umin_bands, index=T0_bands)
    # print(table)
    # plt.contourf(umin_bands, T0_bands, table, cmap=plt.cm.inferno, levels=20)
    # to be continue...
    return umin_fit, T0_fit, chi2_fit #, table


def plot_fit_part_b(df, a):
    umin, T0, tau, f_bl = a

    t_left = df['days'].min()
    t_right = df['days'].max()
    t_arr = linspace(t_left, t_right, 10000)
    u_arr = u(umin, T0, tau, t_arr)
    mu_arr = mu(u_arr)
    I_by_Istar_arr_fit = I_by_Istar(mu_arr, f_bl)

    u_arr_predict = u(umin, T0, tau, df['days'])
    mu_arr_predict = mu(u_arr_predict)
    I_by_Istar_arr_fit_predict = I_by_Istar(mu_arr_predict, f_bl)

    fig, axes = plt.subplots(2, 1, figsize=(6, 9))

    axes[0].errorbar(df['days'], df['I/I_star'], yerr=df['I/I_star error'], fmt='o', ms=3)
    axes[0].plot(t_arr, I_by_Istar_arr_fit, c='red')
    axes[0].grid(1)
    axes[0].title.set_text('Nonlinear Fitting by Chi Squared')
    axes[0].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='$I/I_{*}$')

    axes[1].errorbar(df['days'], I_by_Istar_arr_fit_predict - df['I/I_star'], yerr=df['I/I_star error'], fmt='o', ms=3)
    axes[1].axhline(y=0, color='r', linestyle='-', zorder=5)
    axes[1].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='$F(x_i)-y_i$')
    axes[1].title.set_text('Nonlinear Fitting by Chi Squared Residuals')
    axes[1].grid(1)

    plt.tight_layout(h_pad=3.0)
    plt.show()


if __name__ == "__main__":
    # Photometry data file containing 5 columns: Hel.JD, I magnitude, magnitude error, seeing estimation (in pixels - 0.26"/pixel) and sky level.
    df = pd.read_csv('phot.dat', sep=' ',
                     names=['HJD', 'I-magnitude', 'Magnitude Error', 'Seeing Estimation', 'sky level'])
    # Cleaning the data
    # convert HJD to time in days and normalized it such that the peak of the curve will be at t = 0.
    M_star = average(df[df['HJD'] < 2.45841 * 10 ** 6]['I-magnitude'])
    M_star_error = sqrt(var(df[df['HJD'] < 2.45841 * 10 ** 6]['I-magnitude']))

    df['I/I_star'] = 10 ** ((df['I-magnitude'] - M_star) / -2.5)
    df['I/I_star error'] = 10 ** (-(1 / 2.5) * (df['I-magnitude'] - M_star)) * log(10) * (1 / 2.5) * \
                           sqrt(df['Magnitude Error'] ** 2 + M_star_error ** 2)

    time_where_mag_is_max = df['HJD'][argmax(df['I/I_star'])]
    # 2019-4-26 9:54:38.02     year-month-day hour:minute:second#
    df['days'] = df['HJD'] - time_where_mag_is_max
    print(f'{time_where_mag_is_max=}')

    # Part a

    # the form of the fitting that we want to minimize chi_squared in is : a + bx + cx^2
    # the unknown vector is : (a,b,c)

    # taking only points near the peak:
    smaller_df = pd.merge(df.loc[df['days'] < 25], df.loc[df['days'] > -25])
    N = len(smaller_df['days'])

    # the initial parameters when running chi squared for real data is:
    umin_theo = 0.919
    umin_theo_error = 0.003
    a, asig = chi_square_parbola_fit(smaller_df)
    initial_params = process_fit(a, asig)
    umin_initial = sqrt(-2 + 2 / sqrt(1 - 1 / initial_params[2] ** 2))
    umin_initial_error = (sqrt(-2 + 2 / sqrt(1 - initial_params[2] ** -2)) * (1 - 1 / initial_params[2] ** 2) ** 1.5 *
                          initial_params[2] ** 3) ** -1 * initial_params[3]
    print(umin_initial_error)
    print(f"{initial_params[0] + time_where_mag_is_max}")
    print(f"{initial_params[1]=}")
    N_sigma = abs(umin_initial - umin_theo) / sqrt(umin_theo_error ** 2 + umin_initial_error ** 2)
    print(f"Umin_initial_par is: \t{umin_initial}")
    print(f"N sigma is: \t{N_sigma}")

    plot_parabolic_fit_part_a(smaller_df, a)
    # now we use bootstrap method for 10k simulations:
    fits_df = bootstrap_part_a(smaller_df, 10 ** 3)
    plot_histogram_part_a(fits_df)

    # Part b
    # need some work...
    umin_low = 0.919 - 0.01
    umin_high = 0.919 + 0.01
    n_umin = 20
    T0_low = -5
    T0_high = +5
    n_T0 = 20
    tau = 42.826
    umin_fit, T0_fit, chi2_fit = chi_square_nonlinear_fit(df, umin_low, umin_high, n_umin, T0_low, T0_high, n_T0, tau)
    f_bl = 1
    a_arr = array([umin_fit, T0_fit, tau, f_bl])
    plot_fit_part_b(df, a_arr)
