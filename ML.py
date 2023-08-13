import numpy as np
from numpy import array, linspace, max, min, argmax, argmin, sqrt, abs, exp, histogram, average, var, transpose, \
    meshgrid
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from scipy.optimize import curve_fit
from time import sleep
from tqdm import tqdm


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
    chi = 0
    for i in range(len(x)):
        chi = chi + (y[i]  - ( a[2] + a[1]*x[i] + a[0]*(x[i])**2 ))**2 / y_er[i]
    #print(chi/(len(x)-3)) #chi squared reduced
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
    print(param_optimised)

    return x_hist_2, param_optimised, param_covariance_matrix, x_data


def plot_histogram_part_a(df):
    x_hist_t0, params_t0, params_covar_t0, x_data_t0 = fit_gauss(df, 'T_0')
    x_hist_imax, params_imax, params_covar_imax, x_data_imax = fit_gauss(df, 'u_min')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(df['T_0'], 50)
    axes[0].plot(x_hist_t0, gauss(x_hist_t0, *params_t0), 'k', linewidth=2, label='Gaussian fit')
    axes[0].grid(1)
    axes[0].title.set_text('$T_0\ Histogram$')
    axes[0].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='Count')

    axes[1].hist(df['u_min'], 50)
    axes[1].plot(x_hist_imax, gauss(x_hist_imax, *params_imax), 'k', linewidth=2, label='Gaussian fit')
    axes[1].grid(1)
    axes[1].title.set_text('$u_{min}\ Histogram$')
    axes[1].set(xlabel='$u_{min}$', ylabel='Count')

    plt.show()


def u(umin, T0, tau, t):
    return sqrt(umin ** 2 + ((t - T0) / tau) ** 2)


def u_arr(df, umin, T0, tau):
    return array([sqrt(umin ** 2 + ((t - T0) / tau) ** 2) for t in df["days"]])


def mu(u):
    return (u ** 2 + 2) / (u * sqrt(u ** 2 + 4))


def I_by_Istar(mu, f_bl):
    return mu * f_bl + 1 - f_bl


def chi_square_nonlinear_fit(df,is_bootstrap, umin_low, umin_high, n_umin, T0_low, T0_high, n_T0, tau, f_bl=1):
    umin_bands = linspace(umin_low, umin_high, n_umin)
    T0_bands = linspace(T0_low, T0_high, n_T0)
    umin_v, T0_v = np.meshgrid(umin_bands, T0_bands)
    umin_flatten = umin_v.flatten()
    T0_flatten = T0_v.flatten()
    u_values = u_arr(df,umin_flatten, T0_flatten, tau)
    mu_values = mu(u_values)
    I_by_Istar_valus = I_by_Istar(mu_values, f_bl)
    chi2_flatten = array([(((df['I/I_star'] - I) / df['I/I_star error']) ** 2).sum() for I in I_by_Istar_valus.T])
    index_min_chi2 = chi2_flatten.argmin()

    chi2_fit = chi2_flatten[index_min_chi2]
    chi2_flatten = chi2_flatten - np.min(chi2_flatten)
    chi2_v = chi2_flatten.reshape(umin_v.shape)
    chi2_v = chi2_v - np.min(chi2_v)
    umin_fit, T0_fit= umin_flatten[index_min_chi2], T0_flatten[index_min_chi2]

    # extract u_min error and T_0 error:
    if is_bootstrap == 0 :
        i = index_min_chi2
        while chi2_flatten[i] < 1:
            i = i + 1
        u_plus_error = np.abs(umin_flatten[i] - umin_fit)

        i = index_min_chi2
        while chi2_flatten[i] < 1:
            i = i - 1
        u_minus_error = np.abs(umin_flatten[i] - umin_fit)
        print(f"{u_plus_error=}")
        print(f"{u_minus_error=}")

        i = index_min_chi2

        while chi2_flatten[i] < 1:
            i = i - n_umin
        T0_plus_error = np.abs(T0_flatten[i] - T0_fit)

        i = index_min_chi2

        while chi2_flatten[i] < 1:
            i = i + n_umin
        T0_minus_error = np.abs(T0_flatten[i] - T0_fit)
        print(f"{T0_plus_error=}")
        print(f"{T0_minus_error=}")

        return umin_fit, T0_fit, chi2_fit, T0_plus_error, T0_minus_error, u_plus_error, u_minus_error

    return umin_fit, T0_fit, chi2_fit


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
    axes[0].title.set_text('4D Nonlinear Fitting by Chi Squared')
    axes[0].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='$I/I_{*}$')

    axes[1].errorbar(df['days'], I_by_Istar_arr_fit_predict - df['I/I_star'], yerr=df['I/I_star error'], fmt='o', ms=3)
    axes[1].axhline(y=0, color='r', linestyle='-', zorder=5)
    axes[1].set(xlabel='$T_0$ [days since 26-4-2019]', ylabel='$F(x_i)-y_i$')
    axes[1].title.set_text('4D Nonlinear Fitting by Chi Squared Residuals')
    axes[1].grid(1)

    plt.tight_layout(h_pad=3.0)
    plt.show()


def plot_contour_part_b(xx,yy,chi2_2d_grid):
    plt.contourf(xx, yy, chi2_2d_grid, cmap=plt.cm.inferno, levels=[0, 2.3, 4.61, 6.17, 9.21])
    plt.title("2D contours $T_0$ vs. $u_{min}$ and confidence levels")
    plt.xlabel("$u_{min}$")
    plt.ylabel("$T_0$ [days since 5-4-2019]")
    plt.grid()
    plt.colorbar()
    plt.show()


def bootstrap_part_b(df,is_bootstrap,num_simulations,umin_low, umin_high, n_umin, T0_low, T0_high, n_T0, tau, f_bl=1):
    fits = []
    for i in tqdm(range(num_simulations)):
        sleep(3)
        # Sample a new dataset
        data = df.loc[np.random.choice(df.index.values, len(df))]
        umin_fit, T0_fit ,chi2_fit = chi_square_nonlinear_fit(data,is_bootstrap,umin_low, umin_high, n_umin, T0_low, T0_high, n_T0, tau, f_bl=1)
        fits.append([umin_fit,T0_fit])
    fits_df = pd.DataFrame(fits, columns=["u_min","T_0"])
    return fits_df



def plot_t_u_histogram_part_b(df):

    pass


def chi_square_nonlinear_fit_4d(df,is_bootstrap, umin_low, umin_high, n_umin, T0_low, T0_high, n_T0, tau_low,tau_high,n_tau,fbl_low,fbl_high,n_fbl):
    umin_bands = linspace(umin_low, umin_high, n_umin)
    T0_bands = linspace(T0_low, T0_high, n_T0)
    tau_bands = linspace(tau_low, tau_high, n_tau)
    fbl_bands = linspace(fbl_low, fbl_high, n_fbl)

    umin_v, T0_v , tau_v, fbl_v = np.meshgrid(umin_bands, T0_bands,tau_bands,fbl_bands)
    umin_flatten = umin_v.flatten()
    T0_flatten = T0_v.flatten()
    tau_flatten = tau_v.flatten()
    fbl_flatten = fbl_v.flatten()

    u_values = u_arr(df, umin_flatten, T0_flatten, tau_flatten)
    mu_values = mu(u_values)
    I_by_Istar_valus = I_by_Istar(mu_values, fbl_flatten)
    chi2_flatten = array([(((df['I/I_star'] - I) / df['I/I_star error']) ** 2).sum() for I in I_by_Istar_valus.T])
    index_min_chi2 = chi2_flatten.argmin()
    chi2_fit = chi2_flatten[index_min_chi2]
    chi2_v = chi2_flatten.reshape(umin_v.shape)

    umin_min_index , T0_min_index , tau_min_index , fbl_min_index = np.unravel_index(chi2_v.argmin(), chi2_v.shape)

    umin_fit, T0_fit, tau_fit, fbl_fit = umin_flatten[index_min_chi2], T0_flatten[index_min_chi2], tau_flatten[
        index_min_chi2], fbl_flatten[index_min_chi2]


    return umin_fit,T0_fit,tau_fit,fbl_fit , chi2_fit



def plot_contours_part_c(umin_high,umin_low,T0_high,T0_low,tau_high,tau_low,fbl_high,fbl_low,n, T0_vs_tau,T0_vs_fbl,T0_vs_umin,tau_vs_umin,tau_vs_fbl,fbl_vs_umin):
    umin_bands = np.linspace(umin_low, umin_high, n)
    T0_bands = np.linspace(T0_low,T0_high,n)
    tau_bands = np.linspace(tau_low, tau_high, n)
    fbl_bands = np.linspace(fbl_low, fbl_high, n)

    T0_1 , tau_1 = meshgrid(T0_bands,tau_bands)
    T0_2 , fbl_1 = meshgrid(T0_bands,fbl_bands)
    T0_3 , umin_1 = meshgrid(T0_bands,umin_bands)
    tau_2 , umin_2 = meshgrid(tau_bands,umin_bands)
    tau_3 , fbl_2 = meshgrid(tau_bands,fbl_bands)
    fbl_3 , umin_3 = meshgrid(fbl_bands,umin_bands)

    fig, axs = plt.subplots(3, 3, figsize=(7, 7))
    fig.delaxes(axs[0][1])
    fig.delaxes(axs[0][2])
    fig.delaxes(axs[1][2])

    axs[0,0].contourf(T0_1, tau_1, T0_vs_tau, cmap=plt.cm.inferno, levels=[0, 2.3, 4.61, 6.17, 9.21])
    axs[1, 0].contourf(T0_2,fbl_1 , T0_vs_fbl,cmap=plt.cm.inferno,levels=[0, 2.3, 4.61, 6.17, 9.21])
    axs[2, 0].contourf(T0_3, umin_1, T0_vs_umin, cmap=plt.cm.inferno, levels=[0, 2.3, 4.61, 6.17, 9.21])
    axs[2, 1].contourf(tau_2, umin_2, tau_vs_umin, cmap=plt.cm.inferno, levels=[0, 2.3, 4.61, 6.17, 9.21])
    axs[1, 1].contourf(tau_3, fbl_2, tau_vs_fbl, cmap=plt.cm.inferno, levels=[0, 2.3, 4.61, 6.17, 9.21])
    axs[2, 2].contourf(fbl_3, umin_3, fbl_vs_umin, cmap=plt.cm.inferno, levels=[0, 2.3, 4.61, 6.17, 9.21])

    axs[0, 0].set_ylabel("$\\tau\ [day]$")
    axs[1, 0].set_ylabel("$f_{bl}$")
    axs[2, 0].set_ylabel("$u_{min}$")
    axs[2, 0].set_xlabel("$T_0$ [days since 5-4-2019]")
    axs[2, 1].set_xlabel("$\\tau\ [day]$")
    axs[2, 2].set_xlabel("$f_{bl}$")

    axs[0, 0].set_xticks([])
    axs[1, 0].set_xticks([])
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[2, 1].set_yticks([])
    axs[2, 2].set_yticks([])

    #fig.tight_layout()
    cax = plt.axes([0.9, 0.75, 0.02, 0.2])
    fig.colorbar(plt.contourf(tau_2, umin_2, tau_vs_umin, cmap=plt.cm.inferno, levels=[0, 2.3, 4.61, 6.17, 9.21]), cax=cax)
    fig.suptitle("Corner Plot of Confidence Levels")
    plt.show()


def improve_chi_squared():
    pass

def general_2D_contour_data(x_high,x_low,nx,y_high,y_low,ny,index,fixed_param1,fixed_param2):
    global I_by_Istar_valus
    x_bands = linspace(x_low, x_high, nx)
    y_bands = linspace(y_low, y_high, ny)
    x_v, y_v = np.meshgrid(x_bands, y_bands)
    x_flatten = x_v.flatten()
    y_flatten = y_v.flatten()

    if (index == 0):
        u_values = u_arr(df, fixed_param1, x_flatten, y_flatten)
        mu_values = mu(u_values)
        I_by_Istar_valus = I_by_Istar(mu_values, fixed_param2)

    if (index == 1):
        u_values = u_arr(df, fixed_param1, x_flatten, fixed_param2)
        mu_values = mu(u_values)
        I_by_Istar_valus = I_by_Istar(mu_values, y_flatten)

    if (index == 2):
        u_values = u_arr(df, y_flatten, x_flatten, fixed_param1)
        mu_values = mu(u_values)
        I_by_Istar_valus = I_by_Istar(mu_values, fixed_param2)

    if (index == 3):
        u_values = u_arr(df, y_flatten, fixed_param1, x_flatten)
        mu_values = mu(u_values)
        I_by_Istar_valus = I_by_Istar(mu_values, fixed_param2)

    if (index == 4):
        u_values = u_arr(df, fixed_param1, fixed_param2, x_flatten)
        mu_values = mu(u_values)
        I_by_Istar_valus = I_by_Istar(mu_values, y_flatten)

    if (index == 5):
        u_values = u_arr(df, y_flatten, fixed_param1, fixed_param2)
        mu_values = mu(u_values)
        I_by_Istar_valus = I_by_Istar(mu_values, x_flatten)

    chi2_flatten = array([(((df['I/I_star'] - I) / df['I/I_star error']) ** 2).sum() for I in I_by_Istar_valus.T])
    index_min_chi2 = chi2_flatten.argmin()

    chi2_fit = chi2_flatten[index_min_chi2]
    chi2_flatten = chi2_flatten - np.min(chi2_flatten)
    chi2_v = chi2_flatten.reshape(x_v.shape)
    chi2_v = chi2_v - np.min(chi2_v)
    x_fit, y_fit = x_flatten[index_min_chi2], y_flatten[index_min_chi2]

    #extract 1D errors:

    i = index_min_chi2
    while chi2_flatten[i] < 1:
        i = i + 1
    x_plus_error = np.abs(x_flatten[i] - x_fit)

    i = index_min_chi2
    while chi2_flatten[i] < 1:
        i = i - 1
    x_minus_error = np.abs(x_flatten[i] - x_fit)

    i = index_min_chi2

    while chi2_flatten[i] < 1:
        i = i - nx
    y_plus_error = np.abs(y_flatten[i] - y_fit)

    i = index_min_chi2

    while chi2_flatten[i] < 1:
        i = i + nx
    y_minus_error = np.abs(y_flatten[i] - y_fit)

    return (x_v,y_v,chi2_v,x_minus_error,x_plus_error,y_minus_error,y_plus_error)


if __name__ == "__main__":
    # Photometry data file containing 5 columns: Hel.JD, I magnitude, magnitude error, seeing estimation (in pixels - 0.26"/pixel) and sky level.
    df = pd.read_csv('phot1.dat', sep=' ',
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

    #print(f'{time_where_mag_is_max=}')

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

    #print(f"{initial_params[0] + time_where_mag_is_max}")
    #print(f"{initial_params[1]=}")
    N_sigma = abs(umin_initial - umin_theo) / sqrt(umin_theo_error ** 2 + umin_initial_error ** 2)
    #print(f"Umin_initial_par is: \t{umin_initial}")
    #print(f"Umin_initial_par is: \t{umin_initial}")

    #print(f"N sigma is: \t{N_sigma}")

    plot_parabolic_fit_part_a(smaller_df, a)
    #now we use bootstrap method for 10k simulations:
    fits_df = bootstrap_part_a(smaller_df, 10 ** 4)
    plot_histogram_part_a(fits_df)

    # Part b
    umin_low = 0.9
    umin_high = 0.95
    n_umin = 250
    T0_low = -2.2
    T0_high = 2.2
    n_T0 = 250
    #tau = 48.3
    tau = 42.826
    umin_fit, T0_fit, chi2_fit = chi_square_nonlinear_fit(df,1,umin_low, umin_high, n_umin, T0_low, T0_high, n_T0, tau)

    f_bl = 1
    a_arr = array([umin_fit, T0_fit, tau, f_bl])
    plot_fit_part_b(df, a_arr)
    fits_df_part_b = bootstrap_part_b(df,1,1000,umin_low,umin_high,n_umin,T0_low,T0_high,n_T0,tau)
    plot_histogram_part_a(fits_df_part_b)

    # part c
    tau_low = tau - 1.1
    tau_high = tau + 0.7
    n_tau = 10
    fbl_low = 0.51 - 0.014
    fbl_high = 0.51 + 0.017
    n_fbl = 10

    # more resolution grid for corner plot
    nT0 = 100
    ntau = 100
    numin = 100
    nfbl = 100

    Nonlinear_4D_params = chi_square_nonlinear_fit_4d(df, 0, umin_low, umin_high, n_umin, T0_low, T0_high, n_T0, tau_low, tau_high,
                                n_tau, fbl_low, fbl_high, n_fbl)

    print(Nonlinear_4D_params[1]+time_where_mag_is_max)
    print(Nonlinear_4D_params[4]/(len(df["days"])-4))

    T0_and_tau = general_2D_contour_data(T0_high,T0_low,nT0,tau_high,tau_low,ntau,0,Nonlinear_4D_params[0],Nonlinear_4D_params[3])
    T0_and_fbl = general_2D_contour_data(T0_high,T0_low,nT0,fbl_high,fbl_low,nfbl,1,Nonlinear_4D_params[0],Nonlinear_4D_params[2])
    T0_and_umin = general_2D_contour_data(T0_high,T0_low,nT0,umin_high,umin_low,numin,2,Nonlinear_4D_params[2],Nonlinear_4D_params[3])
    tau_and_umin = general_2D_contour_data(tau_high,tau_low,ntau,umin_high,umin_low,numin,3,Nonlinear_4D_params[1],Nonlinear_4D_params[3])
    tau_and_fbl = general_2D_contour_data(tau_high,tau_low,ntau,fbl_high,fbl_low,nfbl,4,Nonlinear_4D_params[0],Nonlinear_4D_params[1])
    fbl_and_umin = general_2D_contour_data(fbl_high,fbl_low,nfbl,umin_high,umin_low,numin,5,Nonlinear_4D_params[1],Nonlinear_4D_params[2])
    plot_contours_part_c(umin_high,umin_low,T0_high,T0_low,tau_high,tau_low,fbl_high,fbl_low,numin,T0_and_tau[2],T0_and_fbl[2],T0_and_umin[2],tau_and_umin[2],tau_and_fbl[2],fbl_and_umin[2])

    print("T0 4D plus error is:",T0_and_tau[4])
    print("T0 4D minus error is:", T0_and_tau[3])
    print("tau 4D plus error is:",T0_and_tau[6])
    print("tau 4D minus error is:", T0_and_tau[5])
    print("fbl 4D plus error is:", fbl_and_umin[4])
    print("fbl 4D minus error is:", fbl_and_umin[3])
    print("umin 4D plus error is:", fbl_and_umin[6])
    print("umin 4D minus error is:", fbl_and_umin[5])

    plot_fit_part_b(df, Nonlinear_4D_params[0:4])


