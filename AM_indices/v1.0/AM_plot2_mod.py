#=============================================================
# Plotting functions for the annular mode index
#=============================================================
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#=============================================================
def cov_lag(X, lag_time, X2=None):
    """ Input: X(years, days, x), optional: X2(years, days, x2), where years+days is time, and x and x2 are space
        First calculate the covariance in the dimension of `days`, and then take the average along the dimension of `years`
        return squeeze(Ct(x, x2))
    """

    if X2 is None:
        X2 = X

    if X.ndim == 2:
        X = X[:, :, None]

    if X2.ndim == 2:
        X2 = X2[:, :, None]

    len_year, len_day, len_x = X.shape
    len_x2 = X2.shape[2]
    Ct = np.empty((len_year, len_x, len_x2))
    for n in range(len_year):
        Ct[n, :, :] = X[n, lag_time:, :].T @ X2[n, 0:len_day-lag_time, :] / (len_day-lag_time)

    return np.squeeze(Ct.mean(axis=0))


#=============================================================
# plot autocovariance
#=============================================================
def plot_cov(y, p, p_level, lag_time=60, ax1=None, model=None):

    k = np.isin(p, p_level).nonzero()[0][0]
    lags = np.linspace(0, lag_time, lag_time+1, dtype=int)
    Ct = np.array([cov_lag(y[:, :, k], lag) for lag in lags])

    # # fitting to an exp function
    from scipy.optimize import curve_fit
    func = lambda x, b: np.exp(-b * x)
    popt, pcov = curve_fit(func, lags, Ct/Ct[0])

    ax1.plot(lags, Ct/Ct[0], label=model)
    # ax1.plot(lags, func(lags, *popt), '--k', label=f'exp(-t/{1/popt[0]:.1f})')
    ax1.set_xlabel('Lag (days)')
    # ax1.legend()

    return popt


#=============================================================
# plot lagged covariance
#=============================================================
def plot_lag_reg(y, p, lag_time=40, model=None):

    kk = np.isin(p, 10).nonzero()[0][0]
    lags = np.linspace(0, lag_time, lag_time+1, dtype=int)
    Ct_p = np.array([cov_lag(y[:, :, :], lag, y[:, :, kk]) for lag in lags])
        
    cmax = 1
    cm = 'seismic' # 'bwr' 
    # fig = plt.figure(figsize=(12,5))
    # ax = fig.add_subplot(1,2,1)
    plt.contourf(lags, p, Ct_p.T, np.linspace(-cmax, cmax, 21), cmap=cm)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(-cmax, cmax, 11))
    plt.gca().invert_yaxis()
    # plt.xlabel('lag (days)')
    plt.ylabel('Pressure (hPa)')
    plt.yscale('log')
    plt.yticks([10, 30, 100, 300, 1000])
    plt.gca().get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.title(f"{model}")

#=============================================================
# plot composites for weak and strong vortex events
#=============================================================
def find_event(y, p, threshold=-3, separation=30, lag_time=60):
    """
    Input data: y(years, days, x), where years and days are time, and x is space
    p(x): pressure levels
    """
    
    # composite based on the annular mode index at p_level = 10 hPa
    p_level = 10
    kk = np.isin(p, p_level).nonzero()[0][0]
    y10 = y[:, :, kk]

    if threshold > 0:
        idx_threshold = np.argwhere(y10 > threshold)
    else:
        idx_threshold = np.argwhere(y10 < threshold)

    idx_event = idx_threshold[0, :][None, :]
    for idx in idx_threshold[1:, :]:
        # print(idx_event, idx)
        if (idx[0] > idx_event[-1,0]) or (idx[1] - idx_event[-1,1] > separation):
            idx_event = np.vstack((idx_event, idx))

    y_event = np.zeros((0, lag_time+1, y.shape[2]))
    for idx in idx_event:
        yy, dd = idx[:]
        if dd+lag_time+1 <= y.shape[1]:
            y_event = np.vstack((y_event, y[yy, dd:dd+lag_time+1, :][None,:]))

    return y_event

#=============================================================
# plot composites for weak and strong vortex events
#=============================================================
def plot_event(y, p, lag_time=60, event=None, model=None):
    """
    Input data: y(years, days, x), where years and days are time, and x is space
    p(x): pressure levels
    """

    lags = np.linspace(0, lag_time, lag_time+1)

    if event == 'negative':
        y_neg_event = find_event(y, p, threshold=-2, lag_time=lag_time)
    else:
        y_pos_event = find_event(y, p, threshold=1, lag_time=lag_time)

    cmax = 3
    cm = 'seismic' # 'bwr' 
    # fig = plt.figure(figsize=(12,10))
    # ax = fig.add_subplot(2,2,1)
    if event == 'negative':
        plt.contourf(lags, p, y_neg_event.mean(axis=0).T, np.linspace(-cmax, cmax, 21), cmap=cm)
        cbar = plt.colorbar()
        cbar.set_ticks(np.linspace(-cmax, cmax, 11))
        plt.gca().invert_yaxis()
        # plt.xlabel('lag (days)')
        plt.ylabel('Pressure (hPa)')
        plt.yscale('log')
        plt.yticks([10, 30, 100, 300, 1000])
        plt.gca().get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        plt.title(f'{model} ({len(y_neg_event)})')
    else:
        plt.contourf(lags, p, y_pos_event.mean(axis=0).T, np.linspace(-cmax, cmax, 21), cmap=cm)
        cbar = plt.colorbar()
        cbar.set_ticks(np.linspace(-cmax, cmax, 11))
        plt.gca().invert_yaxis()
        # plt.xlabel('lag (days)')
        plt.ylabel('Pressure (hPa)')
        plt.yscale('log')
        plt.yticks([10, 30, 100, 300, 1000])
        plt.gca().get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        plt.title(f'{model} ({len(y_pos_event)})')

def plot_event2(y, p, lag_time=60, event=None, model=None):
    """
    Input data: y(years, days, x), where years and days are time, and x is space
    p(x): pressure levels
    """

    lags = np.linspace(0, lag_time, lag_time+1)

    if event == 'negative':
        y_neg_event = find_event(y, p, threshold=-2, lag_time=lag_time)
    else:
        y_pos_event = find_event(y, p, threshold=1, lag_time=lag_time)

    # plot the evolution of annular mode at p_output
    p_output = 850 
    k = np.isin(p, p_output).nonzero()[0][0]

    if event == 'negative':
        # fig = plt.figure(figsize=(12,5))
        # ax = fig.add_subplot(1,2,1)
        if model == "JRA55":
            plt.plot(lags, y_neg_event.mean(axis=0)[:,k], color='k', linewidth=5, label=model)
        else:
            plt.plot(lags, y_neg_event.mean(axis=0)[:,k], label=model)
        plt.xlabel('lag (days)')
        # plt.title(f'weak vortex composite (p={p[k]})')
    else:
        # ax = fig.add_subplot(1,2,2)
        if model == "JRA55":
            plt.plot(lags, y_pos_event.mean(axis=0)[:,k], color='k', linewidth=5, label=model)
        else:
            plt.plot(lags, y_pos_event.mean(axis=0)[:,k], label=model)
        plt.xlabel('lag (days)')
        # plt.title(f'{model} ({len(y_pos_event)})')