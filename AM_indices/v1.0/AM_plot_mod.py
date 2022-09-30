#=============================================================
# Plotting functions for the annular mode index
#=============================================================
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#=============================================================
# plot autocovariance
#=============================================================
def plot_cov(y, yf=None, k=None, lag_time=60, model_name='LIM'):
    """
    y(t, x): input data, where t is time, and x is space
    """

    data_size = len(y)
    lags = np.linspace(0, lag_time, lag_time+1)

    if k is None:
        Ct = np.array([np.trace(y[lag:].T @ y[0:data_size-lag] / (data_size-lag)) for lag in np.arange(0, lag_time+1)])
    else:
        Ct = np.array([y[lag:, k].T @ y[0:data_size-lag, k] / (data_size-lag) for lag in np.arange(0, lag_time+1)])

    # fitting to an exp function
    from scipy.optimize import curve_fit
    func = lambda x, b: np.exp(-b * x)
    popt, pcov = curve_fit(func, lags, Ct/Ct[0])

    fig1 = plt.figure(figsize=(12,5))
    ax1 = fig1.add_subplot(1,2,1)
    ax1.plot(lags, Ct/Ct[0], '-k', label='Obs')
    ax1.plot(lags, func(lags, *popt), '--k', label=f'exp(-t/{1/popt[0]:.1f})')
    ax1.set_xlabel('Lag (days)')
    ax1.legend()

    if yf is not None:
        if k is None:
            Cft = np.array([np.trace(yf[lag,:,:].T @ yf[0,:,:] / yf.shape[1]) for lag in np.arange(0, lag_time+1)])
        else:
            Cft = np.array([yf[lag,:,k].T @ yf[0,:,k] / yf.shape[1] for lag in np.arange(0, lag_time+1)])

        # ax1 = fig1.add_subplot(1,2,2)
        ax1.plot(lags, Cft/Cft[0], '-r', label=f'{model_name}')
        # ax1.set_xlabel('Lag (days)')
        ax1.legend()

#=============================================================
# plot lagged covariance
#=============================================================
def plot_lag_reg(y, p, yf=None, lag_time=60, model_name='LIM'):
    """
    y(t, x): input data, where t is time, and x is space
    p(x): pressure levels
    """

    data_size = len(y)
    kk = np.isin(p, 10).nonzero()[0][0]
    lags = np.linspace(0, lag_time, lag_time+1)

    Ct_p = np.zeros((lag_time+1, y.shape[1]))
    for lag in range(lag_time+1):
        Ct_p[lag, :] = y[lag:, :].T @ y[0:data_size-lag, kk] / (data_size-lag)
        
    if yf is not None:
        Ct_p_f = np.zeros((lag_time+1, yf.shape[2]))
        for lag in range(lag_time+1):
            Ct_p_f[lag, :] = yf[lag, :, :].T @ yf[0, :, kk] / yf.shape[1]

    cmax = 1
    cm = 'seismic' # 'bwr' 
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,2,1)
    plt.contourf(lags, p, Ct_p.T, np.linspace(-cmax, cmax, 21), cmap=cm)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(-cmax, cmax, 11))
    plt.gca().invert_yaxis()
    plt.xlabel('lag (days)')
    plt.ylabel('Pressure (hPa)')
    plt.yscale('log')
    plt.yticks([1, 3, 10, 30, 100, 300, 1000])
    plt.gca().get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.title('regression to 10 hPa, Obs')

    if yf is not None:
        ax = fig.add_subplot(1,2,2)
        plt.contourf(lags, p, Ct_p_f.T, np.linspace(-cmax, cmax, 21), cmap=cm)
        cbar = plt.colorbar()
        cbar.set_ticks(np.linspace(-cmax, cmax, 11))
        plt.gca().invert_yaxis()
        plt.xlabel('lag (days)')
        plt.ylabel('Pressure (hPa)')
        plt.yscale('log')
        plt.yticks([1, 3, 10, 30, 100, 300, 1000])
        plt.gca().get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        plt.title(f'regression to 10 hPa, {model_name}')

#=============================================================
# plot composites for weak and strong vortex events
#=============================================================
def find_event(y, p, yf=None, threshold=-3, separation=30, lag_time=60):
    """
    y(t, x): input data, where t is time, and x is space
    p(x): pressure levels
    """
    
    # composite based on the annular mode index at p_level = 10 hPa
    p_level = 10
    kk = np.isin(p, p_level).nonzero()[0][0]
    y10 = y[:, kk]

    if threshold > 0:
        idx_threshold = np.argwhere(y10 > threshold)
    else:
        idx_threshold = np.argwhere(y10 < threshold)

    idx_event = idx_threshold[0]
    for idx in idx_threshold[1:]:
        if idx - idx_event[-1] > separation:
            idx_event = np.vstack((idx_event, idx))

    if yf is None:
        y_event = np.zeros((0, lag_time+1, y.shape[1]))
        for idx in idx_event:
            i = idx.item()
            if i+lag_time+1 <= len(y):
                y_event = np.vstack((y_event, y[i:i+lag_time+1, :][None,:]))
    
        return y_event
    else:
        yf_event = np.zeros((0, lag_time+1, yf.shape[2]))
        for idx in idx_event:
            i = idx.item()
            if i <= yf.shape[1]:
                yf_event = np.vstack((yf_event, yf[:lag_time+1, i, :][None,:]))
    
        return yf_event

def plot_event(y, p, yf=None, lag_time=60, model_name='LIM'):
    """
    y(t, x): input data, where t is time, and x is space
    p(x): pressure levels
    """

    lags = np.linspace(0, lag_time, lag_time+1)

    y_neg_event = find_event(y, p, threshold=-3, lag_time=lag_time)
    y_pos_event = find_event(y, p, threshold=1.5, lag_time=lag_time)

    if yf is not None:
        yf_neg_event = find_event(y, p, yf, threshold=-3, lag_time=lag_time)
        yf_pos_event = find_event(y, p, yf, threshold=1.5, lag_time=lag_time)

    cmax = 4
    cm = 'seismic' # 'bwr' 
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(2,2,1)
    plt.contourf(lags, p, y_neg_event.mean(axis=0).T, np.linspace(-cmax, cmax, 21), cmap=cm)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(-cmax, cmax, 11))
    plt.gca().invert_yaxis()
    plt.xlabel('lag (days)')
    plt.ylabel('Pressure (hPa)')
    plt.yscale('log')
    plt.yticks([1, 3, 10, 30, 100, 300, 1000])
    plt.gca().get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.title(f'weak vortex composite ({len(y_neg_event)}), Obs')

    ax = fig.add_subplot(2,2,2)
    plt.contourf(lags, p, y_pos_event.mean(axis=0).T, np.linspace(-cmax, cmax, 21), cmap=cm)
    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(-cmax, cmax, 11))
    plt.gca().invert_yaxis()
    plt.xlabel('lag (days)')
    plt.ylabel('Pressure (hPa)')
    plt.yscale('log')
    plt.yticks([1, 3, 10, 30, 100, 300, 1000])
    plt.gca().get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.title(f'strong vortex composite ({len(y_pos_event)}), Obs')

    if yf is not None:
        ax = fig.add_subplot(2,2,3)
        plt.contourf(lags, p, yf_neg_event.mean(axis=0).T, np.linspace(-cmax, cmax, 21), cmap=cm)
        cbar = plt.colorbar()
        cbar.set_ticks(np.linspace(-cmax, cmax, 11))
        plt.gca().invert_yaxis()
        plt.xlabel('lag (days)')
        plt.ylabel('Pressure (hPa)')
        plt.yscale('log')
        plt.yticks([1, 3, 10, 30, 100, 300, 1000])
        plt.gca().get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        plt.title(f'weak vortex composite ({len(y_neg_event)}), {model_name}')

        ax = fig.add_subplot(2,2,4)
        plt.contourf(lags, p, yf_pos_event.mean(axis=0).T, np.linspace(-cmax, cmax, 21), cmap=cm)
        cbar = plt.colorbar()
        cbar.set_ticks(np.linspace(-cmax, cmax, 11))
        plt.gca().invert_yaxis()
        plt.xlabel('lag (days)')
        plt.ylabel('Pressure (hPa)')
        plt.yscale('log')
        plt.yticks([1, 3, 10, 30, 100, 300, 1000])
        plt.gca().get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        plt.title(f'strong vortex composite ({len(y_pos_event)}), {model_name}')

    # plot the evolution of annular mode at p_output
    p_output = 850 
    k = np.isin(p, p_output).nonzero()[0][0]
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,2,1)
    plt.plot(lags, y_neg_event.mean(axis=0)[:,k], label='Obs')
    if yf is not None:
        plt.plot(lags, yf_neg_event.mean(axis=0)[:,k], label=f'{model_name}')
    plt.legend()
    plt.xlabel('lag (days)')
    plt.title(f'weak vortex composite (p={p[k]})')
    
    ax = fig.add_subplot(1,2,2)
    plt.plot(lags, y_pos_event.mean(axis=0)[:,k], label='Obs')
    if yf is not None:
        plt.plot(lags, yf_pos_event.mean(axis=0)[:,k], label=f'{model_name}')
    plt.legend()
    plt.xlabel('lag (days)')
    plt.title(f'strong vortex composite (p={p[k]})')
