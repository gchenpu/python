#=============================================================
# tools for the annular mode index
#=============================================================
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import calendar
import datetime as dt
from netCDF4 import date2num


#=============================================================
def find_month_ticks(year=1999, day=15):

    units = 'days since 1999-01-01'
    date_start = dt.datetime(year, 1, 1, 0)
    num_start = date2num(date_start, units=units)

    month_ticks = []
    month_ticklabels = []
    for month in range(1, 13):
        date_end = dt.datetime(year, month, day, 0)
        num_end = date2num(date_end, units=units)
        month_ticks.append(num_end-num_start+1)
        month_ticklabels.append(calendar.month_abbr[month][0])

    return month_ticks, month_ticklabels


#=============================================================
def cov_lag(X, lag_time, X2=None):
    """ Input: X(years, days, x), optional: X2(years, days, x2), where years and days are time, and x and x2 are space
        First calculate the covariance in the dimension of `days` using `matmul`, and then take the average along the dimension of `years`
        return Ct(x, x2)
    """
    
    if isinstance(X, np.ma.MaskedArray):
        X = X.data

    if X2 is None:
        X2 = X
    elif isinstance(X2, np.ma.MaskedArray):
        X2 = X2.data

    if X.ndim == 2:
        X = X[:, :, None]   # last dim of 'x' is size=1
    
    if X2.ndim == 2:
        X2 = X2[:, :, None]   # last dim of 'x' is size=1

    len_day = X.shape[1]
    X1 = np.transpose(X, (0, 2, 1))     # prepare for np.matmul or @; Note that @ does not seem to work for N-dim MaskedArray with N>=2
    Ct = X1[:, :, lag_time:] @ X2[:, 0:len_day-lag_time, :] / (len_day-lag_time)
    
    return np.squeeze(Ct.mean(axis=0))


#=============================================================
def find_event(y, p, yf=None, threshold=-3, separation=30, lag_time=60):
    """
    Input data: y(years, days, x), where years and days are time, and x is space
    p(x): pressure levels
    """
    
    # composite based on the annular mode index at p_level = 10 hPa
    p_level = 10
    k = np.isin(p, p_level)
    y10 = np.squeeze(y[:, :, k])

    if threshold > 0:
        idx_threshold = np.argwhere(y10 > threshold)
    else:
        idx_threshold = np.argwhere(y10 < threshold)

    idx_event = idx_threshold[0, :][None, :]    # first event
    for idx in idx_threshold[1:, :]:
        # print(idx_event, idx)
        if (idx[0] > idx_event[-1,0]) or (idx[1] - idx_event[-1,1] > separation):   # next year or sufficiently seperated 
            idx_event = np.vstack((idx_event, idx))

    if yf is None:
        y_event = np.zeros((0, lag_time+1, y.shape[2]))
        for idx in idx_event:
            yy, dd = idx[:]
            if dd+lag_time+1 <= y.shape[1]:
                y_event = np.vstack((y_event, y[yy, dd:dd+lag_time+1, :][None,:]))

        print(f'# of events: {len(y_event)}({len(y_event)/len(y):.2f})')
        return y_event.mean(axis=0), len(y_event)
    else:
        yf_event = np.zeros((0, lag_time+1, yf.shape[3]))
        for idx in idx_event:
            yy, dd = idx[:]
            if dd+lag_time+1 <= y.shape[1]:
                yf_event = np.vstack((yf_event, yf[yy, :lag_time+1, dd, :][None,:]))

        # print(f'# of events: {len(yf_event)}')
        return yf_event.mean(axis=0), len(yf_event)
