#=============================================================
# tools for the annular mode index
#=============================================================
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import calendar
import datetime as dt
from netCDF4 import date2num, num2date

from LIM_AM_mod import Myla
cov_lag = Myla.cov_lag

#=============================================================
def find_month_ticks(year=1999, day=15, len_label=1):

    units = 'days since 1999-01-01'
    date_start = dt.datetime(year, 1, 1, 0)
    num_start = date2num(date_start, units=units)

    month_ticks = []
    month_ticklabels = []
    for month in range(1, 13):
        date_end = dt.datetime(year, month, day, 0)
        num_end = date2num(date_end, units=units)
        month_ticks.append(num_end-num_start+1)
        month_ticklabels.append(calendar.month_abbr[month][:len_label])

    return month_ticks, month_ticklabels

#=============================================================
def find_month_ticks_test():
    print('testing find_month_ticks() ......')
    year, day = 1999, 15
    month_ticks, month_ticklabels = find_month_ticks(year=year, day=day, len_label=3)
    for i in range(len(month_ticks)):
        print(month_ticklabels[i], f'{day}:', month_ticks[i])

#=============================================================
def find_idx_event(y10, threshold=-3, separation=30):
    """
    Input data: y10(years, days), where years and days are time \\
    output data: idx_event consisting of (years, days) of events identified based on `threshold` and `separation`
    """
    if y10.ndim == 1:
        y10 = y10[None, :]

    if threshold > 0:
        idx_threshold = np.argwhere(y10 > threshold)
    else:
        idx_threshold = np.argwhere(y10 < threshold)

    # selection of events
    idx_event = idx_threshold[0, :][None, :]    # first event
    for idx in idx_threshold[1:, :]:
        # print(idx_event, idx)
        # idx[0] denotes year, and idx[1] denotes day
        # record an event from next year or from the same year with the separation
        if (idx[0] > idx_event[-1,0]) or (idx[1] - idx_event[-1,1] > separation):   # next year or sufficiently seperated 
            idx_event = np.vstack((idx_event, idx))

    return idx_event

#=============================================================
def find_idx_event_test():
    from get_AM_mod import ERA5, CMIP6, JRA55, NAM, get_y

    print('testing find_idx_event() ......')

    plev = [850.,  700.,  500.,  250.,  100.,   50.,   10.]
    len_slice = 150
    offset = 40
    data = ERA5(name_dir='ERA5', year_start=1950, year_end=2014, plev=plev, source_dir='cmip6')
    D, p, y, t, y2 = get_y(data, 'NAM', len_slice=len_slice, offset=offset)    
    print(f't.shape = {t.shape}, y.shape = {y.shape}, y2.shape = {y2.shape}')

    threshold = -2
    # threshold = 1.5
    y10 = np.squeeze(y[:, :, p==10])
    idx_event = find_idx_event(y10, threshold, separation=30)

    print(f'# of events: {len(idx_event)}({len(idx_event)/len(y10):.2f})')
    units = 'days since 1800-01-01'
    calendar = D.data.calendar

    print(f'event #:\tdate\t\tvalue')
    for i in range(len(idx_event)):
        # date_event = pd.to_datetime(f'{1950+idx_event[i,0]}-11-01') + pd.Timedelta(days=idx_event[i,1])
        # print(f'event {i+1}:\t', date_event.date(), f'\t{y10[idx_event[i,0], idx_event[i,1]]:.2f}')
        date_start = dt.datetime(1950+idx_event[i,0], 11, 1, 0)
        num_start = date2num(date_start, units=units, calendar=calendar)
        date_event = num2date(num_start+idx_event[i,1], units=units, calendar=calendar)
        print(f'event {i+1}:\t', date_event.strftime("%Y-%m-%d"), f'\t{y10[idx_event[i,0], idx_event[i,1]]:.4f}')

#=============================================================
def find_event(y, p, y2=None, yf=None, threshold=-3, separation=30, lag_time=60, event_mean=True):
    """
    Input data: y(years, days, x), where years and days are time, and x is space
    p(x): pressure levels
    Output data: three options of composites based on y at 10 hPa
        1. composite of `y`; y is in the format of y[years, days, :];
        2. composite of `y2`; y2 is y with the padding of `lag_time` in the beginning and end in the format of y2[years, 2*lag_time+days, :];
        3. composite of yf; yf is the forecast of y in the format of y[years, lag_time, days, :].
    """
    
    # composite based on the annular mode index at p_level = 10 hPa
    p_level = 10
    k = np.isin(p, p_level)
    y10 = np.squeeze(y[:, :, k])
    # y10 = np.squeeze(y[:, :, p==p_level])
    if y10.ndim == 1:
        y10 = y10[None, :]

    idx_event = find_idx_event(y10, threshold, separation)

    # make composites
    if yf is None and y2 is None:
        # composite of y;
        y_event = np.zeros((0, lag_time+1, y.shape[2]))
        for idx in idx_event:
            yy, dd = idx[:]
            if dd+lag_time+1 <= y.shape[1]:
                y_event = np.vstack((y_event, y[yy, dd:dd+lag_time+1, :][None,:]))

        print(f'# of events: {len(y_event)}({len(y_event)/len(y):.2f})')
        if event_mean:
            return y_event.mean(axis=0), len(y_event)
        else:
            return y_event, len(y_event)
    elif yf is None:
        # composite of y2; y2 is y with the padding of `lag_time` in the beginning and end in the format of y[years, 2*lag_time+days, :]
        y2_event = np.zeros((0, lag_time*2+1, y.shape[2]))
        for idx in idx_event:
            yy, dd = idx[:]
            if dd+lag_time*2+1 <= y2.shape[1]:
                y2_event = np.vstack((y2_event, y2[yy, dd:dd+lag_time*2+1, :][None,:]))

        print(f'# of events: {len(y2_event)}({len(y2_event)/len(y2):.2f})')
        if event_mean:
            return y2_event.mean(axis=0), len(y2_event)
        else:
            return y2_event, len(y2_event)
    else:
        # composite of yf; yf is the forecast of y in the format of y[years, lag_time, days, :]
        yf_event = np.zeros((0, lag_time+1, yf.shape[3]))
        for idx in idx_event:
            yy, dd = idx[:]
            if dd+lag_time+1 <= y.shape[1]:
                yf_event = np.vstack((yf_event, yf[yy, :lag_time+1, dd, :][None,:]))

        # print(f'# of events: {len(yf_event)}')
        if event_mean:
            return yf_event.mean(axis=0), len(yf_event)
        else:
            return yf_event, len(yf_event)

#=============================================================
# main function
if __name__ == "__main__":
    find_idx_event_test()