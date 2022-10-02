def find_month_ticks(year=1999, day=15):
    import calendar
    import datetime as dt
    from netCDF4 import date2num

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
