#=============================================================
# Module to get the AM indices from reanalyses and cmip models
#=============================================================
import numpy as np
import os

import netCDF4
from netCDF4 import Dataset
from netCDF4 import date2num, num2date

import calendar
import datetime as dt

#=============================================================
# MyDataSet class
#=============================================================
class MyDataSet():    
    def __init__(self, name_dir, year_start, year_end, name=None, source_dir=None):
        """
        Basic information of the data set: \\
        data directory: `source_dir` + `/` + `name_dir` \\
        period: from `year_start` to `year_end`        \\
        name: name of the dataset; default=`name_dir`
        """

        self.name_dir = name_dir
        self.year_start = year_start
        self.year_end = year_end
        self.num_years = year_end - year_start + 1

        if name:                            # specify dataset name
            self.name = name
        else:
            self.name = name_dir
                
        self.root_dir = '.'                 # or use os.getcwd()
        if source_dir:
            self.source_dir = self.root_dir + '/' + source_dir    # specify source_dir
        else:
            self.source_dir = self.root_dir    # use cwd as source_dir

#=============================================================
# JRA55 DataSet class
#=============================================================
class JRA55(MyDataSet):
    def __init__(self, name_dir, year_start, year_end, plev=None, name=None, source_dir=None):
        """
        Initializing attributes and methods for JRA55 @ python/AM_indices
        The input data is 4xdaily, and the output is daily average.
        """

        super().__init__(name_dir, year_start, year_end, name, source_dir)

        file = self.source_dir + '/' + self.name_dir + '/MODES_4xdaily_2007_01.nc'
        ncfile = Dataset(file, 'r')

        level_input = ncfile.variables['pressure'][:].astype(np.float32)
        if ncfile.variables['pressure'].units == 'Pa':   # converting from Pa to hPa
            level_input /= 100
        if plev:
            self.level = plev
        else:
            self.level = level_input            # use level_input as self.level if self.level is not specified
        if (level_input[1]-level_input[0])*(self.level[1]-self.level[0]) < 0:
            self.level = np.flip(self.level, axis=0)    # adjust the direction of variations in self.level according to level_input
        self.level_index = np.isin(level_input, self.level)   # indices in level_input for elements in self.level
        self.num_levels = len(self.level)
        # print(f'Pressure input: {level_input}\n Pressure output: {self.level}\n Pressure levels used: {level_input[self.level_index]}\n')

        # http://cfconventions.org/cf-conventions/cf-conventions#calendar
        self.calendar = '365_day'
        self.length_of_year = 365

    def read_data(self, var_name):
        """
        Read `var_name` from `year_start` to `year_end`.     \\
        return var(year, days_in_a_year, plev), where year and days_in_a_year are time, and plev is pressure.
        """

        var = np.empty((0, self.length_of_year, self.num_levels), np.float32)    # dim(year, days_in_a_year, pressure)
        for year in range(self.year_start, self.year_end+1):
            var = np.vstack((var, self.read_data_1year(year, var_name)[None, :]))

        return var  # dim(year, days_in_a_year, pressure)

    def read_data_1year(self, year, var_name):
        """
        read 4xdaily data for var_name and concatenate into the data in a year.
            2/29 in a leap year is discarded
        return var_year(days_in_a_year, pressure)
        """

        if year < self.year_start or year > self.year_end:
            raise Exception('Year out of range!')
        
        var_year = np.empty((0, self.num_levels), np.float32)    # dim(days_in_a_year, pressure)
        for month in range(1,13):
            file = self.source_dir + '/' + self.name_dir + '/' + f'MODES_4xdaily_{year}_{str(month).zfill(2)}.nc'
            ncfile = Dataset(file, 'r')
            var_4xdaily = ncfile.variables[var_name][:, self.level_index]

            var_tmp = 0.25*(var_4xdaily[0::4, :] +var_4xdaily[1::4, :]
                           +var_4xdaily[2::4, :] +var_4xdaily[3::4, :])
                
            if calendar.isleap(year) and month==2:
                # remove the last day of Feburary in a leap year
                var_year = np.vstack((var_year, var_tmp[:-1, :]))
            else:
                var_year = np.vstack((var_year, var_tmp[:]))

        return var_year  # dim(days_in_a_year, pressure)

#=============================================================
# CMIP6 DataSet class
#=============================================================
class CMIP6(MyDataSet):
    def __init__(self, name_dir, year_start, year_end, plev=None, name=None, source_dir=None):
        """
        Initializing attributes and methods for CMIP6 @python/cmip6
        Both the input and output data are daily.
        """

        super().__init__(name_dir, year_start, year_end, name, source_dir)
        
        file = self.source_dir + '/' + self.name_dir + '/AM_daily_1950_2014.nc'
        ncfile = Dataset(file, 'r')

        level_input = ncfile.variables['plev'][:].astype(np.float32)
        if ncfile.variables['plev'].units == 'Pa':   # converting from Pa to hPa
            level_input /= 100
        if plev:
            self.level = plev
        else:
            self.level = level_input            # use level_input as self.level if self.level is not specified
        if (level_input[1]-level_input[0])*(self.level[1]-self.level[0]) < 0:
            self.level = np.flip(self.level, axis=0)    # adjust the direction of variations in self.level according to level_input
        self.level_index = np.isin(level_input, self.level)   # indices in level_input for elements in self.level
        self.num_levels = len(self.level)
        # print(f'Pressure input: {level_input}\n Pressure output: {self.level}\n Pressure levels used: {level_input[self.level_index]}\n')

        # http://cfconventions.org/cf-conventions/cf-conventions#calendar
        # 365_day or 360_day calendar
        try:
            calendar = ncfile.variables['time'].calendar
        except:
            calendar = ncfile.variables['time2'].calendar
        if calendar:
            # print(f"Calendar of {self.name} is {calendar}")
            if calendar == "365_day" or calendar == "noleap":
                self.calendar = '365_day'
                self.length_of_year = 365
            elif calendar == "360_day":
                self.calendar = '360_day'
                self.length_of_year = 360
            else:
                self.calendar = '365_day'
                self.length_of_year = 365
        else:
            raise Exception('Calendar is not specified!')
        
    def read_data(self, var_name):
        """
        Read `var_name` from `year_start` to `year_end`.     \\
        return var(year, days_in_a_year, plev), where year and days_in_a_year are time, and plev is pressure.
        """

        cmip6_start = 1950
        cmip6_end = 2014
        if self.year_start < cmip6_start or self.year_end > cmip6_end:
            raise Exception('Year out of range!')

        ys = self.year_start - cmip6_start
        ye = self.year_end - cmip6_start
        ylen = self.length_of_year

        file = self.source_dir + '/' + self.name_dir + '/AM_daily_1950_2014.nc'
        # print(file)
        # print(f"{self.name}, {self.level_index}")
        ncfile = Dataset(file, 'r')
        if var_name == 'GLOBAL':
            var_o = ncfile.variables['Global'][ys*ylen:(ye+1)*ylen, self.level_index]        # correct the variable name for CMIP6
        else:
            var_o = ncfile.variables[var_name][ys*ylen:(ye+1)*ylen, self.level_index]        # dim(year*days_in_a_year, pressure)
        # print(f'Length of {var_name} in {self.name}: {var_o.shape[0]/self.length_of_year} years')
        var = var_o.reshape(-1, ylen, self.num_levels)    # dim(year, days_in_a_year, pressure)

        return var

#=============================================================
# ERA5 DataSet class
#=============================================================
class ERA5(MyDataSet):
    def __init__(self, name_dir, year_start, year_end, plev=None, name=None, source_dir=None):
        """
        Initializing attributes and methods for ERA5 @python/cmip6
        Both the input and output data are daily. Data already removed 2/29 in a leap year
        """

        super().__init__(name_dir, year_start, year_end, name, source_dir)

        file = self.source_dir + '/' + self.name_dir + '/AM_daily_1950_2021.nc'
        ncfile = Dataset(file, 'r')

        level_input = ncfile.variables['level'][:].astype(np.float32)
        if ncfile.variables['level'].units == 'Pa':   # converting from Pa to hPa
            level_input /= 100
        if plev:
            self.level = plev
        else:
            self.level = level_input            # use level_input as self.level if self.level is not specified
        if (level_input[1]-level_input[0])*(self.level[1]-self.level[0]) < 0:
            self.level = np.flip(self.level, axis=0)    # adjust the direction of variations in self.level according to level_input
        self.level_index = np.isin(level_input, self.level)   # indices in level_input for elements in self.level
        self.num_levels = len(self.level)
        # print(f'Pressure input: {level_input}\n Pressure output: {self.level}\n Pressure levels used: {level_input[self.level_index]}\n')

        # http://cfconventions.org/cf-conventions/cf-conventions#calendar
        self.calendar = '365_day'
        self.length_of_year = 365

    def read_data(self, var_name):
        """
        Read `var_name` from `year_start` to `year_end`.     \\
        return var(year, days_in_a_year, plev), where year and days_in_a_year are time, and plev is pressure.
        """

        era5_start = 1950
        era5_end = 2021
        if self.year_start < era5_start or self.year_end > era5_end:
            raise Exception('Year out of range!')

        ys = self.year_start - era5_start
        ye = self.year_end - era5_start
        ylen = self.length_of_year

        file = self.source_dir + '/' + self.name_dir + '/AM_daily_1950_2021.nc'
        # print(file)
        # print(f"{self.name}, {self.level_index}")
        ncfile = Dataset(file, 'r')
        if var_name == 'GLOBAL':
            var_o = ncfile.variables['Global'][ys*ylen:(ye+1)*ylen, self.level_index]        # correct the variable name for CMIP6
        else:
            var_o = ncfile.variables[var_name][ys*ylen:(ye+1)*ylen, self.level_index]        # dim(year*days_in_a_year, pressure)
        # print(f'Length of {var_name} in {self.name}: {var_o.shape[0]/self.length_of_year} years')
        var = var_o.reshape(-1, ylen, self.num_levels)    # dim(year, days_in_a_year, pressure)

        return var

#=============================================================
# tests of MyDataSet class
#=============================================================
def MyDataSet_test():
    print("Testing MyDataSet class ......")

    data = JRA55(name_dir='jra_55', year_start=1958, year_end=2016, name='JRA55')
    nam = data.read_data('NAM')
    print(f'{data.name}:', nam.shape)

    data = CMIP6(name_dir='CESM2', year_start=1950, year_end=2014, source_dir='cmip6')
    nam = data.read_data('NAM')
    print(f'{data.name}:', nam.shape)

    data = ERA5(name_dir='ERA5', year_start=1950, year_end=2021, source_dir='cmip6')
    nam = data.read_data('NAM')
    print(f'{data.name}:', nam.shape)

#=============================================================
# MyGetData class
#=============================================================
class MyGetData():
    def __init__(self, data, annual_cycle_fft=3, running_mean=0):
        """
        Use the method `read_data` in data (class: MyDataSet)` \\
        Key options: \\
        annual_cycle_fft: option to remove the harmonics above `annual_cycle_fft` from the annual cycle \\
        running_mean: option to conduct running average for the anomaly with `running_mean`
        """

        self.data = data
        self.annual_cycle_fft = annual_cycle_fft
        self.running_mean = running_mean

    def get_data(self, var_name):

        # print(f"{self.data.name}: {var_name}")
        var = self.data.read_data(var_name)    # use the method  `read_data`in data (object of class: MyDataSet)
        if isinstance(var, np.ma.MaskedArray):
            return var.data     # get the data from masked array
        else:
            return var

    def get_anomaly(self, var_name):
        """
        calculate anomalies and remove the annual cycle and apply running mean as needed \\
        Input: var(year, days_in_a_year, :), where year and days_in_a_year are time, and `:` is space \\
        return: \\
            var_mean_o(days_in_a_year, :),    # annual cycle, option to remove the harmonics above `annual_cycle_fft` \\
            var_anomaly_o(year, days_in_a_year, :)    # anomaly, option to conduct running average with `running_mean`
        """

        var = self.get_data(var_name)

        var_mean_o = MyGetData.cal_annual_cycle(var, self.annual_cycle_fft)

        var_anomaly = (var - var_mean_o).reshape(-1, var.shape[2])    # broadcasting the 1st dimension
        var_anomaly_o = MyGetData.cal_running_mean(var_anomaly, self.running_mean)

        return var_mean_o, var_anomaly_o.reshape(-1, var.shape[1], var.shape[2])

    def get_slice(self, var_i, month_start, len_slice, slice_offset=0):
        """
        Slice `var_i` each year starting from `month_start` by the length of 'len_slice' \\
        Input: var_i(year, days_in_a_year, :), where year and days_in_a_year are time, and `:` is space \\
        offset: add data before `month_start` and after 'len_slice' with the length of `offset`
        return  my_slice(year, len_slice+2*slice_offset, :)
        """

        var = var_i.reshape(-1, var_i.shape[2])

        self.slice_offset = slice_offset
        units = 'days since 1800-01-01'
        calendar = self.data.calendar
        date_start = dt.datetime(self.data.year_start, 1, 1, 0)
        num_start = date2num(date_start, units=units, calendar=calendar)

        my_slice = np.empty((0, len_slice+2*slice_offset, var.shape[1]))
        for y in range(self.data.year_start, self.data.year_end+1):
            date_end = dt.datetime(y, month_start, 1, 0)
            num_end = date2num(date_end, units=units, calendar=calendar)
            slice_start = num_end - num_start
            # print(y, slice_start, len_slice, var.shape[0])
            if slice_start+len_slice+slice_offset <= var.shape[0]:
                my_slice = np.vstack((my_slice, var[slice_start-slice_offset:slice_start+len_slice+slice_offset, :][None,:,:]))

        my_slice /= my_slice[:,slice_offset:len_slice+slice_offset,:].reshape(-1, var.shape[1]).std(axis=0, dtype=np.float64)    # `slice_offset` is not included for calculating std

        return my_slice 

    @staticmethod
    def cal_annual_cycle(var, annual_cycle_fft=4):
        """ 
        Input: var(year, days_in_a_year, :), where year and days_in_a_year are time, and : is space \\
        calculate the annual cycle of `var` by averaging in year \\
        annual_cycle_fft: option to remove the harmonics above `annual_cycle_fft` from the annual cycle
        """

        var_mean = var.mean(axis=0)
        var_mean_o = np.empty_like(var_mean)

        if annual_cycle_fft > 0:
            var_mean_fft = np.fft.fft(var_mean, axis=0)
            var_mean_fft[annual_cycle_fft+1:-annual_cycle_fft, :] = 0
            var_mean_o = (np.fft.ifft(var_mean_fft, axis=0)).real
        else:
            var_mean_o = var_mean

        return var_mean_o

    @staticmethod
    def cal_running_mean(var, running_mean=0):
        """ 
        Input: var(days, :), where days is time, and : is space \\
        calculate the running mean of `var` along axis=0 \\
        running_mean: smoothing with `running_mean` (using `np.convolve`)
        """

        var_o = np.empty_like(var)
        if running_mean > 0:
            for k in range(var.shape[1]):
                var_o[:, k] = np.convolve(var[:, k], np.ones((running_mean,))/running_mean, mode='same')
        else:
            var_o = var

        return var_o

#=============================================================
# AM index class
#=============================================================
class AM(MyGetData):
    def __init__(self, data, index_name, annual_cycle_fft=3, running_mean=0, save_index=False):

        super().__init__(data, annual_cycle_fft, running_mean)

        if index_name != 'NAM' and index_name != 'SAM':
            raise Exception(f"Cannot recognize index_name={index_name}")
         
        self.index_name = index_name

        if self.annual_cycle_fft == 3 and self.running_mean == 0:      # default values
            tag = ""
        else:
            tag = f"_fft{self.annual_cycle_fft}_rm{self.running_mean}"

        self.file_save = self.data.root_dir + '/data/' + self.data.name_dir + f'/{self.index_name}_daily_' \
                         + str(self.data.year_start) + '_' + str(self.data.year_end) + tag + '.npy'

        # read, calculate, and save data in self.index
        if os.path.exists(self.file_save):
            print('Reading from saved data ......')
            with open(self.file_save, 'rb') as f:
                self.index = np.load(f)

        else:
            print('Calculating from the original data .......')
            self.index = self.cal_index()
            if save_index:
                if not os.path.exists(os.path.dirname(self.file_save)):
                    os.makedirs(os.path.dirname(self.file_save))
                with open(self.file_save, 'wb') as f:
                    np.save(f, self.index.astype(np.float32))

    def cal_index(self):
        """
        Calculate the NAM indices from pre-processed geopotential height averaged over the polar caps. \\
        remove the global mean and standardize the anomaly time series  \\
        return NAM_index(year*days_in_a_year, pressure)

        References
        Gerber, E. P., and P. Martineau, 2018: Quantifying the variability of the annular modes: 
        Reanalysis uncertainty vs. sampling uncertainty. Atmos. Chem. Phys., 18, 17099–17117, https://doi.org/10.5194/acp-18-17099-2018.
        """

        _, AM_anomaly = self.get_anomaly(self.index_name)
        _, GLOBAL_anomaly = self.get_anomaly('GLOBAL')

        AM_index = -(AM_anomaly-GLOBAL_anomaly)
        AM_index /= AM_index.reshape(-1, AM_index.shape[-1]).std(axis=0, dtype=np.float64)

        return AM_index
    
    def cal_slice(self, month_start, len_slice, slice_offset=0):
        
        return self.get_slice(self.index, month_start, len_slice, slice_offset)

#=============================================================
# NAM index class
#=============================================================
class NAM(AM):
    def __init__(self, data, index_name='NAM', annual_cycle_fft=3, running_mean=0, save_index=False):

        super().__init__(data, index_name, annual_cycle_fft, running_mean, save_index)

#=============================================================
# SAM index class
#=============================================================
class SAM(AM):
    def __init__(self, data, index_name='SAM', annual_cycle_fft=3, running_mean=0, save_index=False):

        super().__init__(data, index_name, annual_cycle_fft, running_mean, save_index)

#=============================================================
# tests of MyGetData class
#=============================================================
def MyGetData_test():
    print("Testing MyGetData class ......")

    data = JRA55(name_dir='jra_55', year_start=1958, year_end=2016, name='JRA55')
    D = NAM(data, annual_cycle_fft=3, running_mean=0)
    print(f'{D.data.name}:', D.index.shape)

    data = CMIP6(name_dir='CESM2', year_start=1950, year_end=2014, source_dir='cmip6')
    D = NAM(data, annual_cycle_fft=3, running_mean=0)
    print(f'{D.data.name}:', D.index.shape)

    data = ERA5(name_dir='ERA5', year_start=1950, year_end=2021, source_dir='cmip6')
    D = NAM(data, annual_cycle_fft=3, running_mean=0)
    print(f'{D.data.name}:', D.index.shape)

#=============================================================
# get_y function
#=============================================================
def get_y(data, index_name, len_slice=150, offset=40, save_index=False):
    D = AM(data, index_name=index_name, annual_cycle_fft=3, running_mean=0, save_index=save_index)
    p = D.data.level
    print(f'{D.data.name}', D.index.shape, f'\tcalendar:', D.data.calendar)

    # get slices of data for len_slice starting nov 1
    y = D.cal_slice(month_start=11, len_slice=len_slice)
    y = y.astype('float32')
    t = np.arange(y.shape[1], dtype=y.dtype)

    # get slices of data for len_slice starting nov 1 with the padding of offset=40 in the beginning and end
    y2 = D.cal_slice(month_start=11, len_slice=len_slice, slice_offset=offset)
    y2 = y2.astype('float32')
    print(f't.shape = {t.shape}, y.shape = {y.shape}, y2.shape = {y2.shape}')

    return D, p, y, t, y2

def get_y_test():
    print("Testing get_y function ......")

    plev = [850.,  700.,  500.,  250.,  100.,   50.,   10.]
    len_slice = 150
    offset = 40
    data = ERA5(name_dir='ERA5', year_start=1950, year_end=2014, plev=plev, source_dir='cmip6')
    D, p, y, t, y2 = get_y(data, 'NAM', len_slice=len_slice, offset=offset)    
    print(f't.shape = {t.shape}, y.shape = {y.shape}, y2.shape = {y2.shape}')

#=============================================================
# main function
#=============================================================
if __name__ == "__main__":
    # MyDataSet_test()
    # MyGetData_test()
    get_y_test()