#=============================================================
# Module to read the AM indices from reanalyses and cmip models
#=============================================================
import numpy as np
import os

import netCDF4
from netCDF4 import Dataset
from netCDF4 import date2num, num2date

import calendar
import datetime as dt


#=============================================================
# My DataSet class
#=============================================================
class MyDataSet():

    def __init__(self, name_dir, year_start, year_end, plev=None, name=None, source_dir=None):
        """
        read the data set from `source_dir` + `/` + `name_dir` for the period from `year_start` to `year_end`        \\
        plev: specify the axis for x
        name: specify the dataset name
        method `get_data`: return var(year, days_in_a_year, x), where year and days_in_a_year are time, and x is space
        """

        self.name_dir = name_dir
        self.year_start = year_start
        self.year_end = year_end
        self.num_years = year_end - year_start + 1

        if plev:                            # specify plev as needed
            self.level = plev
        else:
            self.level = None

        if name:                            # specify dataset name
            self.name = name
        else:
            self.name = name_dir
                
        self.root_dir = os.getcwd()
        if source_dir:
            self.source_dir = self.root_dir + '/' + source_dir    # specify source_dir
        else:
            self.source_dir = self.root_dir    # use cwd as source_dir

    def get_data(self, var_name):
        """
        Need to specify the method with inheritance
        """
        raise Exception('Please specify the `get_data` method via inheritance!')

#=============================================================
# JRA55 DataSet class
#=============================================================
class JRA55(MyDataSet):
    
    def __init__(self, name_dir, year_start, year_end, plev=None, name=None, source_dir=None):
        """
        Initializing attributes and methods for JRA55 @ python/AM_indices
        The input data is 4xdaily, and the output is daily average.
        """

        super().__init__(name_dir, year_start, year_end, plev, name, source_dir)

        file = self.source_dir + '/' + self.name_dir + '/MODES_4xdaily_2007_01.nc'
        ncfile = Dataset(file, 'r')

        level_input = ncfile.variables['pressure'][:].astype(np.float32)
        if ncfile.variables['pressure'].units == 'Pa':   # converting from Pa to hPa
            level_input /= 100
        if self.level is None:
            self.level = level_input            # use level_input as self.level if self.level is not specified
        if (level_input[1]-level_input[0])*(self.level[1]-self.level[0]) < 0:
            self.level = np.flip(self.level, axis=0)    # adjust the direction of variations in self.level according to level_input
        self.level_index = np.isin(level_input, self.level)   # indices in level_input for elements in self.level
        self.num_levels = len(self.level)
        # print(f'Pressure input: {level_input}\n Pressure output: {self.level}\n Pressure levels used: {level_input[self.level_index]}\n')

        # http://cfconventions.org/cf-conventions/cf-conventions#calendar
        self.calendar = '365_day'
        self.length_of_year = 365

    def get_data(self, var_name):
        """
        read data from year_start to year_end
        return var(year, days_in_a_year, pressure)
        """

        var = np.empty((0, self.length_of_year, self.num_levels), np.float32)    # dim(year, days_in_a_year, pressure)
        for year in range(self.year_start, self.year_end+1):
            var = np.vstack((var, self.get_data_1year(year, var_name)[None, :]))

        return var  # dim(year, days_in_a_year, pressure)

    def get_data_1year(self, year, var_name):
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

        super().__init__(name_dir, year_start, year_end, plev, name, source_dir)

        file = self.source_dir + '/' + self.name_dir + '/AM_daily_1950_2014.nc'
        ncfile = Dataset(file, 'r')

        level_input = ncfile.variables['plev'][:].astype(np.float32)
        if ncfile.variables['plev'].units == 'Pa':   # converting from Pa to hPa
            level_input /= 100
        if self.level is None:
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
        
    def get_data(self, var_name):
        """
        read data from year_start to year_end
        return var(year, days_in_a_year, pressure)
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

        super().__init__(name_dir, year_start, year_end, plev, name, source_dir)

        file = self.source_dir + '/' + self.name_dir + '/AM_daily_1950_2021.nc'
        ncfile = Dataset(file, 'r')

        level_input = ncfile.variables['level'][:].astype(np.float32)
        if ncfile.variables['level'].units == 'Pa':   # converting from Pa to hPa
            level_input /= 100
        if self.level is None:
            self.level = level_input            # use level_input as self.level if self.level is not specified
        if (level_input[1]-level_input[0])*(self.level[1]-self.level[0]) < 0:
            self.level = np.flip(self.level, axis=0)    # adjust the direction of variations in self.level according to level_input
        self.level_index = np.isin(level_input, self.level)   # indices in level_input for elements in self.level
        self.num_levels = len(self.level)
        # print(f'Pressure input: {level_input}\n Pressure output: {self.level}\n Pressure levels used: {level_input[self.level_index]}\n')
        
        # http://cfconventions.org/cf-conventions/cf-conventions#calendar
        self.calendar = '365_day'
        self.length_of_year = 365

    def get_data(self, var_name):
        """
        read data from year_start to year_end
        return var(year, days_in_a_year, pressure)
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
# tests of My DataSet class
#=============================================================
def MyDataSet_test():
    jra55 = JRA55(name_dir='jra_55', year_start=1958, year_end=2016, name='JRA55')
    nam = jra55.get_data('NAM')
    print(nam.shape)

    CESM2 = CMIP6(name_dir='CESM2', year_start=1950, year_end=2014, source_dir='cmip6')
    nam = CESM2.get_data('NAM')
    print(nam.shape)

    era5 = ERA5(name_dir='ERA5', year_start=1950, year_end=2021, source_dir='cmip6')
    nam = era5.get_data('NAM')
    print(nam.shape)


#=============================================================
# My Index class
#=============================================================
class MyIndex():
    def __init__(self, data, index_name, annual_cycle_fft=2, running_mean=0, save_index=False):
        """
        Calulate indices from `data (type: MyDataSet)` and write data with the name of `index_name`
        Use the method `get_data` from the class `MyDataSet`
        Key options:
        annual_cycle_fft: option to remove the harmonics above `annual_cycle_fft` from the annual cycle
        running_mean: option to conduct running average for the anomaly with `running_mean`
        """

        self.data = data
        self.index_name = index_name
        self.annual_cycle_fft = annual_cycle_fft
        self.running_mean = running_mean
        self.save_index = save_index

        if self.annual_cycle_fft != 2 or self.running_mean != 0:      # default values
            tag = f"_fft{self.annual_cycle_fft}_rm{self.running_mean}"
        else:
            tag = ""
        self.file_save = self.data.root_dir + '/data/' + self.data.name_dir + '/AM_daily_' \
                         + str(self.data.year_start) + '_' + str(self.data.year_end) + tag + '.nc'

        # read, calculate, and save data in self.index
        if os.path.exists(self.file_save):
            print('Reading from saved data ......')
            self.index = self.read_index()
        else:
            print('Calculating from the original data .......')
            self.index = self.cal_index()
            if self.save_index:
                self.write_index()        
        # print('Completed! .......')

    def read_index(self):
        """
        Reading the existing index from 'file_save'
        """
        
        if os.path.exists(self.file_save):
            ncfile = Dataset(self.file_save, mode='r')
            return ncfile.variables[self.index_name][:]
        else:
            raise Exception("'file_save' does not exist!")

    def cal_index(self):
        """
        return index(year*days_in_a_year, x)
        """

        NAM, NAM_mean, NAM_anomaly = self.cal_anomaly('NAM')
        return NAM_anomaly / NAM_anomaly.std(axis=0)

    def write_index(self):
        """
        save index in 'file_save'

        see more about the netCDF4 package at
        https://unidata.github.io/python-training/workshop/Bonus/netcdf-writing/
        """

        if self.file_save is None:
            raise Exception("'file_save' is not initialized! ......")

        if not os.path.exists(os.path.dirname(self.file_save)):
            os.makedirs(os.path.dirname(self.file_save))

        # Opening a file
        try: 
            ncfile.close()  # just to be safe, make sure dataset is not already open.
        except: 
            pass

        ncfile = Dataset(self.file_save, mode='w', format='NETCDF4_CLASSIC') 

        # Creating dimensions
        x_dim = ncfile.createDimension('x', self.index.shape[1])
        time_dim = ncfile.createDimension('time', None)     # unlimited axis (can be appended to).

        # Creating attributes
        ncfile.title = self.index_name

        # Creating coordinate and data variables
        x = ncfile.createVariable('x', np.float32, ('x',))
        x.units = ''
        x.long_name = 'x axis'

        time = ncfile.createVariable('time', np.float64, ('time',))
        time.units = 'hours since 1800-01-01'
        time.long_name = 'time'
        time.calendar = self.data.calendar

        index = ncfile.createVariable(self.index_name, np.float32, ('time','x'))    # note: unlimited dimension is leftmost
        index.units = ''
        index.standard_name = self.index_name

        # Writing data
        # Note: the ":" is necessary in these "write" statements
        x[:] = np.linspace(1.0, self.index.shape[1], self.index.shape[1])   # from 1 to the size of `self.index``

        date_start = dt.datetime(self.data.year_start, 1, 1, 0)
        if self.data.calendar == "365_day":
            date_end = dt.datetime(self.data.year_end, 12, 31, 0)
        else:
            date_end = dt.datetime(self.data.year_end, 12, 30, 0)
        num_start = date2num(date_start, units=time.units, calendar=time.calendar)
        num_end = date2num(date_end, units=time.units, calendar=time.calendar)
        time[:] = np.linspace(num_start, num_end, self.data.length_of_year*(self.data.year_end-self.data.year_start+1))
        #print(num2date(time[0:4], units=time.units, calendar=time.calendar))
        #print(num2date(time[-4:], units=time.units, calendar=time.calendar))

        index[:,:] = self.index
        
        # Closing a netCDF file
        print(ncfile)
        ncfile.close()

    def cal_anomaly(self, var_name):
        """
        calculate anomalies and remove the annual cycle and apply running mean as needed
        return var_o(year*days_in_a_year, x),    # total field
                var_mean_o(days_in_a_year, x),    # annual cycle, option to remove the harmonics above `annual_cycle_fft`
                var_anomaly_o(year*days_in_a_year, x)    # anomaly, option to conduct running average with `running_mean`
        """

        # print(f"{self.name}: {var_name}")
        var = self.data.get_data(var_name)    # use the `get_data` method of the data
        var_o = var.reshape(-1, self.data.num_levels)    # dim(year*days_in_a_year, x)

        var_name_o = MyIndex.cal_annual_cycle(var, self.annual_cycle_fft)

        var_anomaly = (var - var_name_o).reshape(-1, self.data.num_levels)    # broadcasting the 1st dimension
        var_anomaly_o = MyIndex.cal_running_mean(var_anomaly, self.running_mean)

        return var_o, var_name_o, var_anomaly_o

    def cal_slice(self, month_start, len_slice):
        """
        slice `self.index` each year starting from `month_start` by the length of 'len_slice'
        return  my_slice(year, len_slice, x)
        """
        
        units = 'days since 1800-01-01'
        calendar = self.data.calendar
        date_start = dt.datetime(self.data.year_start, 1, 1, 0)
        num_start = date2num(date_start, units=units, calendar=calendar)

        my_slice = np.empty((0, len_slice, self.index.shape[1]))
        for y in range(self.data.year_start, self.data.year_end+1):
            date_end = dt.datetime(y, month_start, 1, 0)
            num_end = date2num(date_end, units=units, calendar=calendar)
            slice_start = num_end - num_start
            # print(y, slice_start, len_slice, self.index.shape[0])
            if slice_start+len_slice <= self.index.shape[0]:
                my_slice = np.vstack((my_slice, self.index[slice_start:slice_start+len_slice, :][None,:,:]))

        my_slice /= my_slice.reshape(-1, self.index.shape[1]).std(axis=0)

        return my_slice 

    @staticmethod
    def cal_annual_cycle(var, annual_cycle_fft=4):
        """ 
        Input: var(year, days_in_a_year, x), where year and days_in_a_year are time, and x is space
        calculate the annual cycle of `var` by averaging in year
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
        Input: var(year*days_in_a_year, x), where year*days_in_a_year is time, and x is space
        calculate the running mean of `var` along axis=0
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
# NAM index class
#=============================================================
class NAM(MyIndex):
    def __init__(self, data, index_name, annual_cycle_fft=2, running_mean=0, save_index=False):
        super().__init__(data, index_name, annual_cycle_fft, running_mean, save_index)

    def cal_index(self):
        """
        Calculate the NAM indices from pre-processed geopotential height averaged over the polar caps. \\
        remove the global mean and standardize the anomaly time series
        return NAM_index(year*days_in_a_year, pressure)

        References
        Gerber, E. P., and P. Martineau, 2018: Quantifying the variability of the annular modes: 
        Reanalysis uncertainty vs. sampling uncertainty. Atmos. Chem. Phys., 18, 17099–17117, https://doi.org/10.5194/acp-18-17099-2018.
        """

        NAM, NAM_mean, NAM_anomaly = self.cal_anomaly('NAM')
        GLOBAL, GLOBAL_mean, GLOBAL_anomaly = self.cal_anomaly('GLOBAL')

        NAM_index = -(NAM_anomaly-GLOBAL_anomaly)
        NAM_index /= NAM_index.std(axis=0)

        return NAM_index

#=============================================================
# SAM index class
#=============================================================
class SAM(MyIndex):
    def __init__(self, data, index_name, annual_cycle_fft=2, running_mean=0, save_index=False):
        super().__init__(data, index_name, annual_cycle_fft, running_mean, save_index)

    def cal_index(self):
        """
        Calculate the SAM indices from pre-processed geopotential height averaged over the polar caps. \\
        remove the global mean and standardize the anomaly time series
        return SAM_index(year*days_in_a_year, pressure)

        References
        Gerber, E. P., and P. Martineau, 2018: Quantifying the variability of the annular modes: 
        Reanalysis uncertainty vs. sampling uncertainty. Atmos. Chem. Phys., 18, 17099–17117, https://doi.org/10.5194/acp-18-17099-2018.
        """

        SAM, SAM_mean, SAM_anomaly = self.cal_anomaly('SAM')
        GLOBAL, GLOBAL_mean, GLOBAL_anomaly = self.cal_anomaly('GLOBAL')

        SAM_index = -(SAM_anomaly-GLOBAL_anomaly)
        SAM_index /= SAM_index.std(axis=0)

        return SAM_index

#=============================================================
# tests of My Index class
#=============================================================
def MyIndex_test():
    data = JRA55(name_dir='jra_55', year_start=1958, year_end=2016, name='JRA55')
    index = NAM(data, index_name='NAM', annual_cycle_fft=2, running_mean=0, save_index=False)
    print(index.index.shape)

