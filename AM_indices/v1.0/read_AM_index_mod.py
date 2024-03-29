#=============================================================
# module to read the Annular mode indices from reanalysis data 
# and cmip models
#=============================================================
import numpy as np
import os

import netCDF4
from netCDF4 import Dataset
from netCDF4 import date2num, num2date

import calendar
import datetime as dt

import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#=============================================================
# Shared functions
#=============================================================
def cal_annual_cycle(var, annual_cycle_fft=4):
    """ 
    Input: var(year, days_in_a_year, :)
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

def cal_running_mean(var, running_mean=0):
    """ 
    Input: var(year*days_in_a_year, :)
    calculate the running mean of `var` along axis=0
    running_mean: smoothing with `running_mean`
    """

    var_o = np.empty_like(var)
    if running_mean > 0:
        for k in range(var.shape[1]):
            var_o[:, k] = np.convolve(var[:, k], np.ones((running_mean,))/running_mean, mode='same')
    else:
        var_o = var

    return var_o

#=============================================================
# class Reanalysis()
# Read the annular modes from reanalysis data
#=============================================================
class Reanalysis():
    def __init__(self, name_dir, year_start, year_end, plev=None, name=None, source_dir=None, \
                annual_cycle_fft=4, running_mean=0, save_index=False):
        """
        Calculate the annular mode indices from pre-processed geopotential height averaged over the polar caps. \\
        The input data is 6 hourly, and the output is daily average. \\ 
        The global mean height is removed. The AM indices are normalized. \\
        The indices are saved as 
            NAM_index(year*days_in_a_year, plev),
            SAM_index(year*days_in_a_year, plev)

        Key options
        annual_cycle_fft: option to remove the harmonics above `annual_cycle_fft` from the annual cycle
        running_mean: option to conduct running average for the anomaly with `running_mean`

        References
        Gerber, E. P., and P. Martineau, 2018: Quantifying the variability of the annular modes: 
        Reanalysis uncertainty vs. sampling uncertainty. Atmos. Chem. Phys., 18, 17099–17117, https://doi.org/10.5194/acp-18-17099-2018.
        """

        self.name_dir = name_dir
        self.year_start = year_start
        self.year_end = year_end
        self.num_years = year_end - year_start + 1

        if plev:
            self.level = plev
        else:
            self.level = None

        if name:
            self.name = name
        else:
            self.name = name_dir
                
        self.root_dir = os.getcwd()
        if source_dir:
            self.source_dir = self.root_dir + '/' + source_dir    # specify source_dir
        else:
            self.source_dir = self.root_dir    # use cwd as source_dir
                            
        self.annual_cycle_fft = annual_cycle_fft
        self.running_mean = running_mean
        self.save_index = save_index

        # http://cfconventions.org/cf-conventions/cf-conventions#calendar
        self.calendar = '365_day'
        self.length_of_year = 365

        self.file_save = None
        self.NAM = None
        self.SAM = None

    def init_data(self):
        """
        Initializing attributes and methods for Reanalysis
        """

        file = self.source_dir + '/' + self.name_dir + '/MODES_4xdaily_2007_01.nc'
        ncfile = Dataset(file, 'r')

        level_input = ncfile.variables['pressure'][:].astype(np.float32)
        if ncfile.variables['pressure'].units == 'Pa':   # converting from Pa to hPa
            level_input /= 100
        if self.level is None:
            self.level = level_input            # use level_input as self.level if self.level is not specified
        if (level_input[1]-level_input[0])*(self.level[1]-self.level[0]) < 0:
            self.level = np.flip(self.level, axis=0)    # adjust the direction of variations in self.level and level_input
        self.level_index = np.isin(level_input, self.level)   # indices in level_input for elements in self.level
        self.num_levels = len(self.level)
        # print(f'Pressure input: {level_input}\n Pressure output: {self.level}\n Pressure levels used: {level_input[self.level_index]}\n')

        if self.annual_cycle_fft != 4 or self.running_mean != 0:      # default values
            tag = f"_fft{self.annual_cycle_fft}_rm{self.running_mean}"
        else:
            tag = ""
        self.file_save = self.root_dir + '/data/' + self.name_dir + '/AM_daily_' \
            + str(self.year_start) + '_' + str(self.year_end) + tag + '.nc'

    def load_data(self):
        """
        read, calculate and save data in self.NAM, self.SAM
        """

        if self.file_save is None:
            self.init_data()

        if os.path.exists(self.file_save):
            print('Reading from saved data ......')
            self.NAM, self.SAM = self.read_AM_index()
        else:
            print('Calculating from the original data .......')
            self.NAM, self.SAM = self.cal_AM_index()
            if self.save_index:
                self.save_AM_index()        
        # print('Completed! .......')

    def slice_data(self, month_start, len_slice, month_start2=None):
        """
        slice NAM and SAM each year starting from `month_start` by the length of 'len_slice'
        return  NAM_slice(year, len_slice, plev),
                SAM_slice(year, len_slice, plev)
        """

        if self.NAM is None or self.SAM is None:
            self.load_data()

        if month_start2 is None:
            month_start2 = month_start
        
        units = 'days since 1800-01-01'
        calendar = self.calendar
        date_start = dt.datetime(self.year_start, 1, 1, 0)
        num_start = date2num(date_start, units=units, calendar=calendar)

        NAM_slice = np.empty((0, len_slice, self.num_levels))
        SAM_slice = np.empty((0, len_slice, self.num_levels))
        for y in range(self.year_start, self.year_end+1):
            date_end = dt.datetime(y, month_start, 1, 0)
            num_end = date2num(date_end, units=units, calendar=calendar)
            slice_start = num_end - num_start
            date_end2 = dt.datetime(y, month_start2, 1, 0)
            num_end2 = date2num(date_end2, units=units, calendar=calendar)
            slice_start2 = num_end2 - num_start
            # print(y, slice_start, len_slice, self.NAM.shape[0])
            if slice_start+len_slice <= self.NAM.shape[0]:
                NAM_slice = np.vstack((NAM_slice, self.NAM[slice_start:slice_start+len_slice, :][None,:,:]))
            if slice_start2+len_slice <= self.NAM.shape[0]:
                SAM_slice = np.vstack((SAM_slice, self.SAM[slice_start2:slice_start2+len_slice, :][None,:,:]))

        NAM_slice /= NAM_slice.reshape(-1, self.num_levels).std(axis=0)
        SAM_slice /= SAM_slice.reshape(-1, self.num_levels).std(axis=0)

        return NAM_slice, SAM_slice 

    def read_AM_index(self):
        """
        Reading the existing AM indices from 'file_save'
        """
        
        if self.file_save:
            ncfile = Dataset(self.file_save, mode='r')
            return ncfile.variables['NAM'][:], ncfile.variables['SAM'][:]
        else:
            raise Exception("'file_save' does not exist!")

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
            file = self.source_dir + '/' + self.name_dir + '/' \
                + f'MODES_4xdaily_{year}_{str(month).zfill(2)}.nc'
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
        
    def get_data(self, var_name):
        """
        read data from year_start to year_end
        return var(year, days_in_a_year, pressure)
        """

        if self.file_save is None:
            self.init_data()

        var = np.empty((0, self.length_of_year, self.num_levels), np.float32)    # dim(year, days_in_a_year, pressure)
        for year in range(self.year_start, self.year_end+1):
            var = np.vstack((var, self.get_data_1year(year, var_name)[None, :]))

        return var

    def cal_AM_anomaly(self, var_name):
        """
        calculate anomalies
        return var_o(year*days_in_a_year, pressure),    # total field
               var_mean_o(days_in_a_year, pressure),    # annual cycle, option to remove
                                                        # the harmonics above `annual_cycle_fft`
               var_anomaly_o(year*days_in_a_year, pressure) # anomaly, option to conduct
                                                            # running average with `running_mean`
        """

        # print(f"{self.name}: {var_name}")
        var = self.get_data(var_name)
        var_o = var.reshape(-1, self.num_levels)    # dim(year*days_in_a_year, pressure)

        var_name_o = cal_annual_cycle(var, self.annual_cycle_fft)

        var_anomaly = (var - var_name_o).reshape(-1, self.num_levels)    # broadcasting the 1st dimension
        var_anomaly_o = cal_running_mean(var_anomaly, self.running_mean)

        return var_o, var_name_o, var_anomaly_o

    def cal_AM_index(self):
        """
        Calculating the annular mode indices from the original data
        remove the annual cycle and apply running mean as needed
        remove the global mean and standardize the anomaly time series

        return NAM_index(year*days_in_a_year, pressure),
               SAM_index(year*days_in_a_year, pressure)
        """

        NAM, NAM_mean, NAM_anomaly = self.cal_AM_anomaly('NAM')
        SAM, SAM_mean, SAM_anomaly = self.cal_AM_anomaly('SAM')
        GLOBAL, GLOBAL_mean, GLOBAL_anomaly = self.cal_AM_anomaly('GLOBAL')

        NAM_index = -(NAM_anomaly-GLOBAL_anomaly)
        SAM_index = -(SAM_anomaly-GLOBAL_anomaly)

        NAM_index /= NAM_index.std(axis=0)
        SAM_index /= SAM_index.std(axis=0)

        return NAM_index, SAM_index

    def save_AM_index(self):
        """
        save the annular mode indices in 'file_save'

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
        plev_dim = ncfile.createDimension('plev', self.num_levels)
        time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).

        # Creating attributes
        ncfile.title='Annular Mode indices'

        # Creating coordinate and data variables
        plev = ncfile.createVariable('plev', np.float32, ('plev',))
        plev.units = 'hPa'
        plev.long_name = 'pressure level'

        time = ncfile.createVariable('time', np.float64, ('time',))
        time.units = 'hours since 1800-01-01'
        time.long_name = 'time'
        time.calendar = self.calendar

        NAM = ncfile.createVariable('NAM',np.float32,('time','plev')) # note: unlimited dimension is leftmost
        NAM.units = ''
        NAM.standard_name = 'Northern Annular Mode Index'

        SAM = ncfile.createVariable('SAM',np.float32,('time','plev')) # note: unlimited dimension is leftmost
        SAM.units = ''
        SAM.standard_name = 'Southern Annular Mode Index'

        # Writing data
        # Note: the ":" is necessary in these "write" statements
        plev[:] = self.level

        date_start = dt.datetime(self.year_start, 1, 1, 0)
        if self.calendar == "365_day":
            date_end = dt.datetime(self.year_end, 12, 31, 0)
        else:
            date_end = dt.datetime(self.year_end, 12, 30, 0)
        num_start = date2num(date_start, units=time.units, calendar=time.calendar)
        num_end = date2num(date_end, units=time.units, calendar=time.calendar)
        time[:] = np.linspace(num_start, num_end, self.length_of_year*(self.year_end-self.year_start+1))
        #print(num2date(time[0:4], units=time.units, calendar=time.calendar))
        #print(num2date(time[-4:], units=time.units, calendar=time.calendar))

        NAM[:,:] = self.NAM
        SAM[:,:] = self.SAM
        
        # Closing a netCDF file
        print(ncfile)
        ncfile.close()

#=============================================================
# class CMIP6(Reanalysis)
# Read the annular modes from cimp6 data
#=============================================================
class CMIP6(Reanalysis):
    def __init__(self, name_dir, year_start, year_end, plev=None, name=None, source_dir=None, \
                 annual_cycle_fft=4, running_mean=0, save_index=False):
        """
        Calculate the annular mode indices from pre-processed geopotential height averaged over the polar caps. \\
        Inherited from class Reanalysis.     
        The input and output data are daily. \\ 
        """

        super().__init__(name_dir, year_start, year_end, plev, name, source_dir, \
                 annual_cycle_fft, running_mean, save_index)

    def init_data(self):
        """
        overwrite `init_data` in class Reanalysis
        Initializing attributes and methods for CMIP6
        """

        file = self.source_dir + '/' + self.name_dir + '/AM_daily_1950_2014.nc'
        ncfile = Dataset(file, 'r')

        level_input = ncfile.variables['plev'][:].astype(np.float32)
        if ncfile.variables['plev'].units == 'Pa':   # converting from Pa to hPa
            level_input /= 100
        if self.level is None:
            self.level = level_input            # use level_input as self.level if self.level is not specified
        self.level_index = np.isin(level_input, self.level)   # indices in level_input for elements in self.level
        self.num_levels = len(self.level)
        # print(f'Pressure input: {level_input}\n Pressure output: {self.level}\n Pressure levels used: {level_input[self.level_index]}\n')

        if self.annual_cycle_fft != 4 or self.running_mean != 0:      # default values
            tag = f"_fft{self.annual_cycle_fft}_rm{self.running_mean}"
        else:
            tag = ""
        self.file_save = self.root_dir + '/data/' + self.name_dir + '/AM_daily_' \
            + str(self.year_start) + '_' + str(self.year_end) + tag + '.nc'

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
        
    def get_data(self, var_name):
        """
        overwrite `get_data` in class Reanalysis
        read data from year_start to year_end
        return var(year, days_in_a_year, pressure)
        """

        if self.file_save is None:
            self.init_data()

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
# Examples of classes
# class Reanalysis()
# class CMIP6(Reanalysis)
#=============================================================
#
# D = Reanalysis(name_dir='jra_55', year_start=1958, year_end=2014, plev=None, name='JRA55', \
#         annual_cycle_fft=4, running_mean=0, save_index=True)
# D.load_data()

# D = CMIP6(name_dir='GFDL-ESM4', year_start=1958, year_end=2014, plev=None, name='GFDL-ESM4', source_dir='cmip6', \
#         annual_cycle_fft=4, running_mean=0, save_index=True)
# D.load_data()
#


#=============================================================
# get red noise data
#=============================================================
def get_red_noise(gamma=[0.06], data_size=1000, seed=None):
    if seed is not None:
        np.random.seed(19680801+seed)
    else:
        np.random.seed(19680801)
    #eps = np.random.uniform(-1, 1, data_size)
    eps = np.random.randn(data_size)
        
    y = np.zeros_like(eps)
    if len(gamma) == 1:    # AR-1
        for i in range(1, data_size):
            y[i] = (1-gamma[0])*y[i-1] + eps[i-1]
    elif len(gamma) == 2:    # AR-2
        for i in range(2, data_size):
            y[i] = (1-gamma[0])*y[i-1] + gamma[1]*y[i-2] + eps[i-1]

    t = torch.linspace(0, len(y)-1, len(y), dtype=torch.float32)
    true_y = torch.from_numpy(y.astype(np.float32))[:,None]
        
    return  t, true_y


def test_red_noise():
    gamma=[0.06]
    len_data = 10000
    tt, yy = get_red_noise(gamma=gamma, data_size=len_data)

    lag_time = 60
    lags = np.linspace(0, lag_time, lag_time+1, dtype=int)
    Ct = np.array([np.squeeze(yy.numpy()[lag:].T @ yy.numpy()[:len_data-lag]) for lag in lags])

    from scipy.optimize import curve_fit
    func = lambda x, b: np.exp(-b * x)
    popt, pcov = curve_fit(func, lags, Ct/Ct[0])

    plt.plot(lags, Ct/Ct[0])
    plt.plot(lags, np.exp(-lags*popt[0]))
    plt.plot(lags, np.exp(-lags*gamma[0]))
    print(f'gamma={gamma[:]}, popt={popt[:]}')