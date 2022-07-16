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

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#=============================================================
# class Reanalysis()
# Read the annular modes from reanalysis data
#=============================================================
class Reanalysis():
    def __init__(self, name_dir, year_start, year_end, name, output_freq='4xdaily', source_dir=None, \
                 annual_cycle_fft=4, running_mean=0, save_AM_index=False):
        """
        Calculate the annular mode indices from pre-processed geopotential height averaged over the polar caps. \\
        The input data is 6 hourly. The global mean height is removed. The AM indices are normalized. \\
        The indices are saved as 
            NAM_index(year*days_in_a_year, pressure),
            SAM_index(year*days_in_a_year, pressure)

        Key options
        output_freq: take daily average if set as 'daily'
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
        self.name = name
    
        if output_freq == '4xdaily':
            self.freq = 4
        elif output_freq == 'daily':
            self.freq = 1
        else:
            raise Exception('output_freq is not recognized!')
            
        self.root_dir = os.getcwd()
        if source_dir is None:
            self.source_dir = self.root_dir    # use cwd as source_dir
        else:
            self.source_dir = source_dir    # specify source_dir
        
        file = self.source_dir + '/' + self.name_dir + '/' + 'MODES_4xdaily_2007_01.nc'
        ncfile = Dataset(file, 'r')
        self.level = ncfile.variables['pressure'][:]
        self.num_levels = len(self.level)
        
        self.annual_cycle_fft = annual_cycle_fft
        self.running_mean = running_mean
        
        if annual_cycle_fft != 4 or running_mean != 0:      # default values
            tag = f"_fft{annual_cycle_fft}_rm{running_mean}"
        else:
            tag = ""
        self.file_save = self.root_dir + '/' + self.name_dir + '/AM_daily_' + output_freq + '_' \
            + str(self.year_start) + '_' + str(self.year_end) + tag + '.nc'
        
        if os.path.exists(self.file_save):
            print('Reading from saved data ......')
            self.NAM, self.SAM = self.read_AM_index()
        else:
            print('Calculating from the original data .......')
            self.NAM, self.SAM = self.cal_AM_index()
            if save_AM_index:
                self.save_AM_index()

    def read_AM_index(self):
        """
        Reading the AM indices from 'file_save'
        """
        
        ncfile = Dataset(self.file_save, mode='r') 

        return ncfile.variables['NAM'][:], ncfile.variables['SAM'][:]
    
    def get_data_year(self, year, var_name):
        """
        read data for var_name and concatenate into the data in a year
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
            var_4xdaily = ncfile.variables[var_name][:]

            if self.freq == 4:
                var_tmp = var_4xdaily
            elif self.freq == 1:
                # daily average
                var_tmp = 0.25*(var_4xdaily[0::4, :] +var_4xdaily[1::4, :]
                               +var_4xdaily[2::4, :] +var_4xdaily[3::4, :])
                
            if calendar.isleap(year) and month==2:
                # remove the last day of Feburary in a leap year
                var_year = np.vstack((var_year, var_tmp[:-self.freq, :]))
            else:
                var_year = np.vstack((var_year, var_tmp[:]))

        return var_year  # dim(days_in_a_year, pressure)
    
    def get_data(self, var_name):
        """
        read data from year_start to year_end
        return var_o(year*days_in_a_year, pressure),    # total field
               var_mean_o(days_in_a_year, pressure),    # annual cycle, option to remove
                                                        # the harmonics above `annual_cycle_fft`
               var_anomaly_o(year*days_in_a_year, pressure) # anomaly, option to conduct
                                                            # running average with `running_mean`
        """

        var = np.empty((0, 365*self.freq, self.num_levels), np.float32)    # dim(year, days_in_a_year, pressure)
        for year in range(self.year_start, self.year_end+1):
            var = np.vstack((var, self.get_data_year(year, var_name)[None, :]))
        var_o = var.reshape(-1, self.num_levels)    # dim(year*days_in_a_year, pressure)

        var_mean = var.mean(axis=0)
        var_mean_o = np.empty_like(var_mean)
        if self.annual_cycle_fft > 0:
            var_mean_fft = np.fft.fft(var_mean, axis=0)
            var_mean_fft[self.annual_cycle_fft+1:-self.annual_cycle_fft, :] = 0
            var_mean_o = (np.fft.ifft(var_mean_fft, axis=0)).real
        else:
            var_mean_o = var_mean

        var_anomaly = (var - var_mean_o).reshape(-1, self.num_levels)    # broadcasting the 1st dimension        
        var_anomaly_o = np.empty_like(var_anomaly)
        if self.running_mean > 0:
            for k in range(self.num_levels):
                var_anomaly_o[:, k] = np.convolve(var_anomaly[:, k], 
                                               np.ones((self.running_mean,))/self.running_mean, mode='same')
        else:
            var_anomaly_o = var_anomaly

        return var_o, var_mean_o, var_anomaly_o

    def cal_AM_index(self):
        """
        Calculating the annular mode indices from the original data
        removing the global mean and standardize the anomaly time series

        return NAM_index(year*days_in_a_year, pressure),
               SAM_index(year*days_in_a_year, pressure)
        """

        NAM, NAM_mean, NAM_anomaly = self.get_data('NAM')
        SAM, SAM_mean, SAM_anomaly = self.get_data('SAM')
        GLOBAL, GLOBAL_mean, GLOBAL_anomaly = self.get_data('GLOBAL')

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
            
        # Opening a file
        # try: ncfile.close()  # just to be safe, make sure dataset is not already open.
        # except: pass
    
        if not os.path.exists(os.path.dirname(self.file_save)):
            os.makedirs(os.path.dirname(self.file_save))
            
        ncfile = Dataset(self.file_save, mode='w', format='NETCDF4_CLASSIC') 

        # Creating dimensions
        pressure_dim = ncfile.createDimension('pressure', self.num_levels)
        time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).

        # Creating attributes
        ncfile.title='Annular Mode indices'

        # Creating coordinate and data variables
        pressure = ncfile.createVariable('pressure', np.float32, ('pressure',))
        pressure.units = 'hPa'
        pressure.long_name = 'Pressure level'

        time = ncfile.createVariable('time', np.float64, ('time',))
        time.units = 'hours since 1800-01-01'
        time.long_name = 'time'
        time.calendar = 'noleap'

        NAM = ncfile.createVariable('NAM',np.float32,('time','pressure')) # note: unlimited dimension is leftmost
        NAM.units = ''
        NAM.standard_name = 'Northern Annular Mode Index'

        SAM = ncfile.createVariable('SAM',np.float32,('time','pressure')) # note: unlimited dimension is leftmost
        SAM.units = ''
        SAM.standard_name = 'Southern Annular Mode Index'

        # Writing data
        # Note: the ":" is necessary in these "write" statements
        pressure[:] = self.level

        if self.freq == 4:
            date_start = dt.datetime(self.year_start, 1, 1, 0)
            date_end = dt.datetime(self.year_end, 12, 31, 18)
        elif self.freq == 1:
            date_start = dt.datetime(self.year_start, 1, 1, 0)
            date_end = dt.datetime(self.year_end, 12, 31, 0)
                    
        num_start = date2num(date_start, units=time.units, calendar=time.calendar)
        num_end = date2num(date_end, units=time.units, calendar=time.calendar)
        time[:] = np.linspace(num_start, num_end, 365*self.freq*(self.year_end-self.year_start+1))
        #print(num2date(time[0:4], units=time.units, calendar=time.calendar))
        #print(num2date(time[-4:], units=time.units, calendar=time.calendar))

        NAM[:,:] = self.NAM
        SAM[:,:] = self.SAM
        
        # Closing a netCDF file
        print(ncfile)
        ncfile.close()


#=============================================================
# get the default annular mode indices at `p_levels`
#=============================================================
def get_AM_index(p_levels=None):
    """
    get the annular mode indcies at `p_levels`
    """

    D = Reanalysis(name_dir='jra_55', year_start=1958, year_end=2016, name='JRA55', \
                        output_freq='daily', annual_cycle_fft=4, running_mean=0, save_AM_index=True)
    
    if p_levels is None:
        p_levels = D.level

    print(f'Use pressure levels = {p_levels}\n')
    kk = np.isin(D.level, p_levels).nonzero()[0]   # indices of select pressure levels

    t = np.linspace(0, len(D.NAM)-1, len(D.NAM)).astype('float32')
    y = D.NAM[:,kk].data.astype('float32')

    return t, y, p_levels