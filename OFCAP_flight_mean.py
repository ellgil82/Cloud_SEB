import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import iris
import os
import fnmatch
import matplotlib
import matplotlib.collections as mcoll
import matplotlib.cm as cmx
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dateutil
from itertools import groupby, count
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams
import sys
sys.path.append('/users/ellgil82/scripts/Tools/')
from tools import compose_date, compose_time
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap
import os
import fnmatch
import netCDF4
import math
from math import pi
import operator
from mpl_toolkits.basemap import Basemap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
from numpy import cos, sin, flipud
import time
import datetime
#import code # interact with session
import pdb   # interact with session
from scipy.stats import stats
from scipy.stats import nanstd, nanmean

flight_dict = {'flight150': '2011-01-15',
               'flight152': '2011-01-18',
               'flight153': '2011-01-19',
               'flight154': '2011-01-20',
               # 'flight155': '2011-01-21',
               'flight159': '2011-01-25',
               'flight168': '2011-01-31',
               'flight170': '2011-02-01',
               'flight172': '2011-02-02',
               'flight179': '2011-02-06'}

[150,152,155,159,170,179]'flight170',

flight_list = ['flight150', 'flight152', 'flight159',   'flight179']

def load_obs(flight, flight_date):
    ''' Inputs: flight number as a string, e.g. 'flight159' and flight_date as a string, with date only (not time) in YYYY-MM-DD format, e.g. '2011-01-25' '''
    ## ----------------------------------------------- SET UP VARIABLES --------------------------------------------------##
    ## Load core data
    try:
        print('\nYes yes cuzzy, pretty soon you\'re gonna have some nice core data...')
        path = []
        for file in os.listdir('/data/mac/ellgil82/cloud_data/core_data/'):
                if fnmatch.fnmatch(file, '*%(flight)s_*' % locals()):
                    path.append(file)
        os.chdir('/data/mac/ellgil82/cloud_data/core_data/')
        cubes = iris.load(path)
        #RH = iris.load_cube(bsl_path_core, 'relative_humidity')
        core_temp = cubes[34] #de-iced temperature
        core_temp.convert_units('celsius')
        plane_lat = iris.load_cube(path, 'latitude')
        plane_lon = iris.load_cube(path, 'longitude')
        plane_alt = iris.load_cube(path, 'altitude')
        core_time =  iris.load_cube(path, 'time')
        core_time.convert_units('hours since '+flight_date+' 00:00:00')
        # Find times where lons between -64 and -60
        time_range = np.where(plane_lon.data > -65.)
        # convert to 1 Hz
        icepath = '/data/mac/ellgil82/cloud_data/Constantino_Oasis_Peninsula/'
        start_time = time_range[0][0]
        end_time = time_range[-1][-1]
        file = flight+'_s_v2.npz'
        npzfile = np.load(icepath + file)
        timeflight = npzfile['time']
        timeflight = timeflight/(60*60) # convert to hours
        ice_range = np.where((timeflight > core_time[start_time].data) & (timeflight < core_time[end_time].data))
        ice_range = ice_range[0]
        niceflight = npzfile['TestPlot_HI_z'][ice_range[0]:ice_range[-1]] + npzfile['TestPlot_MI_z'][ice_range[0]:ice_range[-1]] # ice particles where lon > -64
        niceflight = niceflight * 2.
        niceflight = niceflight/1000 #in cm-3
        nice_tot = np.sum(niceflight, axis = 1)
        niceflight_masked = np.ma.masked_where(nice_tot < 0.00000001, nice_tot) # in cm-3
        nice_mn = np.ma.mean(niceflight_masked)
        npz_m = np.load(icepath + flight+'_m_v2.npz')
        IWC = npz_m['TestPlot_HI_y'][ice_range[0]:ice_range[-1]] + npz_m['TestPlot_MI_y'][ice_range[0]:ice_range[-1]]
        IWC_masked = np.ma.masked_where(nice_tot < 0.00000001, IWC)
        IWC_mn = np.ma.mean(IWC_masked)
        CAS_path = '/data/mac/ellgil82/cloud_data/netcdfs/' + flight + '_cas.nc'
        LWC_cas = iris.load_cube(CAS_path, 'liquid water content calculated from CAS ')
        aer = iris.load_cube(CAS_path, 'Aerosol concentration spectra measured by cas ')
        aer = np.sum(aer.data, axis =0)
        CAS_time = iris.load_cube(CAS_path, 'time')
        CAS_time.convert_units('milliseconds since 2011-01-25 00:00:00')
        CAS_time.convert_units('hours since 2011-01-25 00:00:00')
        liq_range = np.where((CAS_time.data > core_time[start_time].data) & (CAS_time.data < core_time[end_time].data))
        liq_range = liq_range[0]
        LWCflight = LWC_cas[liq_range[0]:liq_range[-1]]
        ndropflight = aer[liq_range[0]:liq_range[-1]]
        LWC_masked = np.ma.masked_where(ndropflight < 1.0, LWCflight.data)
        ndrop_masked = np.ma.masked_where(ndropflight < 1.0, ndropflight)
        LWC_mn = np.ma.mean(LWC_masked)
        ndrop_mn = np.ma.mean(ndrop_masked)
        return nice_mn, IWC_mn, ndrop_mn, LWC_mn
    except:
        print('Sorry, no can do for that flight')


def print_stats():
    for i in flight_list:
        nice_mn, IWC_mn, ndrop_mn, LWC_mn = load_obs(flight = i, flight_date = flight_dict[i])
        print('\n\nMean stats for %(i)s:' %locals())
        print('\n ice number: ')
        print nice_mn
        print('\n IWC: ')
        print IWC_mn
        print('\n droplet number: ')
        print ndrop_mn
        print('\n LWC: ')
        print LWC_mn

print_stats()

def calc_mean():
    mean_dict = []
    for i in flight_list:
        nice_mn, IWC_mn, ndrop_mn, LWC_mn = load_obs(flight=i, flight_date=flight_dict[i])
        mean_dict = np.append(mean_dict,[nice_mn, IWC_mn, ndrop_mn, LWC_mn])
    mean_dict = np.resize(mean_dict, new_shape = (len(flight_list),4))
    means = np.mean(mean_dict, axis = 0)
    print('\n\nMean stats for OFCAP cloud flights:')
    print('\n ice number: ')
    print(means[0])
    print('\n IWC: ')
    print(means[1])
    print('\n droplet number: ')
    print(means[2])
    print('\n LWC: ')
    print(means[3])


calc_mean()