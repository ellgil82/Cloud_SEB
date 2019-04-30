## -------------------------------- LOAD AND PLOT MONTH-LONG TIME SERIES OF MODEL DATA ----------------------------------- ##
host = 'bsl'

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
if host == 'jasmin':
    sys.path.append('/group_workspaces/jasmin4/bas_climate/users/ellgil82/scripts/Tools/')
elif host == 'bsl':
    sys.path.append('/users/ellgil82/scripts/Tools/')
from find_gridbox import find_gridbox
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap
import time

if host == 'jasmin':
    os.chdir('/group_workspaces/jasmin4/bas_climate/users/ellgil82/OFCAP/proc_data/')
elif host == 'bsl':
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/proc_data/')





OFCAP_QCF = iris.load_cube('timmean_QCF.nc', 'mass_fraction_of_cloud_ice_in_air')
OFCAP_QCF.convert_units('g kg-1')
OFCAP_QCF = OFCAP_QCF[0,:,:,:]
OFCAP_QCL = iris.load_cube('timmean_QCL.nc', 'mass_fraction_of_cloud_liquid_water_in_air')
OFCAP_QCL.convert_units('g kg-1')
OFCAP_QCL = OFCAP_QCL[0,:,:,:]
orog = iris.load_cube('OFCAP_orog.nc', 'surface_altitude')
lsm = iris.load_cube('OFCAP_lsm.nc', 'land_binary_mask')
orog = orog[0,0,:,:]
lsm = lsm[0,0,:,:]
try:
    altitude = profile_var.coord('altitude').points
except iris.exceptions.CoordinateNotFoundError:
    # Take orography data and use it to create hybrid height factory instance
    auxcoord = iris.coords.AuxCoord(orog.data, standard_name=str(orog.standard_name), long_name="orography",
                                    var_name="orog", units=orog.units)
    profile_var.add_aux_coord(auxcoord, (np.ndim(profile_var) - 2, np.ndim(profile_var) - 1))
    profile_var.coord('Hybrid height').convert_units('metres')
    factory = iris.aux_factory.HybridHeightFactory(delta=profile_var.coord("Hybrid height"),
                                                   orography=profile_var.coord("surface_altitude"))
    profile_var.add_aux_factory(
        factory)  # this should produce a 'derived coordinate', 'altitude' (test this with >>> print theta)
    altitude = profile_var.coord('altitude').points[:, 0, 0]


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
        return nice_mn, IWC_mn, ndrop_mn, LWC_mn, IWC_masked, LWC_masked, core_temp
    except:
        print('Sorry, no can do for that flight')


def calc_minmax(flight_list):
    minmax_dict = []
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
    for each_file in flight_list:
        nice_mn, -*+
        IWC_mn, ndrop_mn, LWC_mn, IWC_masked, LWC_masked, core_temp = load_obs(flight=each_file, flight_date=flight_dict[each_file])
        # Find minimum non-zero value in observations
        IWC_min = np.min(IWC_masked)
        LWC_min = np.min(LWC_masked)
        IWC_max = np.max(IWC_masked)
        LWC_max = np.max(LWC_masked)
        minmax_dict = np.append(minmax_dict, [LWC_min, LWC_max, IWC_min, IWC_max])
    minmax_dict = np.resize(minmax_dict, new_shape=(len(flight_list), 4))
    df = pd.DataFrame(minmax_dict, columns=['min_LWC', 'max_LWC', 'min_IWC', 'max_IWC'], index=flight_list)
    df.to_csv('/data/mac/ellgil82/cloud_data/flights/OFCAP_minmax.csv')
    print df

#calc_minmax(flight_list)

profile_var = OFCAP_QCF
# Create Larsen mask
orog = iris.load_cube('OFCAP_orog.nc', 'surface_altitude')
lsm = iris.load_cube('OFCAP_lsm.nc', 'land_binary_mask')
orog = orog[0, 0, :, :]
lsm = lsm[0, 0, :, :]
Larsen_mask = np.zeros((400, 400))
lsm_subset = lsm.data[35:260, 90:230]
Larsen_mask[35:260, 90:230] = lsm_subset
Larsen_mask[orog.data > 25] = 0
Larsen_mask = np.logical_not(Larsen_mask)
profile_mn = np.ma.masked_array(data=profile_var.data, mask=np.broadcast_to(Larsen_mask, profile_var.shape)).mean(axis=(1,2))




# Min of flights considered = 1.5 * 10-6 (LWC) and 1.0 * 10-4 (IWC)
in_cloud_QCF_Larsen = np.ma.masked_where(OFCAP_QCF.data >= 0.0001, OFCAP_QCF)
in_cloud_QCL_Larsen = np.ma.masked_where(OFCAP_QCL.data >= 0.0000015, OFCAP_QCL)


in_cloud_QCF_AWS14 = np.mean(OFCAP_QCF[:,199:201, 199:201].data, axis = (1,2))
in_cloud_QCF_AWS14[in_cloud_QCF_AWS14 <= 0.0001] = 0
in_cloud_QCL_AWS14 = np.mean(OFCAP_QCL[:,199:201, 199:201].data, axis = (1,2))
in_cloud_QCL_AWS14[in_cloud_QCL_AWS14 <= 0.0000015] = 0

in_cloud_QCF_AWS14 = np.ma.masked_where(.data >= 0.0001, OFCAP_QCF[:,:,199:201, 199:201].data)
mn_in_cloud_QCF_AWS14 = in_cloud_QCF_AWS14.mean(axis = (0, 2,3))

def in_cloud(input_var):
    if input_var == 'QCF':
        in_cl_profile = np.ma.masked_where(input_var >= 0.0001, input_var)
    elif input_var == 'QCL':
        in_cl_profile = np.ma.masked_where(input_var >= 0.0000015, input_var)
    return in_cl_profile