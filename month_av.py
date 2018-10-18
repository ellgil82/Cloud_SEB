## -------------------------------- LOAD AND PLOT MONTH-LONG TIME SERIES OF MODEL DATA ----------------------------------- ##

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
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap
import time

os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/')

## Define functions
# Load model data
def load_mp(config, vars):
    ''' Import microphysical quantities from the OFCAP/January long runs.

    Inputs:
    - config: a string that all files should contain that identifies the model configuration, e.g. 'lg_t'
    - vars: a string that tells the scripts which variables to load - should be either 'water paths', 'mass fractions' or 'both'.

    Outputs: a dictionary containing all the necessary variables to plot for your requested variable.

    Author: Ella Gilbert, 2018.

    '''
    start = time.time()
    pb = []
    pa = []
    pf = []
    print('\nimporting data from %(config)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test/'):
            if fnmatch.fnmatch(file, '*%(config)s_pb*' % locals()):
                pb.append(file)
            elif fnmatch.fnmatch(file, '*%(config)s_pa*' % locals()):
                pa.append(file)
            elif fnmatch.fnmatch(file, '*%(config)s_pf*' % locals()):
                pf.append(file)
    if vars == 'water paths':
        # Load only last 12 hours of forecast (i.e. t+12 to t+24, discarding preceding 12 hours as spin-up) for bottom 40 levels,
        # over the coordinates of the ice shelf (in rotated pole coordinates) and perform unit conversion from kg kg-1 to g kg-1.
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test/')
        print('\nice water path')
        try:
            IWP = iris.load_cube(pb, iris.AttributeConstraint(STASH='m01s02i392') & iris.Constraint(grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9, forecast_period=lambda cell: cell >= 12.5))# stash code s02i392
        except iris.exceptions.ConstraintMismatchError:
            print('\n IWP not in this file')
        print('\nliquid water path')
        try:
            LWP = iris.load_cube(pb, iris.AttributeConstraint(STASH='m01s02i391') & iris.Constraint(grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9, forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n LWP not in this file')
        for j in [LWP, IWP,]:
            j.convert_units('g m-2')
        mean_IWP = np.mean(IWP.data, axis=(2, 3))
        mean_LWP = np.mean(LWP.data, axis=(2, 3))
        AWS14_mean_IWP = np.mean(IWP[:, :,165:167, 98:100].data, axis = (2,3))
        AWS14_mean_LWP = np.mean(LWP[:, :,165:167, 98:100].data, axis = (2,3))
        AWS15_mean_IWP = np.mean(IWP[:, :,127:129, 81:83].data, axis = (2,3))
        AWS15_mean_LWP = np.mean(LWP[:, :,127:129, 81:83].data, axis = (2,3))
    elif vars == 'mass fractions':
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/') # quicker
        print('\nice mass fraction')
        try:
            ice_mass_frac = iris.load_cube(pb, iris.Constraint(name='mass_fraction_of_cloud_ice_in_air',
                                                               model_level_number=lambda cell: cell < 40,
                                                               grid_longitude=lambda cell: 178.5 < cell < 180.6,
                                                               grid_latitude=lambda cell: -2.5 < cell < 0.9))  # ,forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n QCF not in this file')
        print('\nliquid mass fraction')
        try:
            liq_mass_frac = iris.load_cube(pb, iris.Constraint(name='mass_fraction_of_cloud_liquid_water_in_air',
                                                               model_level_number=lambda cell: cell < 40,
                                                               grid_longitude=lambda cell: 178.5 < cell < 180.6,
                                                               grid_latitude=lambda cell: -2.5 < cell < 0.9))  # , forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n QCL not in this file')
        for i in [ice_mass_frac, liq_mass_frac]:#, qc]:
            i.convert_units('g kg-1')
        ## ---------------------------------------- CREATE MODEL VERTICAL PROFILES ------------------------------------------ ##
        # Create mean vertical profiles for region of interest (Larsen C)
        print('\ncreating vertical profiles geez...')
        mean_QCF = np.mean(ice_mass_frac.data, axis=(0, 1, 3, 4))
        mean_QCL = np.mean(liq_mass_frac.data, axis=(0, 1, 3, 4))  # 0,2,3
        AWS14_mean_QCF = np.mean(ice_mass_frac[:, :, :40, 165:167, 98:100].data, axis=(0, 1, 3, 4))
        AWS14_mean_QCL = np.mean(liq_mass_frac[:, :, :40, 165:167, 98:100].data, axis=(0, 1, 3, 4))
        AWS15_mean_QCF = np.mean(ice_mass_frac[:, :, :40, 127:129, 81:83].data, axis=(0, 1, 3, 4))
        AWS15_mean_QCL = np.mean(liq_mass_frac[:, :, :40, 127:129, 81:83].data, axis=(0, 1, 3, 4))
        altitude = ice_mass_frac.coord('level_height').points / 1000
    elif vars == 'both':
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test/')
        print('\nice water path')  # as above, and convert from kg m-2 to g m-2
        try:
            IWP = iris.load_cube(pb, iris.AttributeConstraint(STASH='m01s02i392') & iris.Constraint(
                grid_longitude=lambda cell: 178.5 < cell < 180.6, grid_latitude=lambda cell: -2.5 < cell < 0.9,
                forecast_period=lambda cell: cell >= 12.5))  # stash code s02i392
        except iris.exceptions.ConstraintMismatchError:
            print('\n IWP not in this file')
        print('\nliquid water path')
        try:
            LWP = iris.load_cube(pb, iris.AttributeConstraint(STASH='m01s02i391') & iris.Constraint(
                grid_longitude=lambda cell: 178.5 < cell < 180.6, grid_latitude=lambda cell: -2.5 < cell < 0.9,
                forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n LWP not in this file')
        for j in [LWP, IWP, ]:
            j.convert_units('g m-2')
        mean_IWP = np.mean(IWP.data, axis=(2, 3))
        mean_LWP = np.mean(LWP.data, axis=(2, 3))
        AWS14_mean_IWP = np.mean(IWP[:, :, 165:167, 98:100].data, axis=(2, 3))
        AWS14_mean_LWP = np.mean(LWP[:, :, 165:167, 98:100].data, axis=(2, 3))
        AWS15_mean_IWP = np.mean(IWP[:, :, 127:129, 81:83].data, axis=(2, 3))
        AWS15_mean_LWP = np.mean(LWP[:, :, 127:129, 81:83].data, axis=(2, 3))
        print('\nice mass fraction')
        try:
            ice_mass_frac = iris.load_cube(pb, iris.Constraint(name='mass_fraction_of_cloud_ice_in_air',
                                                               model_level_number=lambda cell: cell < 40,
                                                               grid_longitude=lambda cell: 178.5 < cell < 180.6,
                                                               grid_latitude=lambda cell: -2.5 < cell < 0.9,
                                                               forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n QCF not in this file')
        print('\nliquid mass fraction')
        try:
            liq_mass_frac = iris.load_cube(pb, iris.Constraint(name='mass_fraction_of_cloud_liquid_water_in_air',
                                                               model_level_number=lambda cell: cell < 40,
                                                               grid_longitude=lambda cell: 178.5 < cell < 180.6,
                                                               grid_latitude=lambda cell: -2.5 < cell < 0.9,
                                                               forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n QCL not in this file')
        # Convert units and times to useful ones
        for i in [ice_mass_frac, liq_mass_frac]:  # , qc]:
            i.convert_units('g kg-1')
            i.coord('time').convert_units('hours since 2011-01-01 00:00')
        ## ---------------------------------------- CREATE MODEL VERTICAL PROFILES ------------------------------------------ ##
        # Create mean vertical profiles for region of interest (Larsen C)
        print('\ncreating vertical profiles geez...')
        mean_QCF = np.mean(ice_mass_frac.data, axis=(0, 1, 3, 4))
        mean_QCL = np.mean(liq_mass_frac.data, axis=(0, 1, 3, 4))  # 0,2,3
        AWS14_mean_QCF = np.mean(ice_mass_frac[:, :, :40, 165:167, 98:100].data, axis=(0, 1, 3, 4))
        AWS14_mean_QCL = np.mean(liq_mass_frac[:, :, :40, 165:167, 98:100].data, axis=(0, 1, 3, 4))
        AWS15_mean_QCF = np.mean(ice_mass_frac[:, :, :40, 127:129, 81:83].data, axis=(0, 1, 3, 4))
        AWS15_mean_QCL = np.mean(liq_mass_frac[:, :, :40, 127:129, 81:83].data, axis=(0, 1, 3, 4))
        altitude = ice_mass_frac.coord('level_height').points / 1000
    constr_lsm = iris.load_cube(pa, iris.Constraint(name ='land_binary_mask', grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9 ))[0,:,:]
    constr_orog = iris.load_cube(pa, iris.Constraint(name ='surface_altitude', grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9 ))[0,:,:]
    end = time.time()
    print('\nDone, in {:01d} secs'.format(int(end - start)))
    # Find max and min values at each model level
    #time_mean_QCF = np.mean(box_QCF, axis=0)
    #array = pd.DataFrame()
    #for each_lat in np.arange(74):
    #    for each_lon in np.arange(25):
    #        for each_time in np.arange(len(ice_mass_frac.coord('time').points)):
    #            m = pd.DataFrame(box_QCF[each_time, :, each_lat, each_lon])
    #            array = pd.concat([m, array], axis=1)
    #    max_QCF = array.max(axis=1)
    #    min_QCF = array.min(axis=1)
    # Calculate 95th percentile
    #ice_95 = np.percentile(array, 95, axis=1)
    #ice_5 = np.percentile(array, 5, axis=1)
    # Find max and min values at each model level
    #time_mean_QCL = np.mean(box_QCL, axis=0)
    #array = pd.DataFrame()
    #for each_lat in np.arange(74):
    #    for each_lon in np.arange(25):
    #        for each_time in np.arange(len(ice_mass_frac.coord('time').points)):
    #            m = pd.DataFrame(box_QCL[each_time, :, each_lat, each_lon])
    #            array = pd.concat([m, array], axis=1)
        #max_QCL = array.max(axis=1)
        #min_QCL = array.min(axis=1)
    # Calculate 95th percentile
    #liq_95 = np.percentile(array, 95, axis=1)
    #liq_5 = np.percentile(array, 5, axis=1)
    # Calculate PDF of ice and liquid water contents
    #liq_PDF = mean_liq.plot.density(color = 'k', linewidth = 1.5)
    #ice_PDF = mean_ice.plot.density(linestyle = '--', linewidth=1.5, color='k')
    if os.get_cwd() == '/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test':
        config_dict = {'constr_lsm': constr_lsm, 'constr_orog': constr_orog, 'altitude': altitude, 'mean_QCF': mean_QCF,
                    'mean_QCL': mean_QCL, 'AWS14_mean_QCF': AWS14_mean_QCF, 'AWS14_mean_QCL': AWS14_mean_QCL,
                    'AWS15_mean_QCF': AWS15_mean_QCF, 'AWS15_mean_QCL': AWS15_mean_QCL,'AWS14_mean_IWP': AWS14_mean_IWP,
                    'AWS15_mean_IWP': AWS15_mean_IWP, 'AWS15_mean_LWP': AWS15_mean_LWP, 'mean_IWP': mean_IWP, 'mean_LWP': mean_LWP}
    else:
        var_dict = { 'constr_lsm': constr_lsm, 'constr_orog': constr_orog, 'altitude': altitude, 'mean_QCF': mean_QCF,
                     'mean_QCL': mean_QCL, 'AWS14_mean_QCF': AWS14_mean_QCF, 'AWS14_mean_QCL': AWS14_mean_QCL,
                     'AWS15_mean_QCF': AWS15_mean_QCF, 'AWS15_mean_QCL': AWS15_mean_QCL}
    return  var_dict

# 'box_QCF': box_QCF, 'box_QCL': box_QCL,
#'cl_A': cl_A,'qc': qc,'ice_5': ice_5, 'ice_95': ice_95, 'liq_5': liq_5, 'liq_95': liq_95, 'min_QCF': min_QCF, 'max_QCF': max_QCF,'IWP': IWP, 'LWP':LWP,
# 'min_QCL': min_QCL, 'real_lon': real_lon, 'real_lat':real_lat,'box_mean_IWP': box_mean_IWP, 'box_mean_LWP': box_mean_LWP,'AWS14_mean_LWP': AWS14_mean_LWP,

def load_SEB(config):
    pa = []
    pf = []
    print('\nimporting data from %(config)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test/'):
        if fnmatch.fnmatch(file,  '*%(config)s_pf*' % locals()):
            pf.append(file)
        elif fnmatch.fnmatch(file,  '*%(config)s_pa*' % locals()):
            pa.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test/')
    print('\n downwelling longwave')
    try:
        LW_down = iris.load_cube(pf, iris.Constraint(name='surface_downwelling_longwave_flux',grid_longitude=lambda cell: 178.5 < cell < 180.6, grid_latitude=lambda cell: -2.5 < cell < 0.9,
                                                     forecast_period=lambda cell: cell >= 12.5))
    except iris.exceptions.ConstraintMismatchError:
        print('\n Downwelling LW not in this file')
    print('\ndownwelling shortwave')
    try:
        SW_down = iris.load_cube(pf, iris.Constraint(name='surface_downwelling_shortwave_flux_in_air',
                                                           grid_longitude=lambda cell: 178.5 < cell < 180.6,
                                                           grid_latitude=lambda cell: -2.5 < cell < 0.9, forecast_period=lambda cell: cell >= 12.5))
    except iris.exceptions.ConstraintMismatchError:
        print('\n Downwelling SW not in this file')
    # print('\nice water path') # as above, and convert from kg m-2 to g m-2
    # try:
    #    IWP = iris.load_cube(pb, iris.AttributeConstraint(STASH='m01s02i392') & iris.Constraint(grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9, forecast_period=lambda cell: cell >= 12.5))# stash code s02i392
    # except iris.exceptions.ConstraintMismatchError:
    #    print('\n IWP not in this file')
    # print('\nliquid water path')
    # try:
    #    LWP = iris.load_cube(pb, iris.AttributeConstraint(STASH='m01s02i391') & iris.Constraint(grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9, forecast_period=lambda cell: cell >= 12.5))
    # except iris.exceptions.ConstraintMismatchError:
    #    print('\n LWP not in this file')
    #lsm = iris.load_cube(pa, 'land_binary_mask')
    #orog = iris.load_cube(pa, 'surface_altitude')
    #LW_net = iris.load_cube(pf, 'surface_net_downward_longwave_flux')
    #SH =  iris.load_cube(pf, 'surface_upward_sensible_heat_flux')
    #LH = iris.load_cube(pf, 'surface_upward_latent_heat_flux')
    #LW_up = iris.load_cube(pf, 'upwelling_longwave_flux_in_air')
    #SW_up = iris.load_cube(pf, 'upwelling_shortwave_flux_in_air')
    #else:
    #    SW_net = iris.load_cube(pf, 'surface_net_downward_shortwave_flux')
    #    Ts = iris.load_cube(pa, 'surface_temperature')
    #    Ts.convert_units('celsius')
    #for i in [SW_up, LW_up,]:
    #    real_lon, real_lat = rotate_data(i, 2, 3)
    #for j in [SW_down, LW_down, LH, SH, LW_net]:#,SW_net_surf,  Ts
    #    real_lon, real_lat = rotate_data(j, 1, 2)
    #for k in [lsm, orog]:
    #    real_lon, real_lat = rotate_data(k, 0, 1)
    # Convert times to useful ones
    #for i in [SW_down, SW_up, LW_net, LW_down, LW_up, LH, SH]:#, Ts, SW_net_surf,
    #    i.coord('time').convert_units('hours since 2011-01-18 00:00')
    #LH = 0 - LH.data
    #SH = 0 - SH.data
    #if config =='CASIM_24':
    #    var_dict = {'real_lon': real_lon, 'real_lat': real_lat, 'SW_up': SW_up, 'SW_down': SW_down,
    #                'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down, 'LW_net': LW_net, 'SW_net': SW_net}
    #else:
    #    var_dict = {'real_lon': real_lon, 'real_lat':real_lat,  'SW_up': SW_up, 'SW_down': SW_down,
    #                'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down, 'LW_net': LW_net, 'SW_net': SW_net, 'Ts': Ts}
    var_dict = {'LW_down': LW_down, 'SW_down': SW_down}
    return var_dict

def load_met(var):
    start = time.time()
    pa = []
    print('\nimporting data from %(var)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/'):
        if fnmatch.fnmatch(file, '*%(var)s*_pa*' % locals()):
            pa.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/')
    print('\nAir temperature')
    # Load only last 12 hours of forecast (i.e. t+12 to t+24, discarding preceding 12 hours as spin-up) for bottom 40 levels, and perform unit conversion from K to *C
    T_air = iris.load_cube(pa, iris.Constraint(name='air_temperature', model_level_number=lambda cell: cell <= 40, forecast_period=lambda cell: cell >= 12.5,
                                               grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9))
    T_air.convert_units('celsius')
    print('\nAir potential temperature')
    theta = iris.load_cube(pa, iris.Constraint(name='air_potential_temperature', model_level_number=lambda cell: cell <= 40, forecast_period=lambda cell: cell >= 12.5,
                                               grid_longitude=lambda cell: 178.5 < cell < 180.6,
                                               grid_latitude=lambda cell: -2.5 < cell < 0.9))
    theta.convert_units('celsius')
    print('\nSurface temperature')
    Ts = iris.load_cube(pa, iris.Constraint(name='surface_temperature', forecast_period=lambda cell: cell >= 12.5,
                                            grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9))
    Ts.convert_units('celsius')
    print('\nSpecific humidity')
    q = iris.load_cube(pa, iris.Constraint(name='specific_humidity', model_level_number=lambda cell: cell <= 40, forecast_period=lambda cell: cell >= 12.5,
                                           grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9,))
    q.convert_units('g kg-1') # Convert to g kg-1
    print('\nMean sea level pressure')
    MSLP = iris.load_cube(pa, iris.Constraint(name = 'air_pressure_at_sea_level')  & iris.Constraint(forecast_period=lambda cell: cell >= 12.5))
    MSLP.convert_units('hPa')
    print('\nZonal component of wind')
    u = iris.load(pa, iris.Constraint(name = 'x_wind', forecast_period=lambda cell: cell >= 12.5))[0]
    print('\nMeridional component of wind')
    v = iris.load(pa, iris.Constraint(name = 'y_wind', forecast_period=lambda cell: cell >= 12.5))[0]
    print('\nLSM')
    lsm = iris.load_cube(pa, 'land_binary_mask')
    print('\nOrography')
    orog = iris.load_cube(pa, 'surface_altitude')
    for i in [theta, T_air, u, v, q]: # 4-D variables
        real_lon, real_lat = rotate_data(i, 3, 4)
    for j in [Ts, MSLP]:  # 3-D variables
        real_lon, real_lat = rotate_data(j, 2, 3)  # time vars don't load in properly = forecast time + real time
    for k in [lsm, orog]: # 2-D variables
        real_lon, real_lat = rotate_data(k, 0, 1)
    # Convert times to useful ones
    print('\nConverting times...')
    for i in [theta, T_air, Ts, u, v, q, MSLP]:
        i.coord('time').convert_units('hours since 2011-01-01 00:00')
    # Create spatial means for maps
    print('\nCalculating means...')
    mean_MSLP = np.mean(MSLP.data, axis = (0,1))
    mean_Ts = np.mean(Ts.data, axis = (0,1))
    # Sort out time series loading
    def construct_srs(cube):
        i = np.arange(len(cube.coord('forecast_period')))
        k = cube.data
        series = []
        for j in i:
            a = k[:, j,:,:,:]
            a = np.array(a)
            series = np.append(series, a)
        return series
    # Produce time series
    print('\nCreating time series...')
    AWS14_Ts = Ts[:,:,200,200]
    AWS14_Ts_srs = construct_srs(AWS14_Ts)
    AWS14_Tair = T_air[:,:,0, 200,200]
    AWS14_Tair_srs = construct_srs(AWS14_Tair)
    AWS15_Ts = Ts[:,:,162,183]
    AWS15_Ts_srs = construct_srs(AWS14_Ts)
    AWS15_Tair = T_air[:,:,0, 162,183]
    AWS15_Tair_srs = construct_srs(AWS14_Tair)
    ## ---------------------------------------- CREATE MODEL VERTICAL PROFILES ------------------------------------------ ##
    # Create mean vertical profiles for region of interest
    # region of interest = ice shelf. Longitudes of ice shelf along transect =
    # OR: region of interest = only where aircraft was sampling layer cloud: time 53500 to 62000 = 14:50 to 17:00
    # Define box: -62 to -61 W, -66.9 to -68 S
    # Coord: lon = 188:213, lat = 133:207, time = 4:6 (mean of preceding hours)
    print('\ncreating vertical profiles...\n\nBox means first...')
    box_T = np.mean(T_air[:, :, :, 133:207, 188:213].data, axis=(0, 1, 3, 4))
    box_theta = np.mean(theta[:, :, :, 133:207, 188:213].data, axis=(0, 1, 3, 4))
    box_q = np.mean(q[:, :, :, 133:207, 188:213].data, axis=(0, 1, 3, 4))
    print('\nNow for AWS 14...')
    AWS14_mean_T = np.mean(T_air[:, :, 40, 199:201, 199:201].data, axis=(0, 1, 3, 4))
    AWS14_mean_theta = np.mean(theta[:, :, 40, 199:201, 199:201].data, axis=(0, 1, 3, 4))
    AWS14_mean_q= np.mean(q[:, :, 40, 199:201, 199:201].data, axis=(0, 1, 3, 4))
    print('\nLast bit! Repeating for AWS 15...')
    AWS15_mean_T = np.mean(T_air[:, :, 40, 161:163, 182:184].data, axis=(0, 1, 3, 4))
    AWS15_mean_theta = np.mean(theta[:, :, 40, 161:163, 182:184].data, axis=(0, 1, 3, 4))
    AWS15_mean_q= np.mean(q[:, :, 40, 161:163, 182:184].data, axis=(0, 1, 3, 4))
    altitude = T_air.coord('level_height').points[:40] / 1000
    var_dict = {'real_lon': real_lon, 'real_lat': real_lat, 'lsm': lsm, 'orog': orog, 'altitude': altitude, 'box_T': box_T,
                'box_theta': box_q, 'AWS14_mean_T': AWS14_mean_T, 'AWS14_mean_theta': AWS14_mean_theta, 'AWS14_mean_q': AWS14_mean_q,
                'AWS15_mean_T': AWS15_mean_T, 'AWS15_mean_theta': AWS15_mean_theta, 'AWS15_mean_q': AWS15_mean_q,
                'AWS14_Ts_srs': AWS14_Ts_srs, 'AWS14_Tair_srs': AWS14_Tair_srs, 'AWS15_Ts_srs': AWS15_Ts_srs, 'AWS15_Tair_srs': AWS15_Tair_srs}
    end = time.time()
    print
    '\nDone, in {:01d} secs'.format(int(end - start))
    return var_dict

Jan_mp = load_mp('lg_t', vars = 'water paths')
Jan_SEB = load_SEB('lg_t')
#Jan_met = load_met('lg_t')


def load_AWS(station, period):
    ## --------------------------------------------- SET UP VARIABLES ------------------------------------------------##
    ## Load data from AWS 14 and AWS 15 for January 2011
    print('\nDayum grrrl, you got a sweet AWS...')
    os.chdir('/data/clivarm/wip/ellgil82/AWS/')
    for file in os.listdir('/data/clivarm/wip/ellgil82/AWS/'):
        if fnmatch.fnmatch(file, '%(station)s_Jan_2011*' % locals()):
            AWS = pd.read_csv(str(file), header = 0)
            print(AWS.shape)
    if period == 'January':
        Jan18 = AWS.loc[(AWS['Day'] <= 31)]# or ((AWS['month'] == 2) * (AWS['Day'] >= 7))]
    elif period == 'OFCAP':
        Jan18 = AWS.loc[(AWS['Day'] <= 38)]
    return Jan18

AWS15_Jan = load_AWS('AWS15', period = 'January')
AWS14_SEB_Jan  = load_AWS('AWS14_SEB', period = 'January')


def print_stats():
    model_mean = pd.DataFrame()
    for run in Jan_mp:
        #print('\n\nMean cloud box QCL of %(run)s is: '% locals()+str(np.mean(run['mean_QCL'])) )
        #print('\n\nMean cloud box QCF of %(run)s is: '% locals()+str(np.mean(run['mean_QCF'])) )
        #print('\n\nMean QCL at AWS 14 and 15 is ' + str(np.mean(run['AWS14_mean_QCL']))+ ' and ' + str(np.mean(run['AWS15_mean_QCL'])) + ', respectively in %(run)s' % locals())
        #print ('\n\nMean QCF at AWS 14 and 15 is '+str(np.mean(run['AWS14_mean_QCF']))+' and '+str( np.mean(run['AWS15_mean_QCF']))+', respectively in %(run)s \n\n' % locals())
        #print('\n\nMean cloud box LWP of %(run)s is: ' % locals() + str(run['box_mean_LWP']))
        #print('\n\nMean cloud box IWP of %(run)s is: ' % locals() + str(run['box_mean_IWP']))
        #print('\n\nMean LWP at AWS 14 and 15 is ' + str(run['AWS14_mean_LWP']) + ' and ' + str(run['AWS15_mean_LWP']) + ', respectively in %(run)s' % locals())
        #print('\n\nMean IWP at AWS 14 and 15 is ' + str(run['AWS14_mean_IWP']) + ' and ' + str(run['AWS15_mean_IWP']) + ', respectively in %(run)s \n\n' % locals())
        m = pd.DataFrame({'mean QCL': np.mean(run['mean_QCL']), 'mean_QCF': np.mean(run['mean_QCF']), 'AWS 14 QCL': np.mean(run['AWS14_mean_QCL']), 'AWS 15 QCL': np.mean(run['AWS15_mean_QCL']),
                          'AWS 14 QCF' : np.mean(run['AWS14_mean_QCF']), 'AWS 15 QCF' : np.mean(run['AWS15_mean_QCF']), 'mean LWP': run['box_mean_LWP'], 'mean IWP': run['box_mean_IWP'],
                          'AWS 14 LWP': run['AWS14_mean_LWP'],  'AWS 14 IWP': run['AWS14_mean_IWP'], 'AWS 15 LWP': run['AWS15_mean_LWP'],  'AWS 15 IWP': run['AWS15_mean_IWP']}, index = [0])
        model_mean = pd.concat([model_mean, m])
        means = model_mean.mean(axis=0)
        print means

#print_stats()


def construct_srs(var_name):
    i = np.arange(var_name.shape[1])
    k = var_name
    series = []
    for j in i:
        a = k[:, j]
        a = np.array(a)
        series = np.append(series, a)
    return series

IWP14_srs = construct_srs(Jan_mp['AWS14_mean_IWP'])
LWP14_srs = construct_srs(Jan_mp['AWS14_mean_LWP'])
IWP15_srs = construct_srs(Jan_mp['AWS15_mean_IWP'])
LWP15_srs = construct_srs(Jan_mp['AWS15_mean_LWP'])
box_IWP_srs = construct_srs(Jan_mp['mean_IWP'])
box_LWP_srs = construct_srs(Jan_mp['mean_LWP'])
AWS14_SW_srs = construct_srs(np.mean(Jan_SEB['SW_down'][:,:,165:167, 98:100].data, axis = (2,3)))
AWS14_LW_srs = construct_srs(np.mean(Jan_SEB['LW_down'][:,:,165:167, 98:100].data, axis = (2,3)))
AWS15_SW_srs = construct_srs(np.mean(Jan_SEB['SW_down'][:,:,127:129, 81:83].data, axis = (2,3)))
AWS15_LW_srs = construct_srs(np.mean(Jan_SEB['LW_down'][:,:,127:129, 81:83].data, axis = (2,3)))
box_LW_srs = construct_srs(np.mean(Jan_SEB['LW_down'].data, axis = (2,3)))
box_SW_srs = construct_srs(np.mean(Jan_SEB['SW_down'].data, axis = (2,3)))
Jan_SEB['SW_down'].coord('time').convert_units('seconds since 1970-01-01 00:00:00')
Time_srs = construct_srs(np.swapaxes(Jan_SEB['SW_down'].coord('time').points,0,1))
Time_srs = matplotlib.dates.num2date(matplotlib.dates.epoch2num(Time_srs))


def load_obs():
    ## ----------------------------------------------- SET UP VARIABLES --------------------------------------------------##
    ## Load core data
    print('\nYes yes cuzzy, pretty soon you\'re gonna have some nice core data...')
    bsl_path_core = '/data/mac/ellgil82/cloud_data/Constantino_Oasis_Peninsula/flight152/core_masin_20110118_r001_flight152_1hz.nc'
    cubes = iris.load(bsl_path_core)
    RH = iris.load_cube(bsl_path_core, 'relative_humidity')
    core_temp = cubes[34] #de-iced temperature
    core_temp = core_temp.data[84:15432] # trim so times match
    core_temp = core_temp -273.15
    plane_lat = iris.load_cube(bsl_path_core, 'latitude')
    plane_lat = plane_lat.data[84:15432]
    plane_lon = iris.load_cube(bsl_path_core, 'longitude')
    plane_lon = plane_lon.data[84:15432]
    plane_alt = iris.load_cube(bsl_path_core, 'altitude')
    plane_alt = plane_alt.data[84:15432]
    core_time =  iris.load_cube(bsl_path_core, 'time')
    core_time = core_time.data[84:15432]
    ## Load CIP data
    # Load CIP from .npz
    print('\nOi mate, right now I\'m loading some siiiiick CIP data...')
    path = '/data/mac/ellgil82/cloud_data/Constantino_Oasis_Peninsula/'
    s_file = 'flight152_s_v2.npz'
    npz_s=np.load(path+s_file)
    m_file = 'flight152_m_v2.npz'
    npz_m = np.load(path + m_file)
    n_file = 'flight152_n_v2.npz'
    npz_n = np.load(path + n_file)
    CIP_time = npz_m['time']
    CIP_bound = npz_s['TestPlot_all_y']
    m_all = npz_m['TestPlot_all_y']
    IWC = npz_m['TestPlot_HI_y']+ npz_m['TestPlot_MI_y']
    S_LI = npz_m['TestPlot_LI_y']+npz_m['TestPlot_S_y']
    n_drop_CIP = npz_n['TestPlot_LI_y']+npz_n['TestPlot_S_y']
    # Load CAS data
    CAS_file = '/data/mac/ellgil82/cloud_data/netcdfs/flight152_cas.nc'
    # Create variables
    print ('\nOn dis CAS ting...')
    LWC_cas = iris.load_cube(CAS_file, 'liquid water content calculated from CAS ')
    LWC_cas = LWC_cas.data
    CAS_time = iris.load_cube(CAS_file, 'time')
    CAS_time = CAS_time.data
    aer = iris.load_cube(CAS_file, 'Aerosol concentration spectra measured by cas ')
    n_drop_CAS = np.nansum(aer[8:,:].data, axis=0)
    n_drop =  n_drop_CAS[:15348]
    ## ----------------------------------------- PERFORM CALCULATIONS ON DATA --------------------------------------------##
    # Find number concentrations of ice only
    n_ice = npz_s['TestPlot_HI_z']+npz_s['TestPlot_MI_z']
    n_ice = n_ice * 2. # correct data (as advised by TLC and done by Constantino for their 2016 and 2017 papers)
    n_ice = n_ice/1000 #in cm-3
    n_ice = n_ice[1:]
    CIP_mean_ice = []
    j = np.arange(64)
    for i in j:#
        m = np.mean(n_ice[:,i])
        CIP_mean_ice = np.append(CIP_mean_ice,m)
    # Convert times
    unix_time = 1295308800
    CIP_real_time = CIP_time + unix_time
    s = pd.Series(CIP_real_time)
    CIP_time = pd.to_datetime(s, unit='s')
    core_time = core_time + unix_time
    core_time = pd.Series(core_time)
    core_time = pd.to_datetime(core_time, unit='s')
    CAS_time = np.ndarray.astype(CAS_time, float)
    CAS_time = CAS_time / 1000
    CAS_real_time = CAS_time + unix_time
    s = pd.Series(CAS_real_time)
    CAS_time = pd.to_datetime(s, unit='s')
    # Make times match
    CAS_time_short = CAS_time[:15348]
    CIP_time_short = CIP_time[1:]
    ## ------------------------------------- COMPUTE WHOLE-FLIGHT STATISTICS ---------------------------------------------##
    # FIND IN-CLOUD LEGS
    # Find only times when flying over ice shelf
    print('\nYEAH BUT... IS IT CLOUD DOE BRUH???')
    idx = np.where(plane_lon.data > -62) # only in the region of interest (box)
    idx = idx[0]
    # Find only times when flying in cloud
    # Find indices of gridcells where cloud is present
    def is_cloud(): #should be a range of time indices e.g. [87:1863]
        cloud_bins = aer[8:,idx[0]:idx[-1]].data # particles > 1.03 um
        cl_sum = []
        for each_sec in np.arange(len(cloud_bins[0,:])): # at each second
            f = np.sum(cloud_bins[:, each_sec])
            cl_sum = np.append(cl_sum,f)
            cloud_idx = np.nonzero(cl_sum)
            cloud_idx = cloud_idx[0]
        return cloud_idx
    cloud_idx = is_cloud()
    cloud_idx = cloud_idx + idx[0] # Calculate indices relative to length of original dataset, not subset
    # Create array of only in-cloud data within box
    IWC_array = []
    LWC_array = []
    alt_array_ice = []
    alt_array_liq = []
    nconc_ice = []
    drop_array = []
    nconc_ice_all = np.sum(n_ice, axis=1)
    # Calculate number concentration averages only when the instrument detects liquid/ice particles
    for i in cloud_idx:
        if nconc_ice_all[i] > 0.00000001:
            IWC_array.append(IWC[i])
            alt_array_ice.append(plane_alt[i])
            nconc_ice.append(nconc_ice_all[i])
    for i in cloud_idx:
        if n_drop[i] > 1.0: # same threshold as in Lachlan-Cope et al. (2016)
            drop_array.append(n_drop[i])
            LWC_array.append(LWC_cas[i])
            alt_array_liq.append(plane_alt[i])
        #else:
        #    print('naaaaaah mate, nutn \'ere like')
    box_nconc_liq = np.mean(drop_array)
    box_nconc_ice = np.mean(nconc_ice)
    box_LWC = np.nanmean(LWC_array)
    box_IWC = np.nanmean(IWC_array)
    # Calculate mean values at each height in the model
    # Create bins from model data
    print('\nbinning by altitude...')
    #Load model data to get altitude bins
    ice_mass_frac = iris.load_cube('/data/mac/ellgil82/cloud_data/um/means/20110118T0000Z_Peninsula_km1p5_Smith_tnuc_pc000.pp', 'mass_fraction_of_cloud_ice_in_air')
    bins =  ice_mass_frac.coord('level_height').points.tolist()
    # Find index of model level bin to which aircraft data would belong and turn data into pandas dataframe
    icy = {'alt_idx': np.digitize(alt_array_ice, bins = bins), 'IWC': IWC_array,  'n_ice': nconc_ice}
    watery = {'alt_idx': np.digitize(alt_array_liq, bins = bins), 'LWC': LWC_array, 'n_drop': drop_array}
    ice_df = pd.DataFrame(data = icy)
    liq_df = pd.DataFrame(data = watery)
    print('\ncreating observed profiles...')
    # Use groupby to group by altitude index and mean over the groups
    ice_grouped = ice_df.groupby(['alt_idx']).mean()
    liq_grouped = liq_df.groupby(['alt_idx']).mean()
    IWC_profile = ice_grouped['IWC'].values
    IWC_profile = np.append(IWC_profile, [0,0,0,0])
    IWC_profile = np.append([0,0,0], IWC_profile)
    LWC_profile = liq_grouped['LWC'].values
    LWC_profile = np.insert(LWC_profile, 2, [0, 0])
    LWC_profile = np.insert(LWC_profile, 22, [0,0,0,0] )
    LWC_profile = np.append(LWC_profile, [0,0,0,0,0,0])
    LWC_profile = np.append([0,0,0,0], LWC_profile)
    drop_profile = liq_grouped['n_drop'].values
    drop_profile = np.insert(drop_profile, 2, [0, 0])
    drop_profile = np.insert(drop_profile, 22, [0,0,0,0] )
    drop_profile = np.append(drop_profile, [0,0,0,0,0,0])
    drop_profile = np.append([0,0,0,0],drop_profile)
    n_ice_profile = ice_grouped['n_ice'].values
    n_ice_profile = np.append(n_ice_profile, [0,0,0,0])
    n_ice_profile = np.append([0,0,0], n_ice_profile)
    return IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile


## ================================================= PLOTTING ======================================================= ##

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans',
                               'Verdana']
## Caption: mean modelled water paths (in g kg-1) over the Larsen C ice shelf during Jan 2011

def column_totals():
    fig, ax = plt.subplots(6,2, sharex=True, sharey=True, figsize=(12, 28), frameon=False)
    ax = ax.flatten()
    for axs in ax:
        axs.axis('off')
    plot = 0
    CbAx_ice = fig.add_axes([0.15, 0.94, 0.33, 0.015])
    CbAx_liq = fig.add_axes([0.55, 0.94, 0.33, 0.015])
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l'}
    for run in model_runs:
        mesh_ice = ax[plot].pcolormesh(np.mean(run['IWP'][16:25, :,:].data, axis = (0)), cmap='Blues_r', vmin=0., vmax=300) # check times!
        ax[plot].contour(run['lsm'].data, colors='#A6ACAF', lw=2)
        ax[plot].contour(run['orog'].data, levels=[10], colors='#A6ACAF', lw=2)
        ax[plot].text(x=30, y=320, s=lab_dict[plot], color='#A6ACAF', fontweight = 'bold',  fontsize=32)
        mesh_liq = ax[plot+1].pcolormesh(np.mean(run['LWP'][16:25, :,:].data, axis = (0)), cmap='Blues', vmin=0., vmax=300) # check times!
        ax[plot+1].contour(run['lsm'].data, colors='0.3', lw=2)
        ax[plot+1].contour(run['orog'].data, levels=[10], colors='0.3', lw=2)
        ax[plot+1].text(x=30, y=320, s=lab_dict[plot+1], color='dimgrey', fontweight = 'bold', fontsize=32)
        plot = plot + 2
    cb_ice = plt.colorbar(mesh_ice, orientation='horizontal', cax=CbAx_ice, ticks=[0, 300])#, format='.0f')
    cb_liq = plt.colorbar(mesh_liq, orientation='horizontal', cax=CbAx_liq, ticks=[0, 300])#, format='.0f')
    CbAx_ice.set_xlabel('Ice water path (g m$^{-2}$)', fontname='Helvetica', color='dimgrey', fontsize=24, labelpad=10)
    CbAx_liq.set_xlabel('Liquid water path (g m$^{-2}$)', fontname='Helvetica', color='dimgrey', fontsize=24, labelpad=10)
    for cb in [cb_ice, cb_liq]:
        cb.solids.set_edgecolor("face")
        cb.outline.set_edgecolor('dimgrey')
        cb.ax.tick_params(which='both', axis='both', labelsize=28, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
        cb.outline.set_linewidth(2)
        cb.ax.xaxis.set_ticks_position('top')
        #[l.set_visible(False) for (i, l) in enumerate(cb.ax.xaxis.get_ticklabels()) if i % 4 != 0]
    labels = [item.get_text() for item in cb_ice.ax.get_xticklabels()]
    labels[-1] = '300'
    labels[0] = '0'
    cb_ice.ax.set_xticklabels(labels)
    labels = [item.get_text() for item in cb_liq.ax.get_xticklabels()]
    labels[-1] = '300'
    labels[0] = '0'
    cb_liq.ax.set_xticklabels(labels)
    #cb_ice.ax.xaxis.get_major_ticks()[1].label1.set_horizontalalignment('left')
    plt.subplots_adjust(hspace=0.08, wspace=0.08, top=0.88)
    #ax[0].set_title('Total column ice', fontname='Helvetica', color='dimgrey', fontsize=28, )
    #ax[1].set_title('Total column liquid', fontname='Helvetica', color='dimgrey', fontsize=28, )
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/v11_water_paths_Jan_2011.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/v11_water_paths_Jan_2011.eps', transparent=True)
    #plt.show()

#column_totals()

def mod_profile():
    fig, ax = plt.subplots(1,2, figsize=(16, 9))
    ax = ax.flatten()
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axs.set_ylim(0, max(Jan_mp['altitude']))
        #[l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    m_QCF = ax[0].plot(Jan_mp['mean_QCF'], Jan_mp['altitude'], color = 'k', linestyle = '--', linewidth = 2.5)
    ax[0].set_xlabel('Cloud ice mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    ax[0].set_ylabel('Altitude \n(km)', rotation = 0, fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 80)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0].set_xlim(0,0.02)
    ax[0].xaxis.get_offset_text().set_fontsize(24)
    ax[0].xaxis.get_offset_text().set_color('dimgrey')
    #ax[0].fill_betweenx(Jan_mp['altitude'], Jan_mp['ice_5'], Jan_mp['ice_95'], facecolor='lightslategrey', alpha=0.5)  # Shaded region between maxima and minima
    #ax[0].plot(Jan_mp['ice_5'], Jan_mp['altitude'], color='darkslateblue', linestyle=':', linewidth=2)
    #ax[0].plot(Jan_mp['ice_95'], Jan_mp['altitude'], color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
    ax[0].text(0.1, 0.85, transform = ax[0].transAxes, s='a', fontsize=32, fontweight='bold', color='dimgrey')
    m_14 = ax[0].plot(Jan_mp['AWS14_mean_QCF'], Jan_mp['altitude'], color='darkred', linestyle='--', linewidth=3)
    m_15= ax[0].plot(Jan_mp['AWS15_mean_QCF'], Jan_mp['altitude'], color='darkblue', linestyle='--', linewidth=3)
    ax[1].set_xlabel('Cloud liquid mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    m_QCL = ax[1].plot(Jan_mp['mean_QCL'], Jan_mp['altitude'], color = 'k', linestyle = '--', linewidth = 2.5, label = 'Model: \'cloud\' box mean')
    #ax[1].fill_betweenx(Jan_mp['altitude'], Jan_mp['liq_5'], Jan_mp['liq_95'],  facecolor='lightslategrey', alpha=0.5, label = 'Model: 5$^{th}$ & 95$^{th}$ percentiles\n of \'cloud\' box range')  # Shaded region between maxima and minima
    #ax[1].plot(Jan_mp['liq_5'], Jan_mp['altitude'], color='darkslateblue', linestyle=':', linewidth=2, label='')
    #ax[1].plot(Jan_mp['liq_95'], Jan_mp['altitude'], color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
    m_14 = ax[1].plot(Jan_mp['AWS14_mean_QCL'], Jan_mp['altitude'], color='darkred', linestyle='--', linewidth=3, label='Model: AWS 14')
    m_15 = ax[1].plot(Jan_mp['AWS15_mean_QCL'], Jan_mp['altitude'], color='darkblue', linestyle='--', linewidth=3, label='Model: AWS 15')
    from matplotlib.ticker import ScalarFormatter
    class ScalarFormatterForceFormat(ScalarFormatter):
        def _set_format(self, vmin, vmax):  # Override function that finds format to use.
            self.format = "%1.1f"  # Give format here
    xfmt = ScalarFormatterForceFormat()
    xfmt.set_powerlimits((0, 0))
    ax[1].xaxis.set_major_formatter(xfmt)
    ax[1].axes.tick_params(axis = 'both', which = 'both', direction = 'in', length = 5, width = 1.5,  labelsize = 24, pad = 10)
    ax[1].tick_params(labelleft = 'off')
    ax[1].set_xlim(0, 0.41)
    ax[1].text(0.1, 0.85, transform = ax[1].transAxes, s='b', fontsize=32, fontweight='bold', color='dimgrey')
    plt.subplots_adjust(wspace=0.1, bottom=0.23, top=0.95, left=0.17, right=0.98)
    handles, labels = ax[1].get_legend_handles_labels()
    handles = [handles[0], handles[1], handles[-1]]#, handles[2],  handles[3] ]
    labels = [labels[0], labels[1], labels[-1]]#, labels[2], labels[3]]
    lgd = plt.legend(handles, labels, fontsize=20, markerscale=2)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vertical_profiles_OFCAP.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vertical_profiles_OFCAP.png', transparent = True)
    plt.show()

mod_profile()


def T_plot():
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    m_QCL = ax.plot(Jan_met['box_T'], Jan_met['altitude'], color='k', linestyle = '--', linewidth=3, label='Model: Cloud box')
    m_14 = ax.plot(Jan_met['AWS14_mean_T'], Jan_met['altitude'], color='darkred', linestyle = ':', linewidth=3, label='Model: AWS 14')
    m_15= ax.plot(Jan_met['AWS15_mean_T'], Jan_met['altitude'], color='darkblue', linestyle='--', linewidth=3, label='Model: AWS 15')
    #ax[plot].fill_betweenx(run['altitude'], run['liq_5'], run['liq_95'], facecolor='lightslategrey', alpha = 0.5)  # Shaded region between maxima and minima
    #ax[plot].plot(run['liq_5'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)
    #ax[plot].plot(run['liq_95'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
    ax.set_xlim(0, np.ceil(max(Jan_met['box_T'])))
    ax.set_ylim(0, max(Jan_met['altitude']))
    plt.setp(ax.get_xticklabels()[0], visible=False)
    ax.axes.tick_params(axis='both', which='both', tick1On=False, tick2On=False,)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    print('\n PLOTTING DIS BIATCH...')
    ax.set_ylabel('Altitude (km)', fontname='SegoeUI semibold', color = 'dimgrey', fontsize=28, labelpad=20)
    ax.set_xlabel('Air temperature ($^{\circ}$C)', fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=35)
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.12, right=0.95, hspace=0.12, wspace=0.08)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/v11_T_Jan_2011.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/v11_T_Jan_2011.eps')
    #plt.show()

#T_plot()

from itertools import chain
import scipy

def correl_plot():
    fig, ax = plt.subplots(len(model_runs), 2, sharex='col', figsize=(18, len(model_runs * 5) + 3))  # , squeeze=False)
    ax = ax.flatten()
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    plot = 0
    var_names = ['cloud \nice content', 'cloud \nliquid content']
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set(adjustable='box-forced', aspect='equal')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    for run in model_runs:
        slope, intercept, r2, p, sterr = scipy.stats.linregress(IWC_profile, run['mean_QCF'])
        if p <= 0.01:
            ax[plot].text(0.9, 0.15, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight = 'bold', transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        else:
            ax[plot].text(0.9, 0.15, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        ax[plot].scatter(IWC_profile, run['mean_QCF'], color = '#f68080', s = 50)
        ax[plot].set_xlim(min(chain(IWC_profile, run['mean_QCF'])), max(chain(IWC_profile, run['mean_QCF'])))
        ax[plot].set_ylim(min(chain(IWC_profile, run['mean_QCF'])), max(chain(IWC_profile, run['mean_QCF'])))
        ax[plot].plot(ax[plot].get_xlim(), ax[plot].get_ylim(), ls="--", c = 'k', alpha = 0.8)
        slope, intercept, r2, p, sterr = scipy.stats.linregress(LWC_profile, run['mean_QCL'])
        if p <= 0.01:
            ax[plot+1].text(0.9, 0.15, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight='bold', transform=ax[plot+1].transAxes,
                          size=24,
                          color='dimgrey')
        else:
            ax[plot+1].text(0.9, 0.15, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax[plot+1].transAxes, size=24,
                          color='dimgrey')
        ax[plot+1].scatter(LWC_profile, run['mean_QCL'], color='#f68080', s=50)
        ax[plot+1].set_xlim(min(chain(LWC_profile, run['mean_QCL'])), max(chain(LWC_profile, run['mean_QCL'])))
        ax[plot+1].set_ylim(min(chain(LWC_profile, run['mean_QCL'])), max(chain(LWC_profile, run['mean_QCL'])))
        ax[plot+1].plot(ax[plot+1].get_xlim(), ax[plot+1].get_ylim(), ls="--", c='k', alpha=0.8)
         #'r$^{2}$ = %s' % r2,
        ax[plot].set_xlabel('Observed %s' % var_names[0], size = 24, color = 'dimgrey', rotation = 0, labelpad = 10)
        ax[plot].set_ylabel('Modelled %s' % var_names[0], size = 24, color = 'dimgrey', rotation =0, labelpad= 80)
        ax[plot+1].set_xlabel('Observed %s' % var_names[1], size = 24, color = 'dimgrey', rotation = 0, labelpad = 10)
        ax[plot+1].set_ylabel('Modelled %s' % var_names[1], size = 24, color = 'dimgrey', rotation =0, labelpad= 80)
        lab = ax[plot].text(0.1, 0.85, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        lab2 = ax[plot+1].text(0.1, 0.85, transform = ax[plot+1].transAxes, s=lab_dict[plot+1], fontsize=32, fontweight='bold', color='dimgrey')
        titles = ['    RA1M','    RA1M','RA1M_mod','RA1M_mod','     fl_av', '     fl_av','    RA1T','    RA1T',  'RA1T_mod', 'RA1T_mod','   CASIM','   CASIM']
        ax[plot].text(0.83, 1.1, transform=ax[plot].transAxes, s=titles[plot], fontsize=28, color='dimgrey')
        plt.setp(ax[plot].get_xticklabels()[-2], visible=False)
        plt.setp(ax[plot].get_yticklabels()[-2], visible=False)
        ax[plot+1].yaxis.tick_right()
        [l.set_visible(False) for (w, l) in enumerate(ax[plot + 1].yaxis.get_ticklabels()) if w % 2 != 0]
        ax[plot].yaxis.set_label_coords(-0.6, 0.5)
        ax[plot+1].yaxis.set_label_coords(1.6, 0.5)
        ax[plot].spines['right'].set_visible(False)
        ax[plot+1].spines['left'].set_visible(False)
        plot = plot + 2
        plt.subplots_adjust(top = 0.98, hspace = 0.15, bottom = 0.05, wspace = 0.15, left = 0.25, right = 0.75)
    #plt.setp(ax[5].get_xticklabels()[-2], visible=False)
    #plt.setp(ax[6].get_xticklabels()[-2], visible=False)
    #plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    #plt.setp(ax[2].get_xticklabels()[-3], visible=False)
    #plt.setp(ax[2].get_yticklabels()[-1], visible=False)
    #plt.setp(ax[5].get_yticklabels()[-2], visible=False)
    #plt.setp(ax[6].get_yticklabels()[-2], visible=False)
    #plt.setp(ax[1].get_yticklabels()[-3], visible=False)
    #plt.setp(ax[2].get_yticklabels()[-3], visible=False)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/correlations_Jan_2011.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/correlations_Jan_2011.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/correlations_Jan_2011.pdf', transparent=True)
    #plt.show()

#correl_plot()

from matplotlib.lines import Line2D

def IWP_time_srs():
    model_runs = [Jan_mp]
    fig, ax = plt.subplots(2,1, sharex = True, figsize = (30,14))
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.spines['right'].set_visible(False)
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    plot = 0
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    for run in model_runs:
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/')
        print('\nPLOTTING DIS BIATCH...')
        ax[plot].spines['right'].set_visible(False)
        ax[plot].plot(Time_srs,IWP14_srs*1000, label = 'AWS14 IWP', linewidth = 2,  color = 'darkred')
        ax[plot].plot(Time_srs,IWP15_srs*1000, label='AWS15 IWP', linewidth=2, color='darkblue')
        ax[plot].plot(Time_srs,box_IWP_srs*1000, label='Cloud box IWP', linewidth=2, linestyle='--', color='k')
        lab = ax[plot].text(0.03, 0.85, transform=ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        ax[plot].set_xlim(Time_srs[0], Time_srs[-1])
        ax[plot].set_ylim(0,1050)
        ax[plot].set_yticks([0, 250, 500, 750, 1000])
        ax[plot].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        ax[plot+1].set_ylim(0,500)
        ax[plot+1].set_yticks([0, 125, 250, 375, 500])
        ax[plot+1].plot(Time_srs,LWP14_srs*1000, label = 'AWS14 LWP', linewidth = 2,  color = 'darkred')
        ax[plot+1].plot(Time_srs,LWP15_srs*1000, label='AWS15 LWP', linewidth=2,color='darkblue')
        ax[plot+1].plot(Time_srs,box_LWP_srs*1000, label='Cloud box LWP', linewidth=2, linestyle='--', color='k')
        ax[plot+1].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [w.set_linewidth(2) for w in ax[plot].spines.itervalues()]
        [w.set_linewidth(2) for w in ax[plot+1].spines.itervalues()]
        ax[plot+1].set_xlim(Time_srs[0], Time_srs[-1])
        lab = ax[plot+1].text(0.03, 0.85, transform=ax[plot+1].transAxes, s=lab_dict[plot+1], fontsize=32, fontweight='bold',color='dimgrey')
        print('\nDONE!')
        print('\nNEEEEEXT')
        plot = plot + 2
    ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    plt.setp(ax[1].get_xticklabels()[-1], visible=False)
    lns = [Line2D([0], [0], color='darkred', linewidth=3),
           Line2D([0], [0], color='darkblue', linewidth=3),
           Line2D([0], [0], color='k', linestyle='--', linewidth=3)]
    labs = [ 'AWS 14','AWS 15', 'Ice shelf mean']#  '                      ','                      '
    lgd = plt.legend(lns, labs, ncol=2, bbox_to_anchor=(1., 2.), borderaxespad=0., loc='best', prop={'size': 24})
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.97, wspace = 0.05, hspace = 0.1)
    fig.text(0.5, 0.04, 'Time (hours)', fontsize=24, fontweight = 'bold', ha = 'center', va = 'center', color = 'dimgrey')
    fig.text(0.03, 0.8, 'IWP \n(g m$^{-2}$)', fontsize=30, ha= 'center', va='center', rotation = 0, color = 'dimgrey')
    fig.text(0.03, 0.4, 'LWP \n(g m$^{-2}$)', fontsize=30, ha='center', va='center', color = 'dimgrey', rotation=0)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vn11_water_path_time_srs_Jan_2011.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vn11_water_path_time_srs_Jan_2011.eps')
    plt.show()


def rad_time_srs():
    model_runs = [Jan_mp]
    fig, ax = plt.subplots(2,1, sharex = True, figsize = (30,14))
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.spines['right'].set_visible(False)
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    plot = 0
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    for run in model_runs:
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/')
        print('\nPLOTTING DIS BIATCH...')
        ax[plot].spines['right'].set_visible(False)
        ax[plot].plot(Time_srs, AWS14_SW_srs, label = 'AWS14 SW$_{\downarrow}$: modelled', linewidth = 2,  linestyle = '--', color = 'darkred')
        #ax[plot].plot(Time_srs,AWS15_SW_srs, label='AWS15 SW$_{\downarrow}$', linewidth=2, color='darkblue')
        #ax[plot].plot(Time_srs,box_SW_srs, label='Cloud box SW$_{\downarrow}$', linewidth=2, linestyle='--', color='k')
        ax2 = ax[plot].twiny()
        ax2.plot(AWS14_SEB_Jan['SWin_corr'], label = 'AWS14 SW$_{\downarrow}$: observed', linewidth = 2,  color = 'darkred')
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.spines['top'].set_visible(False)
        plt.setp(ax2.spines.values(), linewidth=3, color='dimgrey')
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        lab = ax[plot].text(0.03, 0.85, transform=ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        ax[plot].set_xlim(Time_srs[0], Time_srs[-1])
        ax[plot].set_ylim(0,1050)
        ax[plot].set_yticks([0, 250, 500, 750, 1000])
        ax[plot].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        ax[plot+1].set_ylim(150,350)
        ax[plot+1].set_yticks([ 200, 300])
        ax[plot+1].plot(Time_srs,AWS14_LW_srs, label = 'AWS14 LW$_{\downarrow}$', linewidth = 2,  linestyle = '--',color = 'darkred')
        #ax[plot+1].plot(Time_srs,AWS15_LW_srs, label = 'AWS15_LW$_{\downarrow}$', linewidth=2,color='darkblue')
        #ax[plot+1].plot(Time_srs,box_LW_srs, label = 'Cloud box LW$_{\downarrow}$', linewidth=2, linestyle='--', color='k')
        ax[plot+1].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        ax2 = ax[plot+1].twiny()
        ax2.plot(AWS14_SEB_Jan['LWin'], label = 'AWS14 LW$_{\downarrow}$: observed', linewidth = 2,  color = 'darkred')
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.spines['top'].set_visible(False)
        plt.setp(ax2.spines.values(), linewidth=3, color='dimgrey')
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [w.set_linewidth(2) for w in ax[plot].spines.itervalues()]
        [w.set_linewidth(2) for w in ax[plot+1].spines.itervalues()]
        ax[plot+1].set_xlim(Time_srs[0], Time_srs[-1])
        lab = ax[plot+1].text(0.03, 0.85, transform=ax[plot+1].transAxes, s=lab_dict[plot+1], fontsize=32, fontweight='bold',color='dimgrey')
        print('\nDONE!')
        print('\nNEEEEEXT')
        plot = plot + 2
    ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    plt.setp(ax[1].get_xticklabels()[-1], visible=False)
    lns = [Line2D([0], [0], color='darkred', linewidth=3),
           Line2D([0], [0], color='darkblue', linewidth=3),
           Line2D([0], [0], color='k', linestyle='--', linewidth=3)]
    labs = [ 'AWS 14','AWS 15', 'Ice shelf mean']#  '                      ','                      '
    lgd = plt.legend(lns, labs, ncol=2, bbox_to_anchor=(1., 2.), borderaxespad=0., loc='best', prop={'size': 24})
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.97, wspace = 0.05, hspace = 0.1)
    fig.text(0.5, 0.04, 'Time (hours)', fontsize=24, fontweight = 'bold', ha = 'center', va = 'center', color = 'dimgrey')
    fig.text(0.03, 0.8, 'SW$_{\downarrow}$\n(W m$^{-2}$)', fontsize=30, ha= 'center', va='center', rotation = 0, color = 'dimgrey')
    fig.text(0.03, 0.4, 'LW$_{\downarrow}$\n(W m$^{-2}$)', fontsize=30, ha='center', va='center', color = 'dimgrey', rotation=0)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vn11_rad_time_srs_AWS14_Jan_2011.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vn11_rad_time_srs_AWS14_Jan_2011.eps')
    plt.show()

rad_time_srs()

def liq_time_srs():
    model_runs = [Jan_mp]
    fig, ax = plt.subplots(2,1, sharex = True, figsize = (30,14))
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.spines['right'].set_visible(False)
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    plot = 0
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    for run in model_runs:
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/')
        print('\nPLOTTING DIS BIATCH...')
        ax[plot].spines['right'].set_visible(False)
        ax[plot].plot(Time_srs, IWP14_srs * 1000, label='AWS14 IWP', linewidth=2, color='darkred')
        ax[plot].plot(Time_srs, IWP15_srs * 1000, label='AWS15 IWP', linewidth=2, color='darkblue')
        ax[plot].plot(Time_srs, box_IWP_srs * 1000, label='Cloud box IWP', linewidth=2, linestyle='--', color='k')
        lab = ax[plot].text(0.03, 0.85, transform=ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        ax[plot].set_xlim(Time_srs[0], Time_srs[-1])
        ax[plot].set_ylim(0,1050)
        ax[plot].set_yticks([0, 250, 500, 750, 1000])
        ax[plot].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        ax[plot+1].set_ylim(150,350)
        ax[plot+1].set_yticks([ 200, 300])
        ax[plot+1].plot(Time_srs,AWS14_LW_srs, label = 'AWS14 LW$_{\downarrow}$', linewidth = 2,  color = 'darkred')
        ax[plot+1].plot(Time_srs,AWS15_LW_srs, label = 'AWS15_LW$_{\downarrow}$', linewidth=2,color='darkblue')
        ax[plot+1].plot(Time_srs,box_LW_srs, label = 'Cloud box LW$_{\downarrow}$', linewidth=2, linestyle='--', color='k')
        ax[plot+1].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [w.set_linewidth(2) for w in ax[plot].spines.itervalues()]
        [w.set_linewidth(2) for w in ax[plot+1].spines.itervalues()]
        ax[plot+1].set_xlim(Time_srs[0], Time_srs[-1])
        lab = ax[plot+1].text(0.03, 0.85, transform=ax[plot+1].transAxes, s=lab_dict[plot+1], fontsize=32, fontweight='bold',color='dimgrey')
        print('\nDONE!')
        print('\nNEEEEEXT')
        plot = plot + 2
    ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    plt.setp(ax[1].get_xticklabels()[-1], visible=False)
    lns = [Line2D([0], [0], color='darkred', linewidth=3),
           Line2D([0], [0], color='darkblue', linewidth=3),
           Line2D([0], [0], color='k', linestyle='--', linewidth=3)]
    labs = [ 'AWS 14','AWS 15', 'Ice shelf mean']#  '                      ','                      '
    lgd = plt.legend(lns, labs, ncol=2, bbox_to_anchor=(1., 2.), borderaxespad=0., loc='best', prop={'size': 24})
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.97, wspace = 0.05, hspace = 0.1)
    fig.text(0.5, 0.04, 'Time (hours)', fontsize=24, fontweight = 'bold', ha = 'center', va = 'center', color = 'dimgrey')
    fig.text(0.03, 0.8, 'SW$_{\downarrow}$\n(W m$^{-2}$)', fontsize=30, ha= 'center', va='center', rotation = 0, color = 'dimgrey')
    fig.text(0.03, 0.4, 'LW$_{\downarrow}$\n(W m$^{-2}$)', fontsize=30, ha='center', va='center', color = 'dimgrey', rotation=0)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vn11_rad_time_srs_Jan_2011.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vn11_rad_time_srs_Jan_2011.eps')
    plt.show()

def correl_SEB_sgl(runSEB, runMP, phase):
    fig, ax = plt.subplots(figsize = (12,6))
    if phase == 'liquid':
        # LW vs LWP
        ax.set_xlim(260,330)
        ax.set_ylim(0,300)
        ax.scatter(AWS14_LW_srs, LWP14_srs, color='#f68080',s=50)
#        ax.set_ylim(np.min(np.mean(runMP['LWP'][:,:, 133:207, 188:213].data, axis=0)),
#                          np.max(np.mean(runMP['LWP'][:,:, 133:207, 188:213].data, axis=(0))))
#        ax.set_xlim(np.min(np.mean(runSEB['LW_down'][:,:, 133:207, 188:213].data, axis=(0))),
#                          np.max(np.mean(runSEB['LW_down'][:,:, 133:207, 188:213].data, axis=(0))))
        slope, intercept, r2, p, sterr = scipy.stats.linregress(AWS14_LW_srs, LWP14_srs)
        if p <= 0.01:
            ax.text(0.75, 0.9, horizontalalignment='right', verticalalignment='top', s='r$^{2}$ = %s' % np.round(r2, decimals=2),
                          fontweight='bold', transform=ax.transAxes, size=24,color='dimgrey')
        else:
            ax.text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax.transAxes, size=24, color='dimgrey')
        ax.set_xlabel('Modelled LW$_{\downarrow}$ (W m$^{-2}$)', size=24, color='dimgrey', rotation=0,labelpad=10)
        ax.set_ylabel('Modelled LWP \n(g m$^{-2}$)', size=24, color='dimgrey', rotation=0, labelpad=80)
        lab = ax.text(0.1, 0.85, transform=ax.transAxes, s='a', fontsize=32, fontweight='bold', color='dimgrey')
        ax.spines['right'].set_visible(False)
    elif phase == 'ice':
        # SW vs IWP
        ax.set_xlim(290,600)
        slope, intercept, r2, p, sterr = scipy.stats.linregress(AWS14_SW_srs, IWP14_srs)
        if p <= 0.01:
            ax.text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                              s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight='bold',
                              transform=ax.transAxes, size=24, color='dimgrey')
        else:
            ax.text(0.75, 0.9, horizontalalignment='right', verticalalignment='top', s='r$^{2}$ = %s' % np.round(r2, decimals=2),
                              transform=ax.transAxes, size=24,color='dimgrey')
        ax.scatter(AWS14_SW_srs, IWP14_srs, color='#f68080', s=50)
        #ax.set_ylim(np.min(np.mean(runMP['IWP'][:,:, 133:207, 188:213].data, axis=0)),
        #                  np.max(np.mean(runMP['IWP'][:,:, 133:207, 188:213].data, axis=(0))))
        #ax.set_xlim(np.min(np.mean(runSEB['SW_down'][:,:, 133:207, 188:213].data, axis=(0))),
        #                  np.max(np.mean(runSEB['SW_down'][:,:, 133:207, 188:213].data, axis=(0))))
        ax.set_xlabel('Modelled SW$_{\downarrow}$ (W m$^{-2}$)', size=24, color='dimgrey', rotation=0,labelpad=10)
        ax.set_ylabel('Modelled IWP \n(g m$^{-2}$)', size=24, color='dimgrey', rotation=0, labelpad=80)
        lab = ax.text(0.1, 0.85, transform=ax.transAxes, s='b', fontsize=32, fontweight='bold', color='dimgrey')
        ax.yaxis.tick_right()
        [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
        ax.yaxis.set_label_coords(1.3, 0.5)
        ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
    #ax.axis('square')
      # axs.set_adjustable('box')
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey',pad=10)
    [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    #[l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    plt.subplots_adjust(top=0.98, hspace=0.15, bottom=0.1, wspace=0.15, left=0.3, right=0.75)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/Jan_SEB_v_mp'+phase+'.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/Jan_SEB_v_mp'+phase+'.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/Jan_SEB_v_mp'+phase+'.pdf', transparent=True)
    plt.show()



correl_SEB_sgl(Jan_SEB, Jan_mp, phase = 'ice')


#IWP_time_srs(),
#QCF_plot(), QCL_plot()

#T_plot()

