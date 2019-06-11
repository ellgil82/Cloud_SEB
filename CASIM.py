## ------------------------------------------------ CREATE MEAN VERTICAL PROFILES OF ALL MODEL RUNS VS. OBSERVATIONS ------------------------------------------------------ ##
# File management: make sure all model runs are in one containing folder. Presently, this is /data/mac/ellgil82/cloud_data/um/

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
sys.path.append('/users/ellgil82/scripts/Tools/')
from tools import compose_date, compose_time, find_gridbox
from find_gridbox import find_gridbox
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/')

# Load model data
def load_model(config, flight_date, times): #times should be a range in the format 11,21
    pa = []
    pb = []
    pf = []
    print('\nimporting data from %(config)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/'):
            if fnmatch.fnmatch(file, flight_date + '*%(config)s_pb*' % locals()):
                pb.append(file)
            elif fnmatch.fnmatch(file, flight_date + '*%(config)s_pa*' % locals()):
                pa.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/')
    ice_mass_frac = iris.load_cube(pb, 'mass_fraction_of_cloud_ice_in_air')
    liq_mass_frac = iris.load_cube(pb, 'mass_fraction_of_cloud_liquid_water_in_air')
    cl_A_frac = iris.load_cube(pb, 'cloud_area_fraction_assuming_maximum_random_overlap')
    ice_cl_frac = iris.load_cube(pb, 'ice_cloud_volume_fraction_in_atmosphere_layer')
    liq_cl_frac = iris.load_cube(pb, 'liquid_cloud_volume_fraction_in_atmosphere_layer')
    cl_vol = iris.load_cube(pb, 'cloud_volume_fraction_in_atmosphere_layer')
    c = iris.load(pb)# IWP and LWP dont load properly
    IWP = c[1] # stash code s02i392
    LWP = c[0] # stash code s02i391
    #qc = c[3]
    lsm = iris.load_cube(pa, 'land_binary_mask')
    orog = iris.load_cube(pa, 'surface_altitude')
    for i in [ice_mass_frac, liq_mass_frac]:#, ice_cl_frac, liq_cl_frac]:#, qc]:
        real_lon, real_lat = rotate_data(i, 2, 3)
    for j in [LWP, IWP]:#, cl_A_frac]: #cl_A
        real_lon, real_lat = rotate_data(j, 1, 2)
    for k in [lsm, orog]:
        real_lon, real_lat = rotate_data(k, 0, 1)
    # Convert model data to g kg-1
    ice_mass_frac = ice_mass_frac * 1000
    liq_mass_frac = liq_mass_frac * 1000
    IWP = IWP * 1000 # convert to g m-2
    LWP = LWP * 1000
    #qc = qc * 1000
    # Convert times to useful ones
    for i in [IWP, LWP, ice_mass_frac, liq_mass_frac, ice_cl_frac, liq_cl_frac, cl_A_frac, cl_vol]: #qc
        i.coord('time').convert_units('hours since 2011-01-18 00:00')
    altitude = ice_mass_frac.coord('level_height').points[:40] / 1000
    ## ---------------------------------------- CREATE MODEL VERTICAL PROFILES ------------------------------------------ ##
    # Create mean vertical profiles for region of interest
    # region of interest = ice shelf. Longitudes of ice shelf along transect =
    # OR: region of interest = only where aircraft was sampling layer cloud: time 53500 to 62000 = 14:50 to 17:00
    # Define box: -62 to -61 W, -66.9 to -68 S
    # Coord: lon = 188:213, lat = 133:207, time = 4:6 (mean of preceding hours)
    print('\ncreating vertical profiles geez...')
    box_QCF = ice_mass_frac[times[0]:times[1], :40, 133:207, 188:213].data
    box_QCL = liq_mass_frac[times[0]:times[1], :40, 133:207, 188:213].data
    box_QCL[box_QCL <= 0.0000015] = np.nan # c.f. widely used threshold (e.g Cohen & Craig, 2006): > 0.005 g kg-1 for mid-lat clouds
    box_QCF[box_QCF <= 0.0001 ] = np.nan # c.f. Gettelman et al. (2010) threshold: > 0.00005 g kg-1 for mid-lat clouds
    box_mean_QCF = np.nan_to_num(np.nanmean(box_QCF, axis = (0,2,3)))
    box_mean_QCL = np.nan_to_num(np.nanmean(box_QCL, axis=(0, 2, 3)))
    AWS14_QCF = ice_mass_frac[times[0]:times[1], :40, 199:201, 199:201].data
    AWS14_QCL = liq_mass_frac[times[0]:times[1], :40, 199:201, 199:201].data
    AWS14_QCF[AWS14_QCF <= 0.0001 ] = np.nan
    AWS14_QCL[AWS14_QCL <= 0.0000015] = np.nan
    AWS15_QCF = ice_mass_frac[times[0]:times[1], :40,  161:163, 182:184].data
    AWS15_QCL = liq_mass_frac[times[0]:times[1], :40,  161:163, 182:184].data
    AWS15_QCL[AWS15_QCL <= 0.0000015] = np.nan
    AWS15_QCF[AWS15_QCF <= 0.0001 ] = np.nan
    box_mean_IWP = np.mean(IWP[times[0]:times[1], 133:207, 188:213].data)#, axis = (0,1,2))
    box_mean_LWP = np.mean(LWP[times[0]:times[1], 133:207, 188:213].data)#, axis =(0,1,2))
    AWS14_mean_QCF = np.nan_to_num(np.nanmean(AWS14_QCF, axis=(0, 2, 3)))
    AWS14_mean_QCL = np.nan_to_num(np.nanmean(AWS14_QCL, axis=(0, 2, 3)))
    AWS15_mean_QCF = np.nan_to_num(np.nanmean(AWS15_QCF, axis=(0, 2, 3)))
    AWS15_mean_QCL = np.nan_to_num(np.nanmean(AWS15_QCL, axis=(0, 2, 3)))
    # Find max and min values at each model level
    time_mean_QCF = np.nanmean(box_QCF, axis=0)
    array = pd.DataFrame()
    for each_lat in np.arange(74):
        for each_lon in np.arange(25):
            for each_time in np.arange(times[1]-times[0]):
                m = pd.DataFrame(box_QCF[each_time, :, each_lat, each_lon])
                array = pd.concat([m, array], axis=1)
    array[array == np.nan] = 0
    max_QCF = array.max(axis=1)
    min_QCF = array.min(axis=1)
    # Calculate 95th percentile
    ice_95 = np.percentile(array, 95, axis=1)
    ice_5 = np.percentile(array, 5, axis=1)
    # Find max and min values at each model level
    time_mean_QCL = np.mean(box_QCL, axis=0)
    array = pd.DataFrame()
    for each_lat in np.arange(74):
        for each_lon in np.arange(25):
            for each_time in np.arange(times[1]-times[0]):
                m = pd.DataFrame(box_QCL[each_time, :, each_lat, each_lon])
                array = pd.concat([m, array], axis=1)
    array[array == np.nan] = 0
    max_QCL = array.max(axis=1)
    min_QCL = array.min(axis=1)
    # Calculate 95th percentile
    liq_95 = np.percentile(array, 95, axis=1)
    liq_5 = np.percentile(array, 5, axis=1)
    # Calculate PDF of ice and liquid water contents
    #liq_PDF = mean_liq.plot.density(color = 'k', linewidth = 1.5)
    #ice_PDF = mean_ice.plot.density(linestyle = '--', linewidth=1.5, color='k')
    altitude = ice_mass_frac.coord('level_height').points[:40]/1000
    var_dict = {'real_lon': real_lon, 'real_lat':real_lat,   'lsm': lsm, 'orog': orog,  'IWP': IWP, 'LWP':LWP,'mean_QCF': box_mean_QCF, 'mean_QCL': box_mean_QCL, 'altitude': altitude,
                 'AWS14_mean_QCF': AWS14_mean_QCF, 'AWS14_mean_QCL': AWS14_mean_QCL,'ice_5': ice_5, 'ice_95': ice_95, 'liq_5': liq_5, 'liq_95': liq_95, 'min_QCF': min_QCF,
                'max_QCF': max_QCF, 'min_QCL': min_QCL,'AWS15_mean_QCF': AWS15_mean_QCF, 'AWS15_mean_QCL': AWS15_mean_QCL, 'box_QCF': box_QCF, 'box_QCL': box_QCL, 'cl_A': cl_A_frac,
                'ice_cl': ice_cl_frac, 'liq_cl': liq_cl_frac, 'cl_vol': cl_vol} #cl_A': cl_A,'qc': qc,'box_mean_IWP': box_mean_IWP,'box_mean_LWP': box_mean_LWP,
    return  var_dict

def load_SEB(config, flight_date):
    pa = []
    pf = []
    print('\nimporting data from %(config)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/'):
        if fnmatch.fnmatch(file, flight_date + '*%(config)s_pf*' % locals()):
            pf.append(file)
        elif fnmatch.fnmatch(file, flight_date + '*%(config)s_pa*' % locals()):
            pa.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/')
    lsm = iris.load_cube(pa, 'land_binary_mask')
    orog = iris.load_cube(pa, 'surface_altitude')
    LW_net = iris.load_cube(pf, 'surface_net_downward_longwave_flux')
    SH =  iris.load_cube(pf, 'surface_upward_sensible_heat_flux')
    LH = iris.load_cube(pf, 'surface_upward_latent_heat_flux')
    LW_down = iris.load_cube(pf, 'surface_downwelling_longwave_flux')
    LW_up = iris.load_cube(pf, 'upwelling_longwave_flux_in_air')
    SW_up = iris.load_cube(pf, 'upwelling_shortwave_flux_in_air')
    if config == 'CASIM_24' or config == 'DeMott_24' or config == 'CASIM_f152_ice_off':
        c = iris.load(pf)
        SW_net = c[0]
    else:
        SW_net = iris.load_cube(pf, 'surface_net_downward_shortwave_flux')
        Ts = iris.load_cube(pa, 'surface_temperature')
        Ts = Ts - 273.125
    SW_down = iris.load_cube(pf, 'surface_downwelling_shortwave_flux_in_air')
    for i in [SW_up, LW_up,]:
        real_lon, real_lat = rotate_data(i, 2, 3)
    for j in [SW_down, LW_down, LH, SH, LW_net]:#,SW_net_surf,  Ts
        real_lon, real_lat = rotate_data(j, 1, 2)
    for k in [lsm, orog]:
        real_lon, real_lat = rotate_data(k, 0, 1)
    # Convert times to useful ones
    for i in [SW_down, SW_up, LW_net, LW_down, LW_up, LH, SH]:#, Ts, SW_net_surf,
        i.coord('time').convert_units('hours since 2011-01-18 00:00')
    LH = 0 - LH.data
    SH = 0 - SH.data
    if config =='CASIM_24' or config == 'DeMott' or config == 'CASIM_f152_ice_off':
        var_dict = {'real_lon': real_lon, 'real_lat': real_lat, 'SW_up': SW_up, 'SW_down': SW_down,
                    'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down, 'LW_net': LW_net, 'SW_net': SW_net}
    else:
        Etot = SW_net.data + LW_net.data + LH + SH
        Emelt = np.copy(Etot[::4])
        Emelt[Ts.data < 0] = 0
        var_dict = {'real_lon': real_lon, 'real_lat':real_lat,  'SW_up': SW_up, 'SW_down': SW_down, 'melt': Emelt, 'Etot': Etot, 'Model_time': SW_down.coord('time'),
                    'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down, 'LW_net': LW_net, 'SW_net': SW_net, 'Ts': Ts}
    return var_dict

def load_met(config, flight_date, times):
    pa = []
    print('\nimporting data from %(config)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/'):
        if fnmatch.fnmatch(file, flight_date + '*%(config)s_pa*' % locals()):
            pa.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/')
    lsm = iris.load_cube(pa, 'land_binary_mask')
    orog = iris.load_cube(pa, 'surface_altitude')
    Tair = iris.load_cube(pa, 'air_temperature')
    Tair = Tair - 273.125
    Ts = iris.load_cube(pa, 'surface_temperature')
    Ts = Ts - 273.125
    q = iris.load_cube(pa, 'specific_humidity')
    u = iris.load_cube(pa, 'x_wind')
    v = iris.load_cube(pa, 'y_wind')
    v = v[:,:,1:, :]
    BL_ht = iris.load_cube(pa, 'atmosphere_boundary_layer_thickness')
    q.convert_units('g kg-1')
    # Convert times to useful ones
    for i in [Tair, u, v, q]:#, Ts, SW_net_surf,
        i.coord('time').convert_units('hours since 2011-01-18 00:00')
    for i in [Ts, BL_ht]:#, Ts, SW_net_surf,
        i.coord('time').convert_units('hours since 2011-01-18 00:00')
    ff_AWS14 = np.sqrt((np.mean(u[times[0]:times[1],1,199:201, 199:201].data, axis = 0) ** 2) + (np.mean(v[times[0]:times[1],1,199:201, 199:201].data, axis = 0)** 2))
    var_dict = {'Tair': Tair[times[0]:times[1],0,:,:].data, 'Ts': Ts[times[0]:times[1],:,:].data, 'q': q[times[0]:times[1],0,:,:].data, 'ff': ff_AWS14, 'BL_ht': BL_ht[times[0]:times[1],:,:].data, 'Model_time': q.coord('time').points}
    return var_dict


def load_obs(flight, flight_date):
    ###Inputs: flight number as a string, e.g. 'flight159' and flight_date as a string, with date only (not time) in YYYMMDD format, e.g. '20110125'
    ## ----------------------------------------------- SET UP VARIABLES --------------------------------------------------##
    ## Load core data
    print('\nYes yes cuzzy, pretty soon you\'re gonna have some nice core data...')
    bsl_path_core = '/data/mac/ellgil82/cloud_data/Constantino_Oasis_Peninsula/'+flight+'/core_masin_'+flight_date+'_r001_'+flight+'_1hz.nc'
    cubes = iris.load(bsl_path_core)
    #RH = iris.load_cube(bsl_path_core, 'relative_humidity')
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
    s_file = flight+'_s_v2.npz'
    npz_s=np.load(path+s_file)
    m_file = flight+'_m_v2.npz'
    npz_m = np.load(path + m_file)
    n_file = flight+'_n_v2.npz'
    npz_n = np.load(path + n_file)
    CIP_time = npz_m['time']
    CIP_bound = npz_s['TestPlot_all_y']
    m_all = npz_m['TestPlot_all_y']
    IWC = npz_m['TestPlot_HI_y']+ npz_m['TestPlot_MI_y']
    S_LI = npz_m['TestPlot_LI_y']+npz_m['TestPlot_S_y']
    n_drop_CIP = npz_n['TestPlot_LI_y']+npz_n['TestPlot_S_y']
    # Load CAS data
    CAS_file = '/data/mac/ellgil82/cloud_data/netcdfs/'+flight+'_cas.nc'
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



IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs(flight = 'flight152', flight_date = '20110118')

# Load models in for times of interest: (59, 68) for time of flight, (47, 95) for midday-midnight (discard first 12 hours as spin-up)
#HM_vars = load_model('Hallett_Mossop', (11,21))
#ice_off_vars = load_model(config = 'f152_ice_off', flight_date = '20110118T0000', times = (68,80))
#Aranami_vars = load_model(config = 'CASIM_f152_moisture_consvn', flight_date = '20110118T0000', times = (68,80))
#BL_vars = load_model(config = 'RA1M_mods_BL', flight_date = '20110118T0000', times = (68,80))
#DeMott_BL = load_model(config = 'CASIM_DeMott_BL', flight_date = '20110118T0000', times = (68,80))
RA1M_mod_vars = load_model(config = 'RA1M_mod_24', flight_date = '20110118T0000', times = (60,68))
RA1T_mod_vars = load_model(config ='RA1T_mod_24', flight_date = '20110118T0000', times = (60,68))
RA1T_vars = load_model(config ='RA1T_24', flight_date = '20110118T0000', times = (60,68))
RA1M_vars = load_model(config ='RA1M_24', flight_date = '20110118T0000', times = (60,68))
#CASIM_vars = load_model(config ='CASIM_24', flight_date = '20110118T0000', times = (68,80))
#DeMott_vars = load_model(config ='CASIM_24_DeMott', flight_date = '20110118T0000', times = (68,80))
#ice_off_vars = load_model(config ='CASIM_f152_ice_off', flight_date = '20110118T0000', times = (68,80))
#CASIM_orig_vars = load_model(config ='CASIM_24', flight_date = '20110118T0000', times = (59,68))
#DeMott__origvars = load_model(config ='DeMott', flight_date = '20110118T0000', times = (59,68))
#fl_av_vars = load_model(config ='fl_av', flight_date = '20110118T0000', times = (47,95))
#model_runs = [RA1M_vars, RA1M_mod_vars,RA1T_vars, RA1T_mod_vars, CASIM_vars,  DeMott_vars]#fl_av_vars,

#t24_vars = load_model('24', (59,68))

#RA1M_mod_SEB = load_SEB(config = 'RA1M_mod_24', flight_date = '20110118T0000')
#RA1T_mod_SEB = load_SEB(config ='RA1T_mod_24', flight_date = '20110118T0000')
#RA1T_SEB = load_SEB(config ='RA1T_24', flight_date = '20110118T0000')
#RA1M_SEB = load_SEB(config ='RA1M_24', flight_date = '20110118T0000')
#CASIM_SEB = load_SEB(config ='CASIM_24', flight_date = '20110118T0000')
#DeMott_SEB = load_SEB(config ='CASIM_24_DeMott', flight_date = '20110118T0000',)
#ice_off_SEB = load_SEB(config = 'CASIM_f152_ice_off', flight_date = '20110118T0000')

#model_runs = [RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars, CASIM_vars, DeMott_vars]

def load_AWS(station):
    ## --------------------------------------------- SET UP VARIABLES ------------------------------------------------##
    ## Load data from AWS 14 and AWS 15 for January 2011
    print('\nDayum grrrl, you got a sweet AWS...')
    os.chdir('/data/clivarm/wip/ellgil82/AWS/')
    for file in os.listdir('/data/clivarm/wip/ellgil82/AWS/'):
        if fnmatch.fnmatch(file, '%(station)s_Jan_2011*' % locals()):
            AWS = pd.read_csv(str(file), header = 0)
            print(AWS.shape)
    Jan18 = AWS.loc[(AWS['Day'] == 18)]# & (AWS['Hour'] >= 12)]
    if station == 'AWS14_SEB':
        Jan18['Etot'] = Jan18['Rnet_corr'] + Jan18['Hsen'] + Jan18['Hlat'] - Jan18['Gs']
    #Jan18 = Jan18.append(AWS.loc[(AWS['Day'] == 19) & (AWS['Hour'] == 0)])
    Day_mean = Jan18.mean(axis=0) # Calculates averages for whole day
    Flight = Jan18.loc[(Jan18['Hour'] >=15) &  (Jan18['Hour'] <= 17)]#[(Jan18['Hour'] >= 12)]#
    Flight_mean = Flight.mean(axis=0) # Calculates averages over the time period sampled (15:00 - 17:00)
    return Flight_mean, Day_mean, Jan18

AWS15_flight_mean, AWS15_day_mean, AWS15_Jan = load_AWS('AWS15')
AWS14_SEB_flight_mean, AWS14_SEB_day_mean, AWS14_SEB_Jan  = load_AWS('AWS14_SEB')

os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/')

def calc_SEB(run, times):
    AWS14_SEB_flight = AWS14_SEB_Jan['SWnet_corr'][(times[0]/2):(times[-1]/2)] + AWS14_SEB_Jan['LWnet_corr'][(times[0]/2):(times[-1]/2)] + \
                       AWS14_SEB_Jan['Hsen'][(times[0]/2):(times[-1]/2)] + AWS14_SEB_Jan['Hlat'][(times[0]/2):(times[-1]/2)] - AWS14_SEB_Jan['Gs'][(times[0]/2):(times[-1]/2)]
    AWS14_melt_flight = AWS14_SEB_Jan['melt_energy'][(times[0]/2):(times[-1]/2)]
    AWS14_SEB_day = AWS14_SEB_Jan['SWnet_corr'] + AWS14_SEB_Jan['LWnet_corr'] + AWS14_SEB_Jan['Hsen'] + AWS14_SEB_Jan['Hlat'] - AWS14_SEB_Jan['Gs']
    AWS14_melt_day = AWS14_SEB_Jan['melt_energy']
    Model_SEB_flight_AWS14 = np.mean(run['LW_net'][times[0]:times[1], (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)].data, axis = (1,2)) + \
                         np.mean(run['SW_net'][times[0]:times[1], (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)].data, axis = (1,2)) + \
                         np.mean(run['SH'][times[0]:times[1], (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)], axis = (1,2)) + \
                         np.mean(run['LH'][times[0]:times[1], (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)], axis = (1,2))
    Model_SEB_day_AWS14 = np.mean(run['LW_net'][:, (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)].data, axis = (1,2)) + \
                             np.mean(run['SW_net'][:, (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)].data, axis = (1,2)) + \
                          np.mean(run['SH'][:, (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)], axis = (1,2)) + \
                           np.mean(run['LH'][:, (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)], axis = (1,2))
    Model_SEB_flight_AWS15 = np.mean(run['LW_net'][times[0]:times[1], (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)].data, axis = (1,2)) + \
                             np.mean(run['SW_net'][times[0]:times[1], (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)].data, axis = (1,2)) + \
                             np.mean(run['SH'][times[0]:times[1], (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)], axis = (1,2)) + \
                             np.mean(run['LH'][times[0]:times[1], (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)], axis = (1,2))
    Model_SEB_day_AWS15 = np.mean(run['LW_net'][:, (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)].data, axis = (1,2)) + \
                             np.mean(run['SW_net'][:, (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)].data, axis = (1,2)) + \
                          np.mean(run['SH'][:, (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)], axis = (1,2)) + \
                           np.mean(run['LH'][:, (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)], axis = (1,2))
    Time = run['LW_net'].coord('time')
    Model_time = Time.units.num2date(Time.points)
    melt_masked_flight = Model_SEB_flight_AWS14[::4]
    T_srs = np.mean(run['Ts'][:, (AWS14_lat-1):(AWS14_lat+1),(AWS14_lon-1):(AWS14_lon+1)].data, axis = (1,2))
    melt_masked_flight[T_srs[(times[0] / 4):(times[-1] / 4)] < -0.025] = 0
    melt_masked_day = Model_SEB_day_AWS14[::4]
    melt_masked_day[T_srs < -0.025] = 0
    return Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15, melt_masked_day, melt_masked_flight, AWS14_SEB_flight, AWS14_SEB_day, AWS14_melt_flight, AWS14_melt_day


## ----------------------------------------------- COMPARE MODEL & AWS ---------------------------------------------- ##

real_lat = RA1M_mod_vars['real_lat']
real_lon = RA1M_mod_vars['real_lon']

## Finds closest model gridbox to specified point in real lat, lon coordinates (not indices)
def find_gridloc(x,y):
    lat_loc = np.argmin((real_lat-y)**2) #take whole array and subtract lat you want from each point, then find the smallest difference
    lon_loc = np.argmin((real_lon-x)**2)
    return lon_loc, lat_loc

## Load AWS metadata: data are formatted so that row [0] is the latitude, row [1] is the longitude, and each AWS is in a
## separate column, so it can be indexed in the pandas dataframe
AWS_loc = pd.read_csv('/data/clivarm/wip/ellgil82/AWS/AWS_loc.csv', header = 0)
AWS_list = ['AWS15', 'AWS14']#, 'OFCAP']

AWS14_lon, AWS14_lat = find_gridloc(AWS_loc['AWS14'][1], AWS_loc['AWS14'][0])
AWS14_real_lon = real_lon[AWS14_lon]
AWS14_real_lat = real_lat[AWS14_lat]
AWS15_lon, AWS15_lat = find_gridloc(AWS_loc['AWS15'][1], AWS_loc['AWS15'][0])
AWS15_real_lon = real_lon[AWS15_lon]
AWS15_real_lat = real_lat[AWS15_lat]


def make_table():
    mean_dict = []
    model_runs = ['RA1M_24', 'RA1M_mod_24', 'RA1T_24', 'RA1T_mod_24']#, 'CASIM_24', 'CASIM_24_DeMott' 'CASIM_f152_ice_off']
    col_idx = ['AWS 14', 'RA1M', 'RA1M_mod', 'RA1T', 'RA1T_mod']#, 'Cooper'', DeMott', 'ice off']
    mean_dict = np.append(mean_dict, [AWS14_SEB_flight_mean['Tsobs'], AWS14_SEB_flight_mean['Tair_2m'], AWS14_SEB_flight_mean['qair_2m'], AWS14_SEB_flight_mean['FF_10m'], AWS14_SEB_flight_mean['SWin_corr'], AWS14_SEB_flight_mean['SWout'], AWS14_SEB_flight_mean['SWnet_corr'], AWS14_SEB_flight_mean['LWin'],AWS14_SEB_flight_mean['LWout_corr'],
                                      AWS14_SEB_flight_mean['LWnet_corr'], AWS14_SEB_flight_mean['Hlat'],AWS14_SEB_flight_mean['Hsen'],AWS14_SEB_flight_mean['Etot'], AWS14_SEB_flight_mean['melt_energy']])
    for i in model_runs:
        try:
            SEB_dict = load_SEB(config = i, flight_date = '20110118T0000')
            met_dict = load_met(config = i, flight_date = '20110118T0000', times = (15,18))
            #Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15, melt_masked_day, melt_masked_flight, AWS14_SEB_flight, AWS14_SEB_day, AWS14_melt_flight, AWS14_melt_day = calc_SEB(run = SEB_dict, times = (60:68))
            mean_dict = np.append(mean_dict, [np.mean(met_dict['Ts'][:,199:201, 199:201]), np.mean(met_dict['Tair'][:,199:201, 199:201]), np.mean(met_dict['q'][:,199:201, 199:201]), np.mean(met_dict['ff']), np.mean(SEB_dict['SW_down'][60:68, 199:201, 199:201].data),
                                              np.mean(SEB_dict['SW_up'][60:68, 0, 199:201, 199:201].data),  np.mean(SEB_dict['SW_net'][60:68, 199:201, 199:201].data),
                                          np.mean(SEB_dict['LW_down'][60:68, 199:201, 199:201].data), np.mean(SEB_dict['LW_up'][60:68, 0, 199:201, 199:201].data),  np.mean(SEB_dict['LW_net'][60:68, 199:201, 199:201].data),
                                          np.mean(SEB_dict['LH'][60:68, 199:201, 199:201]), np.mean(SEB_dict['SH'][60:68, 199:201, 199:201]), np.mean(SEB_dict['Etot'][60:68, 199:201, 199:201]), np.mean(SEB_dict['melt'][15:18, 199:201, 199:201])])
        except iris.exceptions.ConstraintMismatchError:
            print(i + ' not available')
    mean_dict = np.resize(mean_dict, new_shape=(len(col_idx), 14))
    df = pd.DataFrame(mean_dict, columns = ['Ts', 'Tair', 'q', 'ff', 'SWdown', 'SWup', 'SWnet', 'LWdown', 'LWup', 'LWnet', 'LH', 'SH', 'Etot', 'Emelt'], index=col_idx)
    df.to_csv('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/f152_model_SEB_unshifted.csv')


#make_table()


def calc_bias():
    model_runs = ['RA1M_24', 'RA1M_mod_24', 'RA1T_24','RA1T_mod_24']  # , 'CASIM_24', 'CASIM_24_DeMott' 'CASIM_f152_ice_off']
    col_idx = [ 'RA1M bias', 'RA1M RMSE', 'RA1M_mod bias', 'RA1M_mod RMSE', 'RA1T bias', 'RA1T RMSE', 'RA1T_mod bias', 'RA1T_mod RMSE']  # , 'Cooper'', DeMott', 'ice off']
    #surf_met_obs = [AWS14_SEB_flight_mean['Tsobs'], AWS14_SEB_flight_mean['Tair_2m'], AWS14_SEB_flight_mean['qair_2m'], AWS14_SEB_flight_mean['FF_10m'], AWS14_SEB_flight_mean['SWin_corr'], AWS14_SEB_flight_mean['SWout'], AWS14_SEB_flight_mean['LWout_corr'],AWS14_SEB_flight_mean['LWnet_corr'], AWS14_SEB_flight_mean['Hlat'], AWS14_SEB_flight_mean['Hsen'], AWS14_SEB_flight_mean['Etot'], AWS14_SEB_flight_mean['melt_energy']]
    AWS_var = AWS14_SEB_Jan
    surf_met_obs = [AWS_var['Tsobs'][34:40:2], AWS_var['Tair_2m'][34:40:2], AWS_var['qair_2m'][34:40:2], AWS_var['FF_10m'][34:40:2], AWS_var['SWin_corr'][34:40:2],
                    AWS_var['LWin'][34:40:2], AWS_var['SWnet_corr'][34:40:2], AWS_var['LWnet_corr'][34:40:2], AWS_var['Hsen'][34:40:2], AWS_var['Hlat'][34:40:2],
                    AWS_var['Etot'][34:40:2], AWS_var['melt_energy'][34:40:2]]  # , AWS_var['melt_energy']]
    #surf_mod = [met_dict['Ts_srs'], met_dict['Tair_srs'], met_dict['q_srs'], met_dict['ff_srs'], SEB_dict['SWdown_srs'],
    #            SEB_dict['LWdown_srs'], SEB_dict['SWnet_srs'], SEB_dict['LWnet_srs'], SEB_dict['HS_srs'],
    #            SEB_dict['HL_srs'], SEB_dict['Etot_srs'], SEB_dict['melt_srs']]  # , SEB_1p5['melt_forced']]
    for i in model_runs:
        try:
            SEB_dict = load_SEB(config=i, flight_date='20110118T0000')
            met_dict = load_met(config=i, flight_date='20110118T0000', times=(17, 20))
            Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15, melt_masked_day, melt_masked_flight, AWS14_SEB_flight, AWS14_SEB_day, AWS14_melt_flight, AWS14_melt_day = calc_SEB(run=SEB_dict, times=(68, 80))
            surf_mod = [np.mean(met_dict['Ts'][:, 199:201, 199:201],axis = (1,2)),
                                              np.mean(met_dict['Tair'][:, 199:201, 199:201],axis = (1,2)),
                                              np.mean(met_dict['q'][:, 199:201, 199:201]), np.mean(met_dict['ff'][:, 199:201, 199:201],axis = (1,2)),
                                              np.mean(SEB_dict['SW_down'][68:80:4, 199:201, 199:201].data,axis = (1,2)),
                                              np.mean(SEB_dict['SW_up'][68:80:4, 0, 199:201, 199:201].data,axis = (1,2)),
                                              np.mean(SEB_dict['SW_net'][68:80:4, 199:201, 199:201].data,axis = (1,2)),
                                              np.mean(SEB_dict['LW_down'][68:80:4, 199:201, 199:201].data,axis = (1,2)),
                                              np.mean(SEB_dict['LW_up'][68:80:4, 0, 199:201, 199:201].data,axis = (1,2)),
                                              np.mean(SEB_dict['LW_net'][68:80:4, 199:201, 199:201].data,axis = (1,2)),
                                              np.mean(SEB_dict['LH'][68:80:4, 199:201, 199:201],axis = (1,2)),
                                              np.mean(SEB_dict['SH'][68:80:4, 199:201, 199:201],axis = (1,2)),
                                              np.mean(Model_SEB_flight_AWS14), np.mean(SEB_dict['melt'][17:20, 199:201, 199:201],axis = (1,2))]
            mean_obs = []
            mean_mod = []
            bias = []
            errors = []
            r2s = []
            rmses = []
            for j in np.arange(len(surf_met_obs)):
                b = surf_mod[j] - surf_met_obs[j]
                errors.append(b)
                mean_obs.append(np.mean(surf_met_obs[j]))
                mean_mod.append(np.mean(surf_mod[j]))
                bias.append(mean_mod[j] - mean_obs[j])
                slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_met_obs[j], surf_mod[j])
                r2s.append(r2)
                mse = mean_squared_error(y_true = surf_met_obs[j], y_pred = surf_mod[j])
                rmse = np.sqrt(mse)
                rmses.append(rmse)
                idx = ['Ts', 'Tair', 'RH', 'wind', 'SWd', 'LWd', 'SWn', 'LWn', 'SH', 'LH', 'total', 'melt']
            df = pd.DataFrame(index=idx)
            df['obs mean'] = pd.Series(mean_obs, index=idx)
            df['mod mean'] = pd.Series(mean_mod, index=idx)
            df['bias'] = pd.Series(bias, index=idx)
            df['rmse'] = pd.Series(rmses, index=idx)
            df['% RMSE'] = (df['rmse'] / df['obs mean']) * 100
            df['correl'] = pd.Series(r2s, index=idx)
            df = pd.DataFrame(stats_table,columns=['Ts', 'Tair', 'q', 'ff', 'SWdown', 'SWup', 'SWnet', 'LWdown', 'LWup', 'LWnet','LH', 'SH', 'Etot', 'Emelt'], index=col_idx)
            df.to_csv('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/f152_bias_RMSE_SEB_'+i+'.csv')
        except iris.exceptions.ConstraintMismatchError:
            print(i + ' not available')


#calc_bias()



## ================================================= PLOTTING ======================================================= ##

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans',
                               'Verdana']

def obs_mod_profile(run):
    fig, ax = plt.subplots(1,2, figsize=(16, 9))
    ax = ax.flatten()
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs(flight = 'flight152', flight_date = '20110118')
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axs.set_ylim(0, max(run['altitude']))
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    #m_QCF = ax[0].plot(run['mean_QCF'], run['altitude'], color = 'k', linestyle = '--', linewidth = 2.5)
    p_QCF = ax[0].plot(IWC_profile, run['altitude'], color = 'k', linewidth = 2.5)
    ax[0].set_xlabel('Cloud ice mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    ax[0].set_ylabel('Altitude \n(km)', rotation = 0, fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 80)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0].set_xlim(0,0.04)
    ax[0].xaxis.get_offset_text().set_fontsize(24)
    ax[0].xaxis.get_offset_text().set_color('dimgrey')
    #ax[0].fill_betweenx(run['altitude'], run['ice_5'], run['ice_95'], facecolor='lightslategrey', alpha=0.5)  # Shaded region between maxima and minima
    #ax[0].plot(run['ice_5'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)
    #ax[0].plot(run['ice_95'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
    ax[0].text(0.1, 0.85, s='a', transform = ax[0].transAxes, fontsize=32, fontweight='bold', color='dimgrey')
    #m_14 = ax[0].plot(run['AWS14_mean_QCF'], run['altitude'], color='darkred', linestyle='--', linewidth=3)
    #m_15= ax[0].plot(run['AWS15_mean_QCF'], run['altitude'], color='darkblue', linestyle='--', linewidth=3)
    ax[1].set_xlabel('Cloud liquid mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    p_QCL = ax[1].plot(LWC_profile, run['altitude'], color = 'k', linewidth = 2.5, label = 'Observations')
    #m_QCL = ax[1].plot(run['mean_QCL'], run['altitude'], color = 'k', linestyle = '--', linewidth = 2.5, label = 'Model: \'cloud\' box mean')
    #ax[1].fill_betweenx(run['altitude'], run['liq_5'], run['liq_95'],  facecolor='lightslategrey', alpha=0.5, label = 'Model: 5$^{th}$ & 95$^{th}$ percentiles\n of \'cloud\' box range')  # Shaded region between maxima and minima
    #ax[1].plot(run['liq_5'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2, label='')
    #ax[1].plot(run['liq_95'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
    #m_14 = ax[1].plot(run['AWS14_mean_QCL'], run['altitude'], color='darkred', linestyle='--', linewidth=3, label='Model: AWS 14')
    #m_15 = ax[1].plot(run['AWS15_mean_QCL'], run['altitude'], color='darkblue', linestyle='--', linewidth=3, label='Model: AWS 15')
    ax[1].axes.tick_params(axis = 'both', which = 'both', direction = 'in', length = 5, width = 1.5,  labelsize = 24, pad = 10)
    ax[1].tick_params(labelleft = 'off')
    ax[1].set_xlim(0,0.4)
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[1].xaxis.get_offset_text().set_fontsize(24)
    ax[1].xaxis.get_offset_text().set_color('dimgrey')
    ax[1].text(0.1, 0.85, s='b', transform = ax[1].transAxes,fontsize=32, fontweight='bold', color='dimgrey')
    plt.subplots_adjust(wspace=0.1, bottom=0.23, top=0.95, left=0.17, right=0.98)
    handles, labels = ax[1].get_legend_handles_labels()
    #handles = [handles[0], handles[1]]#, handles[2], handles[-1]]#,  handles[3] ]
    #labels = [labels[0], labels[1]]#, labels[2], labels[-1]]#, labels[3]]
    lgd = plt.legend(handles, labels, fontsize=20, markerscale=2)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.setp(ax[0].get_xticklabels()[-3], visible=False)
    plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profiles_obs_only.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profiles_obs_only.png', transparent = True)
    plt.show()


#obs_mod_profile(BL_vars)

## Caption: mean modelled water paths (in g kg-1) over the Larsen C ice shelf during the time of flight 152 (16:00Z - 18:00Z)

def column_totals():
    fig, ax = plt.subplots(len(model_runs), 2, sharex='col', figsize=(15, len(model_runs * 5) + 3))
    ax = ax.flatten()
    for axs in ax:
        axs.axis('off')
    plot = 0
    CbAx_ice = fig.add_axes([0.15, 0.9, 0.33, 0.03])
    CbAx_liq = fig.add_axes([0.55, 0.9, 0.33, 0.03]) # 0.94, 0.015 width if multiple
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l'}
    for run in model_runs:
        mesh_ice = ax[plot].pcolormesh(np.mean(run['IWP'][47:, :,:].data, axis = (0)), cmap='Blues_r', vmin=0., vmax=300) # check times!
        ax[plot].contour(run['lsm'].data, colors='#A6ACAF', lw=2)
        ax[plot].contour(run['orog'].data, levels=[10], colors='#A6ACAF', lw=2)
        ax[plot].text(x=30, y=350, s=lab_dict[plot], color='#A6ACAF', fontweight = 'bold',  fontsize=32)
        mesh_liq = ax[plot+1].pcolormesh(np.mean(run['LWP'][47:, :,:].data, axis = (0)), cmap='Blues', vmin=0., vmax=300) # check times!
        ax[plot+1].contour(run['lsm'].data, colors='0.3', lw=2)
        ax[plot+1].contour(run['orog'].data, levels=[10], colors='0.3', lw=2)
        ax[plot+1].text(x=30, y=350, s=lab_dict[plot+1], color='dimgrey', fontweight = 'bold', fontsize=32)
        # Add cloud box to the Axes
        ax[plot].add_patch(matplotlib.patches.Rectangle((188, 133), 25, 74, fill=False, linewidth=3, edgecolor='#A6ACAF', facecolor='none',zorder=8))
        ax[plot+1].add_patch(matplotlib.patches.Rectangle((188,133),25, 74, fill=False, linewidth=3, edgecolor='0.3', facecolor='none', zorder = 8))
        # Add AWS locations
        AWS14 = ax[plot].scatter(200,200, marker='o', s=100, facecolor='#fb9a99', color = '#222222', zorder=10)
        AWS15 = ax[plot].scatter(163,183, marker='o', s=100, facecolor='#fb9a99', color = '#222222', zorder=10)
        AWS14 = ax[plot+1].scatter(200,200, marker='o', s=100, facecolor='#fb9a99', color = '#222222', zorder=10)
        AWS15 = ax[plot+1].scatter(163,183, marker='o', s=100, facecolor='#fb9a99', color = '#222222', zorder=10)
        ax[plot].annotate(xy = (0.45, 0.53), s='AWS 14', fontsize='26', color='#A6ACAF',  xycoords = 'axes fraction' , zorder = 11)
        ax[plot].annotate(xy=(0.3, 0.335), s='AWS 15', fontsize = '26', color = '#A6ACAF',  xycoords = 'axes fraction', zorder = 11)
        ax[plot+1].annotate(xy = (0.45, 0.53), s='AWS 14', fontsize='26', color='#222222',  xycoords = 'axes fraction', zorder = 11 )
        ax[plot+1].annotate(xy=(0.3, 0.335), s='AWS 15', fontsize = '26', color = '#222222',  xycoords = 'axes fraction', zorder = 11)
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
    plt.subplots_adjust(hspace=0.08, wspace=0.08, top=0.85)
    #ax[0].set_title('Total column ice', fontname='Helvetica', color='dimgrey', fontsize=28, )
    #ax[1].set_title('Total column liquid', fontname='Helvetica', color='dimgrey', fontsize=28, )
    #plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/v11_mod_comparison_24.png', transparent=True)
    #plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/v11_mod_comparison_24.eps', transparent=True)
    plt.show()

#column_totals()

def plot_struc():
    fig, ax = plt.subplots(2,2, figsize = (10,10))
    ax = ax.flatten()
    plot = 0
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.2)
    cbAx = fig.add_axes([0.25, 0.9, 0.5, 0.03])
    for i in model_runs:
        c = ax[plot].contourf(np.mean(i['box_QCL'], axis = (0,2)), vmin = 0., vmax = 0.2)
        ax[plot].axis('off')
        c.set_norm(norm)
        plot = plot+1
    cb = plt.colorbar(c, cax = cbAx, orientation = 'horizontal', ticks = [0., 0.1, 0.2])
    cbAx.set_xlabel('Liquid water contents (g kg$^{-1}$)', fontname='Helvetica', color='dimgrey', fontsize=24, labelpad=10)
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=28, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.set_xticks([0, 0.1, 0.2])
    plt.subplots_adjust(top = 0.8, bottom = 0.05, hspace = 0.1, wspace=0.1)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/QCL_transects_sgl_mom.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/QCL_transects_sgl_mom.eps')
    plt.show()

#plot_struc()


def QCF_plot():
    fig, ax = plt.subplots(len(model_runs), 2, sharex='col', figsize=(15, len(model_runs * 5) + 3))
    ax = ax.flatten()
    lab_dict = {0:'a', 1:'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6:'g', 7: 'h'}
    plot = 0
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs()
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.set_xlim(0, 0.04)
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    for run in model_runs:
        ax2 = plt.twiny(ax[plot])
        ax2.set_xlim(0,0.04)
        ax2.axis('off')
        ax2.axes.tick_params(axis='both', which='both', tick1On=False, tick2On=False,  pad=10)
        plt.setp(ax2.get_yticklabels()[0], visible=False)
        plt.setp(ax2.get_xticklabels()[0], visible=False)
        ax2.axes.tick_params(labeltop='off')
        p_QCF = ax2.plot(IWC_profile, run['altitude'], color='k', linewidth=3, label='Observations')
        m_QCF = ax[plot].plot(run['mean_QCF'], run['altitude'], color='k', linestyle = '--', linewidth=3, label='Model: Cloud box')
        m_14 = ax[plot].plot(run['AWS14_mean_QCF'], run['altitude'], color='darkred', linestyle = ':', linewidth=3, label='Model: AWS 14')
        m_15= ax[plot].plot(run['AWS15_mean_QCF'], run['altitude'], color='darkred', linestyle='--', linewidth=3, label='Model: AWS 15')
        ax[plot].fill_betweenx(run['altitude'], run['ice_5'], run['ice_95'], facecolor='lightslategrey', alpha = 0.5)# Shaded region between maxima and minima
        ax[plot].plot(run['ice_5'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)
        ax[plot].plot(run['ice_95'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)# Plot 5th and 95th percentiles
        ax[plot].set_xlim(0, 0.04)
        ax[plot].set_ylim(0, max(run['altitude']))
        plt.setp(ax[plot].get_xticklabels()[0], visible=False)
        ax[plot].axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
        ax[plot].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        lab = ax[plot].text(x=0.004, y=4.8, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        titles = ['    RA1M','RA1M_mod', '     fl_av',  '    RA1T',
                  'RA1T_mod', '   CASIM']
        ax[plot].text(0.3, 0.9, transform=ax[plot].transAxes, s=titles[plot], fontsize=28, color='dimgrey')
        print('\n PLOTTING DIS BIATCH...')
        plot = plot+1
        print('\nDONE!')
        print('\nNEEEEEXT')
    ax[0].axes.tick_params(labelbottom='off')
    ax[1].axes.tick_params(labelbottom='off', labelleft='off')
    ax[2].axes.tick_params(labelbottom='off', labelleft='off')
    ax[4].axes.tick_params(labelleft='off')
    ax[5].axes.tick_params(labelleft='off')
    ax[0].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=20)
    ax[3].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=20)
    ax[3].xaxis.get_offset_text().set_fontsize(24)
    ax[4].xaxis.get_offset_text().set_fontsize(24)
    ax[5].xaxis.get_offset_text().set_fontsize(24)
    ax[3].xaxis.get_offset_text().set_color('dimgrey')
    ax[4].xaxis.get_offset_text().set_color('dimgrey')
    ax[5].xaxis.get_offset_text().set_color('dimgrey')
    ax[3].set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey', fontsize=28,labelpad=35)
    ax[4].set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey', fontsize=28,labelpad=35)
    ax[5].set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey', fontsize=28,labelpad=35)
    ax[3].xaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[3].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[4].xaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[4].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[5].xaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[5].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    for axs in ax:
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    lns = p_QCF + m_QCF + m_14 + m_15 # create labels for legend
    labs = [l.get_label() for l in lns]
    ax[plot-1].legend(lns, labs, markerscale=2, loc=1, fontsize=24)
    lgd = ax[plot-1].legend(lns, labs, markerscale=2, loc=7, fontsize=24)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.12, right=0.95, hspace = 0.12, wspace=0.08)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/v11_QCF_obs_v_all_mod_24.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/v11_QCF_obs_v_all_mod_24.eps')
    #plt.show()

def QCL_plot():
    fig, ax = plt.subplots(len(model_runs), 2, sharex='col', figsize=(15, len(model_runs * 5) + 3))
    ax = ax.flatten()
    lab_dict = {0:'a', 1:'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}
    plot = 0
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs()
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.set_xlim(0,0.4)
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    for run in model_runs:
        ax2 = plt.twiny(ax[plot])
        ax2.set_xlim(0, 0.4)
        ax2.axis('off')
        ax2.axes.tick_params(axis='both', which='both',tick1On=False, tick2On=False,)
        plt.setp(ax2.get_yticklabels()[0], visible=False)
        plt.setp(ax2.get_xticklabels()[0], visible=False)
        ax2.axes.tick_params(labeltop='off')
        p_QCL = ax2.plot(LWC_profile, run['altitude'], color='k', linewidth=3, label='Observations')
        m_QCL = ax[plot].plot(run['mean_QCL'], run['altitude'], color='k', linestyle = '--', linewidth=3, label='Model: Cloud box')
        m_14 = ax[plot].plot(run['AWS14_mean_QCL'], run['altitude'], color='darkred', linestyle = ':', linewidth=3, label='Model: AWS 14')
        m_15= ax[plot].plot(run['AWS15_mean_QCL'], run['altitude'], color='darkred', linestyle='--', linewidth=3, label='Model: AWS 15')
        ax[plot].fill_betweenx(run['altitude'], run['liq_5'], run['liq_95'], facecolor='lightslategrey', alpha = 0.5)  # Shaded region between maxima and minima
        ax[plot].plot(run['liq_5'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)
        ax[plot].plot(run['liq_95'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
        ax[plot].set_xlim(0, 0.4)
        ax[plot].set_ylim(0, max(run['altitude']))
        plt.setp(ax[plot].get_xticklabels()[0], visible=False)
        ax[plot].axes.tick_params(axis='both', which='both', tick1On=False, tick2On=False,)
        ax[plot].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        lab = ax[plot].text(x=0.04, y=4.8, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        titles = ['    RA1M', 'RA1M_mod', '     fl_av', '    RA1T',
                  'RA1T_mod',  '   CASIM']
        ax[plot].text(0.3, 0.9, transform=ax[plot].transAxes, s=titles[plot], fontsize=28, color='dimgrey')
        print('\n PLOTTING DIS BIATCH...')
        plot = plot + 1
        print('\nDONE!')
        print('\nNEEEEEXT')
    ax[0].axes.tick_params(labelbottom='off')
    ax[1].axes.tick_params(labelbottom='off', labelleft='off')
    ax[2].axes.tick_params(labelbottom = 'off', labelleft='off')
    ax[4].axes.tick_params(labelleft='off')
    ax[5].axes.tick_params(labelleft='off')
    ax[0].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', color = 'dimgrey', fontsize=28, labelpad=20)
    ax[3].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', color = 'dimgrey',  fontsize=28, labelpad=20)
    ax[3].xaxis.get_offset_text().set_fontsize(24)
    ax[4].xaxis.get_offset_text().set_fontsize(24)
    ax[5].xaxis.get_offset_text().set_fontsize(24)
    ax[3].xaxis.get_offset_text().set_color('dimgrey')
    ax[4].xaxis.get_offset_text().set_color('dimgrey')
    ax[5].xaxis.get_offset_text().set_color('dimgrey')
    ax[3].set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=35)
    ax[4].set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', color = 'dimgrey', fontsize=28, labelpad=35)
    ax[5].set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=35)
    ax[3].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[3].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[4].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[4].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[5].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[5].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    for axs in ax:
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    lns = p_QCL + m_QCL + m_14 + m_15
    labs = [l.get_label() for l in lns]
    lgd = ax[plot-1].legend(lns, labs, markerscale=2, loc=7, fontsize=24)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.12, right=0.95, hspace=0.12, wspace=0.08)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/v11_QCL_obs_v_all_mod_24.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/v11_QCL_obs_v_all_mod_24.eps')
    #plt.show()

#QCF_plot()
#QCL_plot()

from itertools import chain
import scipy

#model_runs = [RA1M_mod_vars]

def correl_plot():
    fig, ax = plt.subplots(len(model_runs), 2, sharex='col', figsize=(18, len(model_runs * 5) + 3))  # , squeeze=False)
    ax = ax.flatten()
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    plot = 0
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice,\
    box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs()
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
            ax[plot].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight = 'bold', transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        else:
            ax[plot].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        ax[plot].scatter(IWC_profile, run['mean_QCF'], color = '#f68080', s = 50)
        ax[plot].set_xlim(min(chain(IWC_profile, run['mean_QCF'])), max(chain(IWC_profile, run['mean_QCF'])))
        ax[plot].set_ylim(min(chain(IWC_profile, run['mean_QCF'])), max(chain(IWC_profile, run['mean_QCF'])))
        ax[plot].plot(ax[plot].get_xlim(), ax[plot].get_ylim(), ls="--", c = 'k', alpha = 0.8)
        slope, intercept, r2, p, sterr = scipy.stats.linregress(LWC_profile, run['mean_QCL'])
        if p <= 0.01:
            ax[plot+1].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight='bold', transform=ax[plot+1].transAxes,
                          size=24,
                          color='dimgrey')
        else:
            ax[plot+1].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
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
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/correlations_24.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/correlations_24.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/correlations_24.pdf', transparent=True)
    plt.show()

#correl_plot()

from matplotlib.lines import Line2D

def IWP_time_srs():
    model_runs = [RA1M_mod_vars]#[RA1M_vars, RA1M_mod_vars,  RA1T_vars, RA1T_mod_vars, CASIM_vars]#[RA1M_vars, RA1M_mod_vars] fl_av_vars -- CASIM has IWP and LWP in the wrong file stream
    fig, ax = plt.subplots(len(model_runs),2, sharex='col', figsize=(18,len(model_runs*5)+5))#, squeeze=False)
    ax = ax.flatten()
    ax2 = np.empty_like(ax)
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        #[l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        #[l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        axs.axvspan(15, 17, edgecolor = 'dimgrey', facecolor='dimgrey', alpha=0.5)
    def my_fmt(x,p):
        return {0}.format(x) + ':00'
    plot = 0
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    for run in model_runs:
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/')
        print('\nPLOTTING DIS BIATCH...')
        ax[plot].spines['right'].set_visible(False)
        ## 1st column = IWP
        ax[plot].plot(run['IWP'].coord('time').points, (np.mean(run['IWP'][:, 199:201, 199:201].data, axis = (1,2))), label = 'AWS14 IWP', linewidth = 3, linestyle = '--', color = 'darkred')
        ax[plot].plot(run['IWP'].coord('time').points, np.mean(run['IWP'][:,161:163, 182:184].data, axis = (1,2)), label='AWS15 IWP', linewidth=3, linestyle='--',color='darkblue')
        ax[plot].plot(run['IWP'].coord('time').points, np.mean(run['IWP'][:,111:227, 162:213].data, axis = (1,2)), label='Cloud box IWP', linewidth=3, color='k')
        ax[plot].text(0.1, 0.85, transform=ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold',color='dimgrey')
        #plt.setp(ax[plot].get_yticklabels()[-1], visible=False)
        ax[plot].set_xlim(12, 23)
        ax[plot].set_ylim(0,300)
        ax[plot].set_yticks([0, 150, 300])
        ax[plot].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        ## 2nd column = downwelling LW. As above.
        ax[plot+1].set_ylim(0,150)
        ax[plot+1].set_xlim(12, 23)
        ax[plot+1].set_yticks([0, 75,150])
        ax[plot+1].yaxis.set_label_position("right")
        ax[plot + 1].spines['left'].set_visible(False)
        ax[plot+1].yaxis.tick_right()
        ax[plot+1].plot(run['LWP'].coord('time').points, (np.mean(run['LWP'][:, 199:201, 199:201].data, axis = (1,2))), label = 'AWS14 LWP', linewidth = 3, linestyle = '--', color = 'darkred')
        ax[plot+1].plot(run['LWP'].coord('time').points, np.mean(run['LWP'][:,161:163, 182:184].data, axis = (1,2)), label='AWS15 LWP', linewidth=3, linestyle='--',color='darkblue')
        ax[plot+1].plot(run['LWP'].coord('time').points, np.mean(run['LWP'][:,111:227, 162:213].data, axis = (1,2)), label='Cloud box LWP', linewidth=3, color='k')
        ax[plot+1].text(0.1, 0.85, transform = ax[plot+1].transAxes, s=lab_dict[plot+1], fontsize=32, fontweight='bold', color='dimgrey')
        ax[plot+1].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        #[l.set_visible(False) for (i, l) in enumerate(ax[plot+1].yaxis.get_ticklabels()) if i % 2 != 0]
        #[l.set_visible(False) for (i, l) in enumerate(ax[plot + 1].xaxis.get_ticklabels()) if i % 2 != 0]
        [w.set_linewidth(2) for w in ax[plot].spines.itervalues()]
        [w.set_linewidth(2) for w in ax[plot+1].spines.itervalues()]
        #ax[plot+1].set_xlim(run['IWP'].coord('time').points[1], run['IWP'].coord('time').points[-1])
        titles = ['    RA1M', '    RA1M','RA1M_mod', 'RA1M_mod', '    RA1T', '    RA1T', 'RA1T_mod','RA1T_mod', '   CASIM', '   CASIM']#['    RA1M','    RA1M', 'RA1M_mod', 'RA1M_mod', '     fl_av','     fl_av', '    RA1T', '    RA1T', 'RA1T_mod','RA1T_mod', '   CASIM', '   CASIM']
        ax[plot].text(0.83, 1.05, transform=ax[plot].transAxes, s=titles[plot], fontsize=28,color='dimgrey')
        print('\nDONE!')
        print('\nNEEEEEXT')
        plot = plot + 2
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d:00"))
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d:00"))
    lns = [Line2D([0], [0], color='k', linewidth=3),
           Line2D([0], [0], color='darkred', linestyle = '--', linewidth=3),
           Line2D([0], [0], color='darkblue', linestyle = '--', linewidth=3)]
    labs = ['Cloud box mean, modelled','AWS 14, modelled', 'AWS 15, modelled']#  '                      ','                      '
    lgd = plt.legend(lns, labs, ncol=2, bbox_to_anchor=(0.9, -0.2), borderaxespad=0., loc='best', prop={'size': 24})
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left=0.22, bottom=0.17, right=0.78, top=0.95, wspace = 0.15, hspace = 0.15)
    fig.text(0.5, 0.08, 'Time (hours)', fontsize=24, fontweight = 'bold', ha = 'center', va = 'center', color = 'dimgrey')
    fig.text(0.08, 0.55, 'IWP (g kg$^{-1}$)', fontsize=30, ha= 'center', va='center', rotation = 0, color = 'dimgrey')
    fig.text(0.92, 0.55, 'LWP (g kg$^{-1}$)', fontsize=30, ha='center', va='center', color = 'dimgrey', rotation=0)
    #plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/vn11_water_path_time_srs.png')
    #plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/vn11_water_path_time_srs.eps')
    plt.show()


#IWP_time_srs()

colour_dict = {RA1M_vars: '#f87e7e',
               RA1M_mod_vars: '#1f78b4',
               RA1T_vars: '#33a02c',
               RA1T_mod_vars: 'dimgrey',
               DeMott_vars: '#EA580F',
               CASIM_vars: '#5D13E8',
               ice_off_vars: '#DC143C'}

def all_mod_plot(run1, run2, run3, run4):
    fig, ax = plt.subplots(1,2, figsize=(16, 9))
    ax = ax.flatten()
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs(flight = 'flight152', flight_date = '20110118')
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axs.set_ylim(0, max(run1['altitude']))
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    #m1_QCF = ax[0].plot(run1['AWS14_mean_QCF'], run1['altitude'], color= '#f87e7e', linestyle='--', linewidth=2.5, label = 'RA1M')
    #m2_QCF = ax[0].plot(run2['AWS14_mean_QCF'], run2['altitude'], color='#1f78b4', linestyle='--', linewidth=2.5, label = 'RA1M_mod')
    #m3_QCF = ax[0].plot(run3['AWS14_mean_QCF'], run3['altitude'], color='#33a02c', linestyle='--', linewidth=2.5, label= 'RA1T')
    #m4_QCF = ax[0].plot(run4['AWS14_mean_QCF'], run4['altitude'], color='dimgrey', linestyle='--', linewidth=2.5, label='RA1T_mod')
    #p_QCF = ax[0].plot(IWC_profile, run1['altitude'], color = 'k', linewidth = 2.5, label = 'Observations')
    ax[0].set_xlabel('Cloud ice mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    ax[0].set_ylabel('Altitude \n(km)', rotation = 0, fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 80)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0].set_xlim(0,0.04)
    ax[0].xaxis.get_offset_text().set_fontsize(24)
    ax[0].xaxis.get_offset_text().set_color('dimgrey')
    ax[0].text(0.1, 0.85, transform=ax[0].transAxes, s='a', fontsize=32, fontweight='bold', color='dimgrey')
    ax[1].set_xlabel('Cloud liquid mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    #p_QCL = ax[1].plot(LWC_profile, run1['altitude'], color = 'k', linewidth = 2.5, label = 'Observations')
    #m1_QCL = ax[1].plot(run1['AWS14_mean_QCL'], run1['altitude'], color= '#f87e7e', linestyle='--', linewidth=2.5, label = 'RA1M')
    #m2_QCL = ax[1].plot(run2['AWS14_mean_QCL'], run2['altitude'], color='#1f78b4', linestyle='--', linewidth=2.5, label = 'RA1M_mod')
    #m3_QCL = ax[1].plot(run3['AWS14_mean_QCL'], run3['altitude'], color='#33a02c', linestyle='--', linewidth=2.5, label= 'RA1T')
    #m4_QCL = ax[1].plot(run4['AWS14_mean_QCL'], run4['altitude'], color='dimgrey', linestyle='--', linewidth=2.5, label='RA1T_mod')
    ax[1].axes.tick_params(axis = 'both', which = 'both', direction = 'in', length = 5, width = 1.5,  labelsize = 24, pad = 10)
    ax[1].tick_params(labelleft = 'off')
    ax[1].set_xlim(0,0.4)
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[1].xaxis.get_offset_text().set_fontsize(24)
    ax[1].xaxis.get_offset_text().set_color('dimgrey')
    ax[1].text(0.1, 0.85, transform=ax[1].transAxes, s='b', fontsize=32, fontweight='bold', color='dimgrey')
    plt.subplots_adjust(wspace=0.1, bottom=0.23, top=0.95, left=0.17, right=0.98)
    #handles, labels = ax[1].get_legend_handles_labels()
    #handles = [handles[0], handles[1], handles[-1], handles[2],  handles[3] ]
    #labels = [labels[0], labels[1], labels[-1], labels[2], labels[3]]
    #lgd = plt.legend(handles, labels, fontsize=20, markerscale=2)
    lgd = plt.legend(fontsize = 20, markerscale = 2)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.setp(ax[0].get_xticklabels()[-3], visible=False)
    #plt.setp(ax[0].get_xticklabels()[-5], visible=False)
    plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/AWS14_vertical_profiles_mean_comparison_sgl_mom_axes_only.eps')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/AWS14_vertical_profiles_mean_comparison_sgl_mom_axes_only.png')
    plt.show()


#all_mod_plot(RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars)

def rad_time_srs():
    model_runs = [RA1M_mod_SEB]#[RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars, Cooper_vars, DeMott_vars, ice_off_vars, ]#
    fig, ax = plt.subplots(len(model_runs),2, sharex='col', figsize=(18,8))#(16,len(model_runs*5)+3), squeeze=False)#
    ax = ax.flatten()
    ax2 = np.empty_like(ax)
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        #[l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        #[l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        #axs.axvline(x = 14.75, color = '#222222', alpha = 0.5, linestyle = ':', linewidth = 3)
        #axs.axvline(x=17, color='#222222', alpha=0.5, linestyle=':', linewidth=3)
        #axs.axvspan(14.75,17, edgecolor = 'dimgrey', facecolor = 'dimgrey', alpha = 0.2,)
        axs.axvspan(15, 17, edgecolor = 'dimgrey', facecolor='dimgrey', alpha=0.5) #shifted  = 17,20 / normal = 14.75, 17
        #axs.arrow(x=0.4, y=0.95, dx = .4, dy = 0., linewidth = 3, color='k', zorder = 10)
    def my_fmt(x,p):
        return {0}.format(x) + ':00'
    plot = 0
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o'}
    for run in model_runs:
        AWS14_flight_mean, AWS14_day_mean, AWS14_Jan = load_AWS('AWS14')
        #AWS15_flight_mean, AWS15_day_mean, AWS15_Jan = load_AWS('AWS15')
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/')
        print('\nPLOTTING DIS BIATCH...')
        ## 1st column = downwelling SW. Dashed lines = model, solid = obs. Red = 14, Blue = 15.
        ax[plot].plot(run['LW_down'].coord('time').points, (np.mean(run['SW_down'][:, 199:201, 199:201].data, axis = (1,2))), label = 'AWS14, modelled', linewidth = 3, linestyle = '--', color = 'darkred')
        ax2[plot] = plt.twiny(ax[plot])
        ax[plot].spines['right'].set_visible(False)
        ax2[plot].axis('off')
        ax2[plot].plot(AWS14_Jan['Hour'], AWS14_Jan['Sin'], label='AWS14, observed', linewidth=3, color='darkred')
        #ax[plot].plot(run['LW_down'].coord('time').points, np.mean(run['SW_down'][:,161:163, 182:184].data, axis = (1,2)), label='AWS15, modelled', linewidth=3, linestyle='--',color='darkblue')
        #ax2[plot].plot(AWS15_Jan['Hour'], AWS15_Jan['Sin'], label='AWS15, observed', linewidth=3, color='darkblue')
        ax[plot].text(0.1, 0.85, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        #plt.setp(ax[plot].get_yticklabels()[-1], visible=False)
        ax[plot].set_xlim(12, 23)
        ax[plot].set_ylim(400,800)
        ax2[plot].set_ylim(400,800)
        ax[plot].set_yticks([400,600, 800])
        ax2[plot].set_xlim(AWS15_Jan['Hour'].values[12], AWS15_Jan['Hour'].values[-1]) ##
        ax[plot].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        ## 2nd column = downwelling LW. As above.
        ax2[plot+1] = plt.twiny(ax[plot+1])
        ax[plot+1].set_xlim(12, 23)
        ax2[plot+1].set_xlim(AWS15_Jan['Hour'].values[12], AWS15_Jan['Hour'].values[-1])
        ax[plot+1].set_ylim(230,310)
        ax2[plot+1].set_ylim(230,310)
        ax[plot+1].set_yticks([250,300])
        ax2[plot + 1].axis('off')
        ax[plot+1].yaxis.set_label_position("right")
        ax[plot + 1].spines['left'].set_visible(False)
        ax[plot+1].yaxis.tick_right()
        mod14 = ax[plot+1].plot(run['LW_down'].coord('time').points, np.mean(run['LW_down'][:,199:201, 199:201].data, axis = (1,2)),  label = 'AWS14, modelled', linewidth = 3, linestyle = '--', color = 'darkred') ##
        #mod15 = ax[plot+1].plot(run['LW_down'].coord('time').points,np.mean(run['LW_down'][:,161:163, 182:184].data, axis = (1,2)), label = 'AWS15, modelled', linewidth = 3, linestyle = '--', color = 'darkblue') ##
        obs14 = ax2[plot+1].plot(AWS14_Jan['Hour'], AWS14_Jan['Lin'], label='AWS14, observed', linewidth=3, color='darkred') ##
        #obs15 = ax2[plot+1].plot(AWS15_Jan['Hour'], AWS15_Jan['Lin'], label='AWS15, observed', linewidth=3, color='darkblue') ##
        ax[plot+1].text(0.1, 0.85, transform = ax[plot+1].transAxes, s=lab_dict[plot+1], fontsize=32, fontweight='bold', color='dimgrey')
        ax[plot+1].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [w.set_linewidth(2) for w in ax[plot].spines.itervalues()]
        [w.set_linewidth(2) for w in ax[plot+1].spines.itervalues()]
        #ax[plot+1].set_xlim(run['LW_down'].coord('time').points[1], run['LW_down'].coord('time').points[-1])
        #ax2[plot+1].set_xlim(AWS15_Jan['Hour'].values[0], AWS15_Jan['Hour'].values[-1]) ##
        plt.setp(ax2[plot].get_xticklabels(), visible=False)
        plt.setp(ax2[plot+1].get_xticklabels(), visible=False)
        titles = ['RA1M_mod', 'RA1M_mod']#, '    RA1M','    RA1M', '     fl_av','     fl_av', '    RA1T', '    RA1T', 'RA1T_mod','RA1T_mod', '   CASIM', '   CASIM']
        #ax[plot].text(0.83, 1.05, transform=ax[plot].transAxes, s=titles[plot], fontsize=28,
        #              color='dimgrey')
        print('\nDONE!')
        print('\nNEEEEEXT')
        ax[plot].arrow(x=0.4, y=0.95, dx=.4, dy=0., linewidth=3, color='k', zorder=10)
        plot = plot + 2
        # ax[plot+1].set_xlim(12,23)
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d:00"))
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d:00"))
    lns = [Line2D([0], [0], color='darkred', linewidth=3),
           Line2D([0], [0], color='darkred', linestyle = '--', linewidth=3)]
           #Line2D([0], [0], color='darkblue', linewidth=3),
           #Line2D([0], [0], color='darkblue', linestyle = '--', linewidth=3)]
    labs = ['AWS 14, observed', 'AWS 14, modelled']#,'AWS 15, observed', 'AWS 15, modelled']#  '                      ','                      '
    lgd = plt.legend(lns, labs, ncol=2, bbox_to_anchor=(0.9, -0.2), borderaxespad=0., loc='best', prop={'size': 24})
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    fig.text(0.5, 0.04, 'Time (hours)', fontsize=24, fontweight = 'bold', ha = 'center', va = 'center', color = 'dimgrey')
    fig.text(0.08, 0.55, 'Downwelling \nshortwave \nflux \n(W m$^{-2}$)', fontsize=30, ha= 'center', va='center', rotation = 0, color = 'dimgrey')
    fig.text(0.92, 0.55, 'Downwelling \nlongwave \nflux \n(W m$^{-2}$)', fontsize=30, ha='center', va='center', color = 'dimgrey', rotation=0)
    plt.subplots_adjust(left=0.22, bottom=0.35, right=0.78, top=0.97, wspace=0.15, hspace=0.15)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/Downwelling_fluxes_RA1M_mod_unshifted_AWS14.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/Downwelling_fluxes_RA1M_mod_unshifted_AWS14.eps', transparent = True)
    plt.show()

#rad_time_srs()

def total_SEB(run):
    fig, axs = plt.subplots(2, 1, figsize=(22, 22), frameon=False)
    hrs = mdates.HourLocator(interval=2)
    hrfmt = mdates.DateFormatter('%H:%M')
    for ax in axs:
        plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_major_formatter(hrfmt)
        ax.xaxis.set_major_locator(hrs)
        ax.tick_params(axis='both', which='both', labelsize=44, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        ax.axhline(y=0, xmin=0, xmax=1, linestyle='--', linewidth=1)
        ax.set_ylabel('Energy flux \n (W m$^{-2}$)', rotation=0, fontsize=44, labelpad=70, color='dimgrey')
        [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
        ax.set_ylim(-100, 120)
        ax.set_yticks([-100,0,100])
    # Plot observed SEB
    axs[0].plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['SWnet_corr'][24:], color='#6fb0d2', lw=5, label='Net shortwave flux')
    axs[0].plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['LWnet_corr'][24:], color='#86ad63', lw=5, label='Net longwave flux')
    axs[0].plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['Hsen'][24:], color='#1f78b4', lw=5, label='Sensible heat flux')
    axs[0].plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['Hlat'][24:], color='#33a02c', lw=5, label='Latent heat flux')
    axs[0].plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['melt_energy'][24:], color='#f68080', lw=5, label='Melt flux')
    #axs[0].axes.get_xaxis().set_visible(False)
    axs[0].set_xlim(AWS14_SEB_Jan['Time'].values[24], AWS14_SEB_Jan['Time'].values[47])
    axs[0].axvspan(18.625, 18.70833, edgecolor='dimgrey', facecolor='dimgrey', alpha=0.5)
    # Plot model SEB
    Model_time = run['LW_net'].coord('time')
    Model_time.convert_units('seconds since 1969-12-31 23:59:59')
    Model_time = mdates.epoch2num(Model_time.points)
    LH = run['LH']
    SH = run['SH']
    ax2 = axs[1].twiny()
    ax2.set_xlim(0,12)
    ax2.axis('off')
    axs[1].plot(Model_time[48:], np.mean(run['SW_net'][48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1, 2)), color='#6fb0d2', lw=5, label='Net shortwave flux')
    axs[1].plot(Model_time[48:], np.mean(run['LW_net'][48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1, 2)), color='#86ad63', lw=5, label='Net longwave flux')
    axs[1].plot(Model_time[48:], np.mean(SH[48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1, 2)), color='#1f78b4', lw=5, label='Sensible heat flux')
    axs[1].plot(Model_time[48:], np.mean(LH[48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1, 2)), color='#33a02c', lw=5, label='Latent heat flux')
    ax2.plot(np.mean(run['melt'][11:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1, 2)), color='#f68080', lw=5, label='Melt flux')
    axs[1].axvspan(Model_time[60], Model_time[68], edgecolor='dimgrey', facecolor='dimgrey', alpha=0.5)
    #axs[1].axvline(Model_time[68], linestyle = ':', lw = 5, color='dimgrey')
    #axs[1].axvline(Model_time[59], linestyle=':', lw=5, color='dimgrey')
    axs[1].set_xlim(Model_time[48], Model_time[-1])
    #axs[1].set_xticks([Model_time[48], Model_time[68], Model_time[80], Model_time[-1]])#Model_time[59],
    lns = [Line2D([0], [0], color='#6fb0d2', linewidth=3),
           Line2D([0], [0], color='#86ad63',  linewidth=3),
           Line2D([0], [0], color='#1f78b4', linewidth=3),
           Line2D([0], [0], color='#33a02c', linewidth=3),
           Line2D([0], [0], color='#f68080', linewidth=3)]
    labs = ['$SW_{net}$','$LW_{net}$', '$H_{S}$', '$H_{L}$', '$E_{melt}$']
    lgd = plt.legend(lns, labs, fontsize=36, bbox_to_anchor = (1.15, 1.3),prop={'size': 44})
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    #axs[0].text(x=AWS14_SEB_Jan['Time'].values[2], y=250, s='a', fontsize=44, fontweight='bold', color='dimgrey')
    #axs[1].text(x=Model_time[1], y=250, s='b', fontsize=32, fontweight='bold', color='dimgrey')
    plt.subplots_adjust(left=0.22, top = 0.95, bottom=0.1, right=0.9, hspace = 0.1, wspace = 0.1)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_SEB_RA1M_mod_unshifted.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_SEB_RA1M_mod_unshifted.png', transparent=True)
    plt.show()

#total_SEB(RA1M_mod_SEB)



def dif_plot():
    fig, ax = plt.subplots(1,2, figsize=(16, 9))
    ax = ax.flatten()
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs()
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axs.set_ylim(0, max(RA1M_vars['altitude']))
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    dif_QCF = ax[0].plot((RA1M_mod_vars['mean_QCF']-RA1M_vars['mean_QCF']), RA1M_vars['altitude'], color = 'k', linestyle = '--', linewidth = 2.5)
    ax[0].set_xlabel('Cloud ice mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    ax[0].set_ylabel('Altitude \n(km)', rotation = 0, fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 80)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0].set_xlim(0,0.002)
    ax[0].xaxis.get_offset_text().set_fontsize(24)
    ax[0].xaxis.get_offset_text().set_color('dimgrey')
    ax[0].text(x=0.00015, y=4.8, s='a', fontsize=32, fontweight='bold', color='dimgrey')
    ax[1].set_xlabel('Cloud liquid mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    dif_QCL = ax[1].plot((RA1M_mod_vars['mean_QCL']-RA1M_vars['mean_QCL']), RA1M_vars['altitude'], color = 'k', linestyle = '--', linewidth = 2.5, label = 'RA1M_mod \n - RA1M')
    ax[1].axes.tick_params(axis = 'both', which = 'both', direction = 'in', length = 5, width = 1.5,  labelsize = 24, pad = 10)
    ax[1].tick_params(labelleft = 'off')
    ax[1].set_xlim(0,0.04)
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[1].xaxis.get_offset_text().set_fontsize(24)
    ax[1].xaxis.get_offset_text().set_color('dimgrey')
    ax[1].text(x=0.003, y=4.8, s='b', fontsize=32, fontweight='bold', color='dimgrey')
    plt.subplots_adjust(wspace=0.1, bottom=0.23, top=0.95, left=0.17, right=0.98)
    handles, labels = ax[1].get_legend_handles_labels()
    handles = [handles[0]]#, handles[1], handles[-1], handles[2],  handles[3] ]
    labels = [labels[0]]#, labels[1], labels[-1], labels[2], labels[3]]
    lgd = plt.legend(handles, labels, fontsize=20, markerscale=2)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.setp(ax[0].get_xticklabels()[-3], visible=False)
    plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profiles_dif_RA1M_v_RA1M_mod.eps')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profiles_dif_RA1M_v_RA1M_mod.png')
    plt.show()

#dif_plot()

#RA1M_SEB = load_SEB(config = 'RA1M_24', flight_date= '20110118T0000')
#RA1M_mod_SEB = load_SEB(config = 'RA1M_mod_24', flight_date= '20110118T0000Z')

#model_runs = ['RA1M_SEB']

def SEB_correl(runSEB, runMP, times, scatter_type):
    fig, ax = plt.subplots(len(model_runs),2, sharex='col', figsize=(18, len(model_runs * 5) + 3), subplot_kw = {'aspect':1})  # , squeeze=False)
    ax = ax.flatten()
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    plot = 0
    if scatter_type == 'spatial' or 'space':
        # LW vs LWP
        ax[plot].set_xlim(250,350)
        ax[plot].scatter(np.ravel(np.mean(runSEB['melt'][times[0]:times[1], 133:207, 188:213].data, axis=(0))), np.ravel(np.mean(runMP['LWP'][times[0]:times[1], 133:207, 188:213].data, axis=(0))), color='#f68080',s=50)
        #ax[plot].set_ylim(np.min(np.mean(runMP['LWP'][times[0]:times[1], 133:207, 188:213].data, axis=0)),
        #                  np.max(np.mean(runMP['LWP'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
        #ax[plot].set_xlim(np.min(np.mean(runSEB['LW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))),
        #                  np.max(np.mean(runSEB['LW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
        slope, intercept, r2, p, sterr = scipy.stats.linregress(np.ravel(np.mean(runSEB['LW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))),
            np.ravel(np.mean(runMP['LWP'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
        if p <= 0.01:
            ax[plot].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top', s='r$^{2}$ = %s' % np.round(r2, decimals=2),
                          fontweight='bold', transform=ax[plot].transAxes, size=24,color='dimgrey')
        else:
            ax[plot].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax[plot].transAxes, size=24, color='dimgrey')
        # SW vs IWP
        slope, intercept, r2, p, sterr = scipy.stats.linregress(np.ravel(np.mean(runSEB['SW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))),
            np.ravel(np.mean(runMP['IWP'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
        if p <= 0.01:
            ax[plot + 1].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                              s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight='bold',
                              transform=ax[plot + 1].transAxes, size=24, color='dimgrey')
        else:
            ax[plot + 1].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top', s='r$^{2}$ = %s' % np.round(r2, decimals=2),
                              transform=ax[plot + 1].transAxes, size=24,color='dimgrey')
        ax[plot + 1].scatter(np.ravel(np.mean(runSEB['SW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))), np.ravel(np.mean(runMP['IWP'][times[0]:times[1], 133:207, 188:213].data, axis=(0))),
                             color='#f68080', s=50)
        #ax[plot+1].set_ylim(np.min(np.mean(runMP['IWP'][times[0]:times[1], 133:207, 188:213].data, axis=0)),
        #                  np.max(np.mean(runMP['IWP'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
        #ax[plot+1].set_xlim(np.min(np.mean(runSEB['SW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))),
        #                  np.max(np.mean(runSEB['SW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
        #ax[plot + 1].plot(ax[plot + 1].get_xlim(), ax[plot + 1].get_ylim(), ls="--", c='k', alpha=0.8)
        ax[plot+1].set_xlim(400,600)
    elif scatter_type == 'temporal' or 'time':
        ax[plot].scatter(np.mean(runSEB['LW_down'][times[0]:times[1],199:201, 199:201].data, axis = (1,2)), np.mean(runMP['LWP'][times[0]:times[1],199:201, 199:201].data, axis = (1,2)), color = '#f68080', s = 50)
        ax[plot].set_ylim(min(np.mean(runMP['LWP'][times[0]:times[1],199:201, 199:201].data, axis = (1,2))), max(np.mean(runMP['LWP'][times[0]:times[1],199:201, 199:201].data, axis = (1,2))))
        ax[plot].set_xlim(min(np.mean(runSEB['LW_down'][times[0]:times[1],199:201, 199:201].data, axis = (1,2))), max(np.mean(runSEB['LW_down'][times[0]:times[1],199:201, 199:201].data, axis = (1,2))))
        slope, intercept, r2, p, sterr = scipy.stats.linregress(
            np.mean(runSEB['LW_down'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2)),
            np.mean(runMP['LWP'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2)))
        if p <= 0.01:
            ax[plot].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight='bold', transform=ax[plot].transAxes,
                          size=24,
                          color='dimgrey')
        else:
            ax[plot].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        slope, intercept, r2, p, sterr = scipy.stats.linregress(
            np.mean(runSEB['SW_down'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2)),
            np.mean(runMP['IWP'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2)))
        if p <= 0.01:
            ax[plot+1].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight='bold', transform=ax[plot+1].transAxes,
                          size=24,
                          color='dimgrey')
        else:
            ax[plot+1].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax[plot+1].transAxes, size=24,
                          color='dimgrey')
        ax[plot+1].scatter(np.mean(runSEB['SW_down'][times[0]:times[1],199:201, 199:201].data, axis = (1,2)), np.mean(runMP['IWP'][times[0]:times[1],199:201, 199:201].data, axis = (1,2)), color='#f68080', s=50)
        #ax[plot+1].set_ylim(min(np.mean(runMP['IWP'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2))), max(np.mean(runMP['IWP'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2))))
        #ax[plot+1].set_xlim(min(np.mean(runSEB['SW_down'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2))),max(np.mean(runSEB['SW_down'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2))))
        #ax[plot+1].plot(ax[plot+1].get_xlim(), ax[plot+1].get_ylim(), ls="--", c='k', alpha=0.8)
    ax[plot].set_xlabel('Modelled LW$_{\downarrow}$ (W m$^{-2}$)', size = 24, color = 'dimgrey', rotation = 0, labelpad = 10)
    ax[plot].set_ylabel('Modelled LWP \n(g m$^{-2}$)', size = 24, color = 'dimgrey', rotation =0, labelpad= 80)
    ax[plot+1].set_xlabel('Modelled SW$_{\downarrow}$ (W m$^{-2}$)', size = 24, color = 'dimgrey', rotation = 0, labelpad = 10)
    ax[plot+1].set_ylabel('Modelled IWP \n(g m$^{-2}$)', size = 24, color = 'dimgrey', rotation =0, labelpad= 80)
    lab = ax[plot].text(0.1, 0.85, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
    lab2 = ax[plot+1].text(0.1, 0.85, transform = ax[plot+1].transAxes, s=lab_dict[plot+1], fontsize=32, fontweight='bold', color='dimgrey')
    titles = ['RA1M_mod','RA1M_mod']#'    RA1M','    RA1M']#,,'     fl_av', '     fl_av','    RA1T','    RA1T',  'RA1T_mod', 'RA1T_mod','   CASIM','   CASIM']
    ax[plot].text(0.83, 1.1, transform=ax[plot].transAxes, s=titles[plot], fontsize=28, color='dimgrey')
    plt.setp(ax[plot].get_xticklabels()[-2], visible=False)
    plt.setp(ax[plot].get_yticklabels()[-2], visible=False)
    ax[plot+1].yaxis.tick_right()
    [l.set_visible(False) for (w, l) in enumerate(ax[plot + 1].yaxis.get_ticklabels()) if w % 2 != 0]
    ax[plot].yaxis.set_label_coords(-0.6, 0.5)
    ax[plot+1].yaxis.set_label_coords(1.6, 0.5)
    ax[plot].spines['right'].set_visible(False)
    ax[plot+1].spines['left'].set_visible(False)
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        #axs.axis('square')
        axs.set_adjustable('box')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    ax[0].set_xlim(250, 350)
    ax[1].set_xlim(400, 600)
    plt.subplots_adjust(top = 0.98, hspace = 0.15, bottom = 0.05, wspace = 0.15, left = 0.15, right = 0.85)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/SEB_v_mp_RA1M_mod_LW_shifted.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/SEB_v_mp_RA1M_mod_LW_shifted.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/SEB_v_mp_RA1M_mod_LW_shifted.pdf', transparent=True)
    plt.show()


#SEB_correl(RA1M_mod_SEB, RA1M_mod_vars, times = (69,79), scatter_type= 'spatial')


def correl_SEB_sgl(runSEB, runMP, times, phase):
    fig, ax = plt.subplots(figsize = (12,6))
    if phase == 'liquid':
        # LW vs LWP
        #ax.set_xlim(260,330)
        #ax.set_ylim(0,300)
        ax.scatter(np.ravel(np.mean(runSEB['melt'][times[0]:times[1], 133:207, 188:213], axis=(0))), np.ravel(np.mean(runMP['cl_A'][times[0]:times[1]:4, 133:207, 188:213].data, axis=(0))), color='#f68080',s=50)
#        ax.set_ylim(np.min(np.mean(runMP['LWP'][times[0]:times[1], 133:207, 188:213].data, axis=0)),
#                          np.max(np.mean(runMP['LWP'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
#        ax.set_xlim(np.min(np.mean(runSEB['LW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))),
#                          np.max(np.mean(runSEB['LW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
        slope, intercept, r2, p, sterr = scipy.stats.linregress(np.ravel(np.mean(runSEB['melt'][times[0]:times[1], 133:207, 188:213], axis=(0))),
            np.ravel(np.mean(runMP['cl_A'][times[0]:times[1]:4, 133:207, 188:213].data, axis=(0))))
        if p <= 0.01:
            ax.text(0.75, 0.9, horizontalalignment='right', verticalalignment='top', s='r$^{2}$ = %s' % np.round(r2, decimals=2),
                          fontweight='bold', transform=ax.transAxes, size=24,color='dimgrey')
        else:
            ax.text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax.transAxes, size=24, color='dimgrey')
        ax.set_xlabel('Modelled melt (W m$^{-2}$)', size=24, color='dimgrey', rotation=0,labelpad=10)
        ax.set_ylabel('Modelled cloud area fraction', size=24, color='dimgrey', rotation=0, labelpad=80)
        lab = ax.text(0.1, 0.85, transform=ax.transAxes, s='a', fontsize=32, fontweight='bold', color='dimgrey')
        ax.spines['right'].set_visible(False)
    elif phase == 'ice':
        # SW vs IWP
        #ax.set_xlim(290,600)
        slope, intercept, r2, p, sterr = scipy.stats.linregress(np.ravel(np.mean(runSEB['SW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))),
            np.ravel(np.mean(runMP['IWP'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
        if p <= 0.01:
            ax.text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                              s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight='bold',
                              transform=ax.transAxes, size=24, color='dimgrey')
        else:
            ax.text(0.75, 0.9, horizontalalignment='right', verticalalignment='top', s='r$^{2}$ = %s' % np.round(r2, decimals=2),
                              transform=ax.transAxes, size=24,color='dimgrey')
        ax.scatter(np.ravel(np.mean(runSEB['SW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))), np.ravel(np.mean(runMP['IWP'][times[0]:times[1], 133:207, 188:213].data, axis=(0))),
                             color='#f68080', s=50)
        #ax.set_ylim(np.min(np.mean(runMP['IWP'][times[0]:times[1], 133:207, 188:213].data, axis=0)),
        #                  np.max(np.mean(runMP['IWP'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
        #ax.set_xlim(np.min(np.mean(runSEB['SW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))),
        #                  np.max(np.mean(runSEB['SW_down'][times[0]:times[1], 133:207, 188:213].data, axis=(0))))
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
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/melt_v_cloud_RA1M_mod_shifted'+phase+'.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/melt_v_cloud_RA1M_mod_shifted'+phase+'.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/melt_v_cloud_RA1M_mod_shifted'+phase+'.pdf', transparent=True)
    plt.show()



#correl_SEB_sgl(RA1M_mod_SEB, RA1M_mod_vars, times = (0,96), phase = 'liquid')



def vol_frac(run1, run2, run3, run4, which_var):
    fig, ax = plt.subplots(1,1, figsize=(10, 9))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax.set_ylim(0, max(run1['altitude']))
    #[l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    m1 = ax.plot(np.mean(run1[which_var][60:68,:40,199:201,199:201].data, axis = (0,2,3)), run1['altitude'], color= '#f87e7e', linestyle='--', linewidth=2.5, label = 'RA1M')
    m2 = ax.plot(np.mean(run2[which_var][60:68,:40,199:201,199:201].data, axis = (0,2,3)), run1['altitude'], color='#1f78b4', linestyle='--', linewidth=2.5, label = 'RA1M_mod')
    m3 = ax.plot(np.mean(run3[which_var][60:68,:40,199:201,199:201].data, axis = (0,2,3)), run1['altitude'], color='#33a02c', linestyle='--', linewidth=2.5, label= 'RA1T')
    m4 = ax.plot(np.mean(run4[which_var][60:68,:40,199:201,199:201].data, axis = (0,2,3)), run1['altitude'], color='dimgrey', linestyle='--', linewidth=2.5, label='RA1T_mod')
    ax.set_xlabel('Cloud volume fraction', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    ax.set_ylabel('Altitude \n(km)', rotation=0, fontname='SegoeUI semibold', fontsize=28, color='dimgrey',
                     labelpad=80)
    ax.set_xlim(0, 1.)
    for i, label in enumerate(ax.get_xticklabels()):
        if i > 0 and i < len(ax.get_xticklabels()) - 1:
            label.set_visible(False)
    ax.text(0.1, 0.85, transform=ax.transAxes, s='a', fontsize=32, fontweight='bold', color='dimgrey')
    plt.subplots_adjust(wspace=0.1, bottom=0.23, top=0.95, left=0.27, right=0.95)
    # handles, labels = ax[1].get_legend_handles_labels()
    # handles = [handles[0], handles[1], handles[-1], handles[2],  handles[3] ]
    # labels = [labels[0], labels[1], labels[-1], labels[2], labels[3]]
    # lgd = plt.legend(handles, labels, fontsize=20, markerscale=2)
    lgd = plt.legend(fontsize=20, markerscale=2)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/AWS14_vertical_profiles_'+which_var+'.eps')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/AWS14_vertical_profiles_'+which_var+'.png')
    plt.show()

vol_frac(RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars, which_var='cl_vol')
vol_frac(RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars, which_var='ice_cl')
vol_frac(RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars, which_var='liq_cl')