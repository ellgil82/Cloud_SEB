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
from divg_temp_colourmap import shiftedColorMap

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
    c = iris.load(pb)# IWP and LWP dont load properly
    IWP = c[1] # stash code s02i392
    LWP = c[0] # stash code s02i391
    #qc = c[3]
    #cl_A = iris.load_cube(pb, 'cloud_area_fraction_assuming_maximum_random_overlap')
    lsm = iris.load_cube(pa, 'land_binary_mask')
    orog = iris.load_cube(pa, 'surface_altitude')
    for i in [ice_mass_frac, liq_mass_frac]:#, qc]:
        real_lon, real_lat = rotate_data(i, 2, 3)
    for j in [LWP, IWP,]: #cl_A
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
    for i in [IWP, LWP, ice_mass_frac, liq_mass_frac,]: #qc
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
    box_mean_IWP = np.mean(IWP[times[0]:times[1], 133:207, 188:213].data)#, axis = (0,1,2))
    box_mean_LWP = np.mean(LWP[times[0]:times[1], 133:207, 188:213].data)#, axis =(0,1,2))
    mean_QCF = np.mean(box_QCF, axis=(0,2,3))
    mean_QCL = np.mean(box_QCL, axis=(0,2,3)) #0,2,3
    AWS14_mean_QCF = np.mean(ice_mass_frac[times[0]:times[1], :40, 199:201, 199:201].data, axis=(0, 2, 3))
    AWS14_mean_QCL = np.mean(liq_mass_frac[times[0]:times[1], :40, 199:201, 199:201].data, axis=(0, 2, 3))
    AWS15_mean_QCF = np.mean(ice_mass_frac[times[0]:times[1], :40, 161:163, 182:184].data, axis=(0, 2, 3))
    AWS15_mean_QCL = np.mean(liq_mass_frac[times[0]:times[1], :40, 161:163, 182:184].data, axis=(0, 2, 3))
    # Find max and min values at each model level
    time_mean_QCF = np.mean(box_QCF, axis=0)
    array = pd.DataFrame()
    for each_lat in np.arange(74):
        for each_lon in np.arange(25):
            for each_time in np.arange(times[1]-times[0]):
                m = pd.DataFrame(box_QCF[each_time, :, each_lat, each_lon])
                array = pd.concat([m, array], axis=1)
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
    max_QCL = array.max(axis=1)
    min_QCL = array.min(axis=1)
    # Calculate 95th percentile
    liq_95 = np.percentile(array, 95, axis=1)
    liq_5 = np.percentile(array, 5, axis=1)
    # Calculate PDF of ice and liquid water contents
    #liq_PDF = mean_liq.plot.density(color = 'k', linewidth = 1.5)
    #ice_PDF = mean_ice.plot.density(linestyle = '--', linewidth=1.5, color='k')
    altitude = ice_mass_frac.coord('level_height').points[:40]/1000
    var_dict = {'real_lon': real_lon, 'real_lat':real_lat,   'lsm': lsm, 'orog': orog,  'IWP': IWP, 'LWP':LWP,'mean_QCF': mean_QCF, 'mean_QCL': mean_QCL, 'altitude': altitude,
                 'AWS14_mean_QCF': AWS14_mean_QCF, 'AWS14_mean_QCL': AWS14_mean_QCL,'ice_5': ice_5, 'ice_95': ice_95, 'liq_5': liq_5, 'liq_95': liq_95, 'min_QCF': min_QCF,
                'max_QCF': max_QCF, 'min_QCL': min_QCL,'AWS15_mean_QCF': AWS15_mean_QCF, 'AWS15_mean_QCL': AWS15_mean_QCL, 'box_QCF': box_QCF, 'box_QCL': box_QCL} #cl_A': cl_A,'qc': qc,'box_mean_IWP': box_mean_IWP,'box_mean_LWP': box_mean_LWP,
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
    if config == 'CASIM_24':
        c = iris.load(pf)
        SW_net = c[0]
    else:
        SW_net = iris.load_cube(pf, 'surface_net_downward_shortwave_flux')
        Ts = iris.load_cube(pa, 'surface_temperature')
        Ts.convert_units('celsius')
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
    if config =='CASIM_24':
        var_dict = {'real_lon': real_lon, 'real_lat': real_lat, 'SW_up': SW_up, 'SW_down': SW_down,
                    'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down, 'LW_net': LW_net, 'SW_net': SW_net}
    else:
        var_dict = {'real_lon': real_lon, 'real_lat':real_lat,  'SW_up': SW_up, 'SW_down': SW_down,
                    'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down, 'LW_net': LW_net, 'SW_net': SW_net, 'Ts': Ts}
    return var_dict

# Load models in for times of interest: (59, 68) for time of flight, (47, 95) for midday-midnight (discard first 12 hours as spin-up)
#HM_vars = load_model('Hallett_Mossop', (11,21))
RA1M_mod_vars = load_model(config = 'RA1M_mod_24', flight_date = '20110118T0000', times = (47,95))
RA1T_mod_vars = load_model(config ='RA1T_mod_24', flight_date = '20110118T0000', times = (47,95))
RA1T_vars = load_model(config ='RA1T_24', flight_date = '20110118T0000', times = (47,95))
RA1M_vars = load_model(config ='RA1M_24', flight_date = '20110118T0000', times = (47,95))
CASIM_vars = load_model(config ='CASIM_24', flight_date = '20110118T0000', times = (47,95))
DeMott_vars = load_model(config ='DeMott', flight_date = '20110118T0000', times = (47,95))
#fl_av_vars = load_model(config ='fl_av', flight_date = '20110118T0000', times = (47,95))
model_runs = [RA1M_vars, RA1M_mod_vars,RA1T_vars, RA1T_mod_vars, CASIM_vars,  DeMott_vars]#fl_av_vars,

#t24_vars = load_model('24', (59,68))


def print_stats():
    model_mean = pd.DataFrame()
    for run in model_runs:
        #print('\n\nMean cloud box QCL of %(run)s is: '% locals()+str(np.mean(run['mean_QCL'])) )
        #print('\n\nMean cloud box QCF of %(run)s is: '% locals()+str(np.mean(run['mean_QCF'])) )
        #print('\n\nMean QCL at AWS 14 and 15 is ' + str(np.mean(run['AWS14_mean_QCL']))+ ' and ' + str(np.mean(run['AWS15_mean_QCL'])) + ', respectively in %(run)s' % locals())
        #print ('\n\nMean QCF at AWS 14 and 15 is '+str(np.mean(run['AWS14_mean_QCF']))+' and '+str( np.mean(run['AWS15_mean_QCF']))+', respectively in %(run)s \n\n' % locals())
        #print('\n\nMean cloud box LWP of %(run)s is: ' % locals() + str(run['box_mean_LWP']))
        #print('\n\nMean cloud box IWP of %(run)s is: ' % locals() + str(run['box_mean_IWP']))
        #print('\n\nMean LWP at AWS 14 and 15 is ' + str(run['AWS14_mean_LWP']) + ' and ' + str(run['AWS15_mean_LWP']) + ', respectively in %(run)s' % locals())
        #print('\n\nMean IWP at AWS 14 and 15 is ' + str(run['AWS14_mean_IWP']) + ' and ' + str(run['AWS15_mean_IWP']) + ', respectively in %(run)s \n\n' % locals())
        m = pd.DataFrame({'mean QCL': np.mean(run['mean_QCL']), 'std QCL profile': np.std(run['mean_QCL']), 'mean_QCF': np.mean(run['mean_QCF']), 'std QCF profile': np.std(run['mean_QCF']),
                          'AWS 14 QCL': np.mean(run['AWS14_mean_QCL']), 'std AWS 14 QCL profile': np.std(run['AWS14_mean_QCL']),'AWS 14 QCF' : np.mean(run['AWS14_mean_QCF']), 'std AWS 14 QCF profile': np.std(run['AWS14_mean_QCF']),
                          'AWS 15 QCL': np.mean(run['AWS15_mean_QCL']), 'std AWS 15 QCL profile': np.std(run['AWS15_mean_QCL']), 'AWS 15 QCF' : np.mean(run['AWS15_mean_QCF']), 'std AWS 15 QCF profile': np.std(run['AWS15_mean_QCF']),
                          'mean LWP': np.mean(run['LWP'][:,133:207, 188:213].data), 'std box LWP': np.std(np.mean(run['LWP'][:,133:207, 188:213].data, axis = (1,2))),'mean IWP': np.mean(run['IWP'][:,133:207, 188:213].data),
                          'std box IWP': np.std(np.mean(run['IWP'][:,133:207, 188:213].data, axis = (1,2))),'AWS 14 LWP': np.mean(run['LWP'][:,199:201, 199:201].data),  'std AWS 14 LWP': np.std(np.mean(run['LWP'][:,199:201, 199:201].data, axis = (1,2))),
                          'AWS 14 IWP': np.mean(run['IWP'][:,199:201, 199:201].data),'std AWS 14 IWP': np.std(np.mean(run['IWP'][:,199:201, 199:201].data, axis = (1,2))), 'AWS 15 LWP': np.mean(run['LWP'][:, 161:163, 182:184].data),
                          'std AWS 15 LWP': np.std(np.mean(run['LWP'][:,161:163, 182:184].data, axis = (1,2))), 'AWS 15 IWP': np.mean(run['IWP'][:, 161:163, 182:184].data),'std AWS 15 IWP': np.std(np.mean(run['IWP'][:, 161:163, 182:184].data, axis = (1,2)))}, index = [0])
        model_mean = pd.concat([model_mean, m])
        means = model_mean.mean(axis=0)
        print means

#print_stats()

def load_obs(flight, flight_date):
    ''' Inputs: flight number as a string, e.g. 'flight159' and flight_date as a string, with date only (not time) in YYYMMDD format, e.g. '20110125' '''
    ## ----------------------------------------------- SET UP VARIABLES --------------------------------------------------##
    ## Load core data
    print('\nYes yes cuzzy, pretty soon you\'re gonna have some nice core data...')
    bsl_path_core = '/data/mac/ellgil82/cloud_data/core_data/core_masin_'+flight_date+'_r001_'+flight+'_50hz.nc'
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

#mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, real_lon, real_lat = load_model('CASIM_ctrl')

## ================================================= PLOTTING ======================================================= ##

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans',
                               'Verdana']

latbox = [-68, -66.9, -66.9, -68]
lonbox = [-62, -62, -60.8, -60.8]

from matplotlib.patches import Polygon
def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, edgecolor='#222222', linewidth = 3, fill = False, facecolor = None, zorder = 5)
    plt.gca().add_patch(poly)

def obs_mod_profile(run):
    fig, ax = plt.subplots(1,2, figsize=(16, 9))
    ax = ax.flatten()
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs(flight = 'flight159', date = '20110125')
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axs.set_ylim(0, max(run['altitude']))
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    m_QCF = ax[0].plot(run['mean_QCF'], run['altitude'], color = 'k', linestyle = '--', linewidth = 2.5)
    p_QCF = ax[0].plot(IWC_profile, run['altitude'], color = 'k', linewidth = 2.5)
    ax[0].set_xlabel('Cloud ice mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    ax[0].set_ylabel('Altitude \n(km)', rotation = 0, fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 80)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0].set_xlim(0,0.02)
    ax[0].xaxis.get_offset_text().set_fontsize(24)
    ax[0].xaxis.get_offset_text().set_color('dimgrey')
    ax[0].fill_betweenx(run['altitude'], run['ice_5'], run['ice_95'], facecolor='lightslategrey', alpha=0.5)  # Shaded region between maxima and minima
    #ax[0].plot(run['ice_5'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)
    #ax[0].plot(run['ice_95'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
    ax[0].text(x=0.0015, y=4.8, s='a', fontsize=32, fontweight='bold', color='dimgrey')
    #m_14 = ax[0].plot(run['AWS14_mean_QCF'], run['altitude'], color='darkred', linestyle='--', linewidth=3)
    #m_15= ax[0].plot(run['AWS15_mean_QCF'], run['altitude'], color='darkblue', linestyle='--', linewidth=3)
    ax[1].set_xlabel('Cloud liquid mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    p_QCL = ax[1].plot(LWC_profile, run['altitude'], color = 'k', linewidth = 2.5, label = 'Observations')
    m_QCL = ax[1].plot(run['mean_QCL'], run['altitude'], color = 'k', linestyle = '--', linewidth = 2.5, label = 'Model: \'cloud\' box mean')
    ax[1].fill_betweenx(run['altitude'], run['liq_5'], run['liq_95'],  facecolor='lightslategrey', alpha=0.5, label = 'Model: 5$^{th}$ & 95$^{th}$ percentiles\n of \'cloud\' box range')  # Shaded region between maxima and minima
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
    ax[1].text(x=0.03, y=4.8, s='b', fontsize=32, fontweight='bold', color='dimgrey')
    plt.subplots_adjust(wspace=0.1, bottom=0.23, top=0.95, left=0.17, right=0.98)
    handles, labels = ax[1].get_legend_handles_labels()
    handles = [handles[0], handles[1], handles[-1]]#, handles[2],  handles[3] ]
    labels = [labels[0], labels[1], labels[-1]]#, labels[2], labels[3]]
    lgd = plt.legend(handles, labels, fontsize=20, markerscale=2)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.setp(ax[0].get_xticklabels()[-3], visible=False)
    plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profiles_RA1M_mod.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profiles_RA1M_mod.png', transparent = True)
    plt.show()


#obs_mod_profile(RA1M_mod_vars)

## Caption: mean modelled water paths (in g kg-1) over the Larsen C ice shelf during the time of flight 152 (16:00Z - 18:00Z)

model_runs = [RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars, CASIM_vars]#[RA1M_mod_vars]

#obs_mod_profile(RA1M_vars)

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
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/v11_mod_comparison_24.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/v11_mod_comparison_24.eps', transparent=True)
    plt.show()

column_totals()

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

#model_runs = [CASIM_vars]

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
    model_runs = [RA1M_vars, RA1M_mod_vars,  RA1T_vars, RA1T_mod_vars, CASIM_vars]#[RA1M_vars, RA1M_mod_vars] fl_av_vars -- CASIM has IWP and LWP in the wrong file stream
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
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/vn11_water_path_time_srs.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/vn11_water_path_time_srs.eps')
    plt.show()



#IWP_time_srs()


def all_mod_plot(run1, run2, run3, run4):
    fig, ax = plt.subplots(1,2, figsize=(16, 9))
    ax = ax.flatten()
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs()
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axs.set_ylim(0, max(run1['altitude']))
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    m1_QCF = ax[0].plot(run1['mean_QCF'], run1['altitude'], color = '#f87e7e', linestyle = '--', linewidth = 2.5, label = 'RA1M')
    m2_QCF = ax[0].plot(run2['mean_QCF'], run2['altitude'], color='#1f78b4', linestyle='--', linewidth=2.5, label = 'RA1M_mod')
    m3_QCF = ax[0].plot(run3['mean_QCF'], run3['altitude'], color='#33a02c', linestyle='--', linewidth=2.5, label='RA1T')
    m4_QCF = ax[0].plot(run4['mean_QCF'], run4['altitude'], color='dimgrey', linestyle='--', linewidth=2.5, label='RA1T_mod')
    p_QCF = ax[0].plot(IWC_profile, run1['altitude'], color = 'k', linewidth = 2.5, label = 'Observations')
    ax[0].set_xlabel('Cloud ice mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    ax[0].set_ylabel('Altitude \n(km)', rotation = 0, fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 80)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0].set_xlim(0,0.02)
    ax[0].xaxis.get_offset_text().set_fontsize(24)
    ax[0].xaxis.get_offset_text().set_color('dimgrey')
    #ax[0].fill_betweenx(run1['altitude'], run1['ice_5'], run1['ice_95'], facecolor='#f87e7e', alpha=0.5)
    #ax[0].fill_betweenx(run2['altitude'], run2['ice_5'], run2['ice_95'], facecolor='#1f78b4', alpha=0.3)# Shaded region between maxima and minima
    #ax[0].plot(run['ice_5'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)
    #ax[0].plot(run['ice_95'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
    ax[0].text(x=0.0015, y=4.8, s='a', fontsize=32, fontweight='bold', color='dimgrey')
    ax[1].set_xlabel('Cloud liquid mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    p_QCL = ax[1].plot(LWC_profile, run1['altitude'], color = 'k', linewidth = 2.5, label = 'Observations')
    m1_QCL = ax[1].plot(run1['mean_QCL'], run1['altitude'], color = '#f87e7e', linestyle = '--', linewidth = 2.5, label = 'RA1M')
    m2_QCL = ax[1].plot(run2['mean_QCL'], run2['altitude'], color='#1f78b4', linestyle='--', linewidth=2.5, label='RA1M_mod')
    m3_QCL = ax[1].plot(run3['mean_QCL'], run3['altitude'], color='#33a02c', linestyle='--', linewidth=2.5, label='RA1T')
    m4_QCL = ax[1].plot(run4['mean_QCL'], run4['altitude'], color='dimgrey', linestyle='--', linewidth=2.5, label='RA1T_mod')
    #ax[1].fill_betweenx(run1['altitude'], run1['liq_5'], run1['liq_95'],  facecolor='#f87e7e', alpha=0.5)
    #ax[1].fill_betweenx(run2['altitude'], run2['liq_5'], run2['liq_95'], facecolor='#1f78b4', alpha=0.3)# Shaded region between maxima and minima
    #ax[1].plot(run['liq_5'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2, label='')
    #ax[1].plot(run['liq_95'], run['altitude'], color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
    ax[1].axes.tick_params(axis = 'both', which = 'both', direction = 'in', length = 5, width = 1.5,  labelsize = 24, pad = 10)
    ax[1].tick_params(labelleft = 'off')
    ax[1].set_xlim(0,0.4)
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[1].xaxis.get_offset_text().set_fontsize(24)
    ax[1].xaxis.get_offset_text().set_color('dimgrey')
    ax[1].text(x=0.03, y=4.8, s='b', fontsize=32, fontweight='bold', color='dimgrey')
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
    plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profiles_mean_comparison.eps')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profiles_mean_comparison.png')
    plt.show()


#all_mod_plot(RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars)


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

RA1M_SEB = load_SEB(config = 'RA1M_24', flight_date= '20110118T0000')
RA1M_mod_SEB = load_SEB(config = 'RA1M_mod_24', flight_date= '20110118T0000Z')

model_runs = ['RA1M_SEB']

def SEB_correl(runSEB, runMP, times):
    fig, ax = plt.subplots(len(model_runs),2, sharex='col', figsize=(18, 15), subplot_kw = {'aspect':1})  # , squeeze=False)
    ax = ax.flatten()
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    plot = 0
    for run in model_runs:
        slope, intercept, r2, p, sterr = scipy.stats.linregress(np.mean(runSEB['LW_down'][times[0]:times[1],199:201, 199:201].data, axis = (1,2)), np.mean(runMP['LWP'][times[0]:times[1],199:201, 199:201].data, axis = (1,2)))
        if p <= 0.01:
            ax[plot].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight = 'bold', transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        else:
            ax[plot].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        ax[plot].scatter(np.mean(runSEB['LW_down'][times[0]:times[1],199:201, 199:201].data, axis = (1,2)), np.mean(runMP['LWP'][times[0]:times[1],199:201, 199:201].data, axis = (1,2)), color = '#f68080', s = 50)
        ax[plot].set_ylim(min(np.mean(runMP['LWP'][times[0]:times[1],199:201, 199:201].data, axis = (1,2))), max(np.mean(runMP['LWP'][times[0]:times[1],199:201, 199:201].data, axis = (1,2))))
        ax[plot].set_xlim(min(np.mean(runSEB['LW_down'][times[0]:times[1],199:201, 199:201].data, axis = (1,2))), max(np.mean(runSEB['LW_down'][times[0]:times[1],199:201, 199:201].data, axis = (1,2))))
        ax[plot].plot(ax[plot].get_xlim(), ax[plot].get_ylim(), ls="--", c = 'k', alpha = 0.8)
        slope, intercept, r2, p, sterr = scipy.stats.linregress(np.mean(runSEB['SW_down'][times[0]:times[1],199:201, 199:201].data, axis = (1,2)), np.mean(runMP['IWP'][times[0]:times[1],199:201, 199:201].data, axis = (1,2)))
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
        ax[plot+1].set_ylim(min(np.mean(runMP['IWP'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2))), max(np.mean(runMP['IWP'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2))))
        ax[plot+1].set_xlim(min(np.mean(runSEB['SW_down'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2))),max(np.mean(runSEB['SW_down'][times[0]:times[1], 199:201, 199:201].data, axis=(1, 2))))
        ax[plot+1].plot(ax[plot+1].get_xlim(), ax[plot+1].get_ylim(), ls="--", c='k', alpha=0.8)
        ax[plot].set_xlabel('Modelled LW$_{\downarrow}$ (W m$^{-2}$)', size = 24, color = 'dimgrey', rotation = 0, labelpad = 10)
        ax[plot].set_ylabel('Modelled LWP (g m$^{-2}$)', size = 24, color = 'dimgrey', rotation =0, labelpad= 80)
        ax[plot+1].set_xlabel('Modelled SW$_{\downarrow}$ (W m$^{-2}$)', size = 24, color = 'dimgrey', rotation = 0, labelpad = 10)
        ax[plot+1].set_ylabel('Modelled IWP (g m$^{-2}$)', size = 24, color = 'dimgrey', rotation =0, labelpad= 80)
        lab = ax[plot].text(0.1, 0.85, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        lab2 = ax[plot+1].text(0.1, 0.85, transform = ax[plot+1].transAxes, s=lab_dict[plot+1], fontsize=32, fontweight='bold', color='dimgrey')
        titles = ['    RA1M','    RA1M']#,'RA1M_mod','RA1M_mod','     fl_av', '     fl_av','    RA1T','    RA1T',  'RA1T_mod', 'RA1T_mod','   CASIM','   CASIM']
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
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set(adjustable='box-forced', aspect='equal')
        axs.set_aspect('equal', 'box')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
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
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/SEB_v_mp.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/SEB_v_mp.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Microphysics/SEB_v_mp.pdf', transparent=True)
    plt.show()


SEB_correl(RA1M_SEB, RA1M_vars, times = (69,79))