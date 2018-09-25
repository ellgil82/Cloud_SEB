''' Script for plotting mean temperature profiles from the UM and aircraft observations during in-cloud conditions,
for however many model experiments have been conducted.

(c) Ella Gilbert, 2018.
'''

## Import necessary python packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import iris
import os
import fnmatch
from matplotlib import rcParams

## Make sure we're in the right directory
os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/')

## Define functions
# Define function to rotate model data plotted on rotated grid
def rotate_data(var, lat_dim, lon_dim):
    ## Rotate projection
    #create numpy arrays of coordinates
    rotated_lat = var.coord('grid_latitude').points
    rotated_lon = var.coord('grid_longitude').points
    ## set up parameters for rotated projection
    pole_lon = var.coord('grid_longitude').coord_system.grid_north_pole_longitude
    pole_lat = var.coord('grid_latitude').coord_system.grid_north_pole_latitude
    #rotate projection
    real_lon, real_lat = iris.analysis.cartography.unrotate_pole(rotated_lon,rotated_lat, pole_lon, pole_lat)
    print ('\nunrotating pole...')
    lat = var.coord('grid_latitude')
    lon = var.coord('grid_longitude')
    lat = iris.coords.DimCoord(real_lat, standard_name='latitude',long_name="grid_latitude",var_name="lat",units=lat.units)
    lon= iris.coords.DimCoord(real_lon, standard_name='longitude',long_name="grid_longitude",var_name="lon",units=lon.units)
    var.remove_coord('grid_latitude')
    var.add_dim_coord(lat, data_dim=lat_dim)
    var.remove_coord('grid_longitude')
    var.add_dim_coord(lon, data_dim=lon_dim)
    return real_lon, real_lat

# Define function to load model data
def load_model(var, times): #times should be a range in the format 11,21
    pa = []
    print('\nimporting data from %(var)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/'):
            if fnmatch.fnmatch(file, '*%(var)s_pa*' % locals()):
                pa.append(file)
    T = iris.load_cube(pa, 'air_temperature')
    #Td = iris.load_cube(pa, 'wet_bulb_temperature')
    #P = iris.load_cube(pa, 'pressure')
    #P_surf = iris.load_cube(pa, 'surface_pressure')
    #MSLP = iris.load_cube(pa, 'mean_sea_level_pressure')
    lsm = iris.load_cube(pa, 'land_binary_mask')
    orog = iris.load_cube(pa, 'surface_altitude')
    for i in [T]:#, Td, P, ]:
        real_lon, real_lat = rotate_data(i, 2, 3)
    #for j in [P_surf, MSLP]:
    #    real_lon, real_lat = rotate_data(j, 1, 2)
    for k in [lsm, orog]:
        real_lon, real_lat = rotate_data(k, 0, 1)
    # Convert model data to g kg-1
    for i in [T]:#, Td]:
        i.convert_units('celsius')
    #for j in [P, MSLP, P_surf]:
    #    j.convert_units('hPa')
    # Convert times to useful ones
    for i in [T]:#, Td, P, MSLP, P_surf]:
        i.coord('time').convert_units('hours since 2011-01-18 00:00')
    altitude = T.coord('level_height').points[:40] / 1000
    ## ---------------------------------------- CREATE MODEL VERTICAL PROFILES ------------------------------------------ ##
    # Create mean vertical profiles for region of interest
    # region of interest = ice shelf. Longitudes of ice shelf along transect =
    # OR: region of interest = only where aircraft was sampling layer cloud: time 53500 to 62000 = 14:50 to 17:00
    # Define box: -62 to -61 W, -66.9 to -68 S
    # Coord: lon = 188:213, lat = 133:207, time = 4:6 (mean of preceding hours)
    print('\ncreating vertical profiles geez...')
    box_T = T[times[0]:times[1], :40, 133:207, 188:213].data
    #box_Td = Td[times[0]:times[1], :40, 133:207, 188:213].data
    box_mean_T = np.mean(T[times[0]:times[1], :40, 133:207, 188:213].data, axis = (0,2,3))
    #box_mean_Td = np.mean(Td[times[0]:times[1], 133:207, 188:213].data)#, axis =(0,1,2))
    AWS14_mean_T = np.mean(T[times[0]:times[1], :40, 199:201, 199:201].data, axis=(0, 2, 3))
    #AWS14_mean_Td = np.mean(Td[times[0]:times[1], :40, 199:201, 199:201].data, axis=(0, 2, 3))
    altitude = T.coord('level_height').points[:40]/1000
    var_dict = {'real_lon': real_lon, 'real_lat': real_lat, 'lsm': lsm, 'orog': orog, 'altitude': altitude, 'T': T, 'box_T': box_mean_T,
                 'AWS14_T': AWS14_mean_T}#,'Td': Td, 'P': P, 'MSLP': MSLP, 'P_surf': P_surf, 'box_Td': box_mean_Td, 'AWS14_Td': AWS14_mean_Td}
    return  var_dict

# Define function to load observations
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
    T_array =[]
    alt_array = []
    nconc_ice_all = np.sum(n_ice, axis=1)
    # Calculate number concentration averages only when the instrument detects liquid/ice particles
    for i in cloud_idx:
        if nconc_ice_all[i] > 0.00000001 or n_drop[i] > 1.0:# same threshold as in Lachlan-Cope et al. (2016)
            alt_array.append(plane_alt[i])
            T_array.append(core_temp[i])
    box_T = np.nanmean(T_array)
    # Calculate mean values at each height in the model
    # Create bins from model data
    print('\nbinning by altitude...')
    #Load model data to get altitude bins
    ice_mass_frac = iris.load_cube('/data/mac/ellgil82/cloud_data/um/means/20110118T0000Z_Peninsula_km1p5_Smith_tnuc_pc000.pp', 'mass_fraction_of_cloud_ice_in_air')
    bins =  ice_mass_frac.coord('level_height').points.tolist()
    # Find index of model level bin to which aircraft data would belong and turn data into pandas dataframe
    temp = {'alt_idx': np.digitize(alt_array, bins = bins), 'T': T_array}
    T_df = pd.DataFrame(data = temp)
    print('\ncreating observed profiles...')
    # Use groupby to group by altitude index and mean over the groups
    T_grouped = T_df.groupby(['alt_idx']).mean()
    T_profile = T_grouped['T'].values
    T_profile = np.append(T_profile, [0,0,0,0])
    T_profile = np.append([0,0,0], T_profile)
    return T_profile

## Load all data required (each model experiment should have a unique string that identifies it from all others)

#Jan_2011 = load_model('lg_t' )
RA1M_vars = load_model('RA1M_24', (59, 68))
RA1M_mod_vars = load_model('RA1M_mod_24', (59,68))
RA1T_vars = load_model('RA1T_24', (59,68))
RA1T_mod_vars = load_model('RA1T_mod_24', (59,68))
T_profile = load_obs()

## ================================================= PLOTTING ======================================================= ##

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans', 'Verdana']

## Define function to plot temperature profiles
def all_mod_plot(run1, run2, run3, run4):
    fig, ax = plt.subplots(1,1, figsize=(10, 9))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax.set_ylim(0, 4.51)
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    p = ax.plot(T_profile, run1['altitude'], color='k', linewidth=2.5, label='Observations')
    m1 = ax.plot(run1['box_T'], run1['altitude'], color = '#f87e7e', linestyle = '--', linewidth = 2.5, label = 'RA1M')
    m2 = ax.plot(run2['box_T'], run2['altitude'], color='#1f78b4', linestyle='--', linewidth=2.5, label = 'RA1M_mod')
    m3 = ax.plot(run3['box_T'], run3['altitude'], color='#33a02c', linestyle='--', linewidth=2.5, label='RA1T')
    m4 = ax.plot(run4['box_T'], run4['altitude'], color='dimgrey', linestyle='--', linewidth=2.5, label='RA1T_mod')
    ax.set_xlabel('Air temperature ($^{\circ}$C)', fontname='SegoeUI semibold', color='dimgrey',
                     fontsize=28, labelpad=35)
    ax.set_ylabel('Altitude \n(km)', rotation = 0, fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 80)
    ax.set_xlim(-30, 10)
    plt.subplots_adjust(wspace=0.1, bottom=0.25, top=0.95, left=0.26, right=0.96)
    lgd = plt.legend(fontsize = 20, markerscale = 2)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.setp(ax.get_xticklabels()[-2], visible=False)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/T_profiles_mean_comparison.eps')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/T_profiles_mean_comparison.png')
    plt.show()

## Plot temperature profiles
all_mod_plot(RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars)
