import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import iris
import os
import fnmatch
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import sys
sys.path.append('/users/ellgil82/scripts/Tools/')
from tools import compose_date, compose_time
from rotate_data import rotate_data
from matplotlib import rcParams
from matplotlib.lines import Line2D


# Load model data
def load_model(config, flight_date, times): #times should be a range in the format 11,21
    pa = []
    pb = []
    pc = []
    pf = []
    print('\nimporting data from %(config)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/f150'):
            if fnmatch.fnmatch(file, flight_date + '*%(config)s_pb*' % locals()):
                pb.append(file)
            elif fnmatch.fnmatch(file, flight_date + '*%(config)s_pa*' % locals()):
                pa.append(file)
            elif fnmatch.fnmatch(file, flight_date + '*%(config)s_pc*' % locals()):
                pc.append(file)
            elif fnmatch.fnmatch(file, flight_date) & fnmatch.fnmatch(file, '*%(config)s_pf*' % locals()):
                pf.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/f150/')
    print('\nIce mass fraction')
    try:
        ice_mass_frac = iris.load_cube(pb, iris.Constraint(name='mass_fraction_of_cloud_ice_in_air'))
    except iris.exceptions.ConstraintMismatchError:
        print('\n Can\'t find \"mass_fraction_of_cloud_ice_in_air\" in this file, searching STASH...\n')
        ice_mass_frac = iris.load_cube(pb, iris.Constraint(STASH='m01s00i012'))
        print('\n Nope. Soz. Not \'ere. \n')
    print('\nliquid mass fraction')
    try:
        liq_mass_frac = iris.load_cube(pb, iris.Constraint(name='mass_fraction_of_cloud_liquid_water_in_air'))
    except iris.exceptions.ConstraintMismatchError:
        print('\n Can\'t find \"mass_fraction_of_cloud_liquid_water_in_air\" in this file, searching STASH...\n')
        ice_mass_frac = iris.load_cube(pb, iris.Constraint(STASH='m01s00i254'))
        print('\n Nope. Soz. Not \'ere. \n')
    print('\nice water path') # as above, and convert from kg m-2 to g m-2
    try:
        IWP = iris.load_cube(pb, iris.AttributeConstraint(STASH='m01s02i392'))# stash code s02i392
    except iris.exceptions.ConstraintMismatchError:
        print('\n Nope. Soz. Not \'ere. \n')
    print('\nliquid water path')
    try:
        LWP = iris.load_cube(pb, iris.AttributeConstraint(STASH='m01s02i391'))
    except iris.exceptions.ConstraintMismatchError:
        print('\n Nope. Soz. Not \'ere. \n')
    print('\nCloud droplet number concentration')
    try:
        CDNC = iris.load_cube(pc, iris.AttributeConstraint(STASH='m01s00i075'))
        CDNC_transect = np.mean(CDNC[:, :, 127:137, 130:185].data, axis=(0, 1, 2))
    except iris.exceptions.ConstraintMismatchError:
        print('\n Nope. Soz. Not \'ere. \n')
    print('\nIce crystal number concentration')
    try:
        ice_number = iris.load_cube(pc, iris.AttributeConstraint(STASH='m01s00i078'))
        ice_num_transect = np.mean(ice_number[:, :, 127:137, 130:185].data, axis=(0, 1, 2))
        for l in [CDNC, ice_number]:
            real_lon, real_lat = rotate_data(l, 2, 3)
            l = l/1e6
            l.convert_units('cm-3')
        CDNC_5 = np.percentile(CDNC[times[0]:times[1], :40, 115:150, 130:185].data, 5, axis=(0, 1, 2))
        CDNC_95 = np.percentile(CDNC[times[0]:times[1], :40, 115:150, 130:185].data, 95, axis=(0, 1, 2))
    except iris.exceptions.ConstraintMismatchError:
        print('\n Nope. Soz. Not \'ere. \n')
    #print('\nAir temperature') # as above, and convert from kg m-2 to g m-2
    #try:
    #    T = iris.load_cube(pa, iris.Constraint(name='air_temperature'))
    #    T.convert_units('celsius')
    #except iris.exceptions.ConstraintMismatchError:
    #    print('\n Can\'t find \"air_temperature\" in this file, searching STASH...\n')
    #    T = iris.load_cube(pa, iris.Constraint(STASH='m01s16i004'))
    #    T.convert_units('celsius')
    #except iris.exceptions.ConstraintMismatchError:
    #    print('\n Nope. Soz. Not \'ere. \n')
    lsm = iris.load_cube(pa, 'land_binary_mask')
    orog = iris.load_cube(pa, 'surface_altitude')
    # Rotate data to ordinary lats/lons and convert units to g kg-1 and g m-2
    for i in [ice_mass_frac, liq_mass_frac]:
        real_lon, real_lat = rotate_data(i, 2, 3)
        i.convert_units('g kg-1')
    for j in [LWP, IWP,]: #cl_A
        real_lon, real_lat = rotate_data(j, 1, 2)
        j.convert_units('g m-2')
    for k in [lsm, orog]:
        real_lon, real_lat = rotate_data(k, 0, 1)
    # Convert times to useful ones
    for i in [IWP, LWP, ice_mass_frac, liq_mass_frac,]: #qc
        i.coord('time').convert_units('hours since 2011-01-15 00:00')
    ## ---------------------------------------- CREATE MODEL TRANSECTS ------------------------------------------ ##
    # Create mean vertical profiles for region of interest
    # region of interest = ice shelf. Longitudes of ice shelf along transect =
    # OR: region of interest = only where aircraft was sampling layer cloud: time 53500 to 62000 = 14:50 to 17:00
    # Define box: -62 to -61 W, -66.9 to -68 S
    # Coord: lon = 188:213, lat = 133:207, time = 4:6 (mean of preceding hours)
    # Calculate number concentration averages only when the instrument detects liquid/ice particles
    print('\ncreating transects geez...')
    box_QCF = ice_mass_frac[times[0]:times[1], :40, 115:150, 130:185].data
    box_QCL = liq_mass_frac[times[0]:times[1], :40, 115:150, 130:185].data
    transect_box_QCF = ice_mass_frac[times[0]:times[1], :40, 127:137, 130:185].data
    transect_box_QCL = liq_mass_frac[times[0]:times[1], :40, 127:137, 130:185].data
    # mask grid cells where no cloud is present so as not to bias the mean according to mass fraction threshold given
    # in Gettelman et al. (2010) JGR 115 (D18) 1-19. doi:10.1029/2009JD013797
    QCF_transect = np.ma.masked_where(np.mean(ice_mass_frac[:,:,127:137, 130:185 ].data, axis = (0,1,2)),
                                      np.mean(ice_mass_frac[:,:,127:137, 130:185 ].data, axis = (0,1,2)) <= 0.005)
    QCL_transect = np.ma.masked_where(np.mean(liq_mass_frac[:, :, 127:137, 130:185].data, axis=(0, 1, 2)),
                                      np.mean(liq_mass_frac[:, :, 127:137, 130:185].data, axis=(0, 1, 2)) <= 0.005)
    #transect_box_T = T[times[0]:times[1], :40, 127:137, 130:185].data
    box_mean_IWP = np.mean(IWP[times[0]:times[1], 115:150, 130:185].data)#, axis = (0,1,2))
    box_mean_LWP = np.mean(LWP[times[0]:times[1], 115:150, 130:185].data)#, axis =(0,1,2))
    IWP_transect = np.mean(IWP[:, 127:137, 130:185 ].data, axis = (0,1))
    LWP_transect = np.mean(LWP[:, 127:137, 130:185  ].data, axis = (0,1))
    QCL_profile = np.mean(liq_mass_frac[:,:,127:137, 130:185].data, axis = (0,2,3))
    #T_profile = np.mean(T[:,:,127:137, 130:185].data, axis = (0,2,3))
    # Calculate 5th and 95th percentiles for each longitude bin, and for vertical profile
    ice_95 = np.percentile(box_QCF, 95, axis=(0,1,2))
    ice_5 = np.percentile(box_QCF, 5, axis=(0,1,2))
    liq_95 = np.percentile(box_QCL, 95, axis=(0,1,2))
    liq_5 = np.percentile(box_QCL, 5, axis=(0,1,2))
    vert_95 = np.percentile(box_QCL, 95, axis=(0,2,3))
    vert_5 = np.percentile(box_QCL, 5, axis=(0, 2, 3))
    # Calculate PDF of ice and liquid water contents
    #liq_PDF = mean_liq.plot.density(color = 'k', linewidth = 1.5)
    #ice_PDF = mean_ice.plot.density(linestyle = '--', linewidth=1.5, color='k')
    if 'CDNC' and 'T_profile' in locals():
        var_dict = {'real_lon': real_lon, 'real_lat':real_lat,   'lsm': lsm, 'orog': orog,  'IWP': IWP, 'LWP':LWP, 'ice_5': ice_5,
                    'ice_95': ice_95, 'liq_5': liq_5, 'liq_95': liq_95, 'box_QCF': box_QCF, 'box_QCL': box_QCL, 'vert_5': vert_5,
                     'vert_95': vert_95, 'LWP_transect': LWP_transect,'IWP_transect': IWP_transect, 'QCL_profile': QCL_profile,
                    'QCF_transect': QCF_transect, 'QCL_transect': QCL_transect, 'QCF': ice_mass_frac, 'QCL': liq_mass_frac,
                    'CDNC': CDNC, 'ice_number': ice_number, 'CDNC_transect': CDNC_transect, 'ice_num_transect': ice_num_transect,
                    'CDNC_5': CDNC_5, 'CDNC_95': CDNC_95, 'T_profile': T_profile, 'transect_box_T': transect_box_T}
    elif 'T_profile' in locals():
        var_dict = {'real_lon': real_lon, 'real_lat':real_lat,   'lsm': lsm, 'orog': orog,  'IWP': IWP, 'LWP':LWP, 'ice_5': ice_5,
                    'ice_95': ice_95, 'liq_5': liq_5, 'liq_95': liq_95, 'box_QCF': box_QCF, 'box_QCL': box_QCL, 'vert_5': vert_5,
                     'vert_95': vert_95, 'LWP_transect': LWP_transect,'IWP_transect': IWP_transect, 'QCL_profile': QCL_profile,
                    'QCF_transect': QCF_transect, 'QCL_transect': QCL_transect, 'QCF': ice_mass_frac, 'QCL': liq_mass_frac,
                    'T_profile': T_profile, 'transect_box_T': transect_box_T}
    else:
        var_dict = {'real_lon': real_lon, 'real_lat':real_lat,   'lsm': lsm, 'orog': orog,  'IWP': IWP, 'LWP':LWP, 'ice_5': ice_5,
                    'ice_95': ice_95, 'liq_5': liq_5, 'liq_95': liq_95, 'box_QCF': box_QCF, 'box_QCL': box_QCL, 'vert_5': vert_5,
                     'vert_95': vert_95, 'LWP_transect': LWP_transect,'IWP_transect': IWP_transect, 'QCL_profile': QCL_profile,
                    'QCF_transect': QCF_transect, 'QCL_transect': QCL_transect, 'QCF': ice_mass_frac, 'QCL': liq_mass_frac}
    return  var_dict

# Load models in for times of interest: (59, 68) for time of flight, (47, 95) for midday-midnight (discard first 12 hours as spin-up)
#RA1M_mod_vars = load_model(config = 'RA1M_mods_f150', flight_date = '20110115T0000', times = (47,95))
#Cooper_vars = load_model(config = 'Cooper', flight_date = '20110115T0000', times = (47,95))
#DeMott_vars = load_model(config = 'f150_DeMott', flight_date = '20110115T0000',  times = (47,95))
#DeMott_2015_vars = load_model(config = 'DeMott_2015', flight_date = '20110115T0000',  times = (47,95))
#model_runs = [RA1M_vars, RA1M_mod_vars,RA1T_vars, RA1T_mod_vars]#, CASIM_vars fl_av_vars, ]

def load_SEB(config, flight_date):
    pa = []
    pf = []
    print('\nimporting data from %(config)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/f150/'):
        if fnmatch.fnmatch(file, flight_date + '*%(config)s_pf*' % locals()):
            pf.append(file)
        elif fnmatch.fnmatch(file, flight_date + '*%(config)s_pa*' % locals()):
            pa.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/f150/')
    lsm = iris.load_cube(pa, 'land_binary_mask')
    orog = iris.load_cube(pa, 'surface_altitude')
    LW_net = iris.load_cube(pf, 'surface_net_downward_longwave_flux')
    SH =  iris.load_cube(pf, 'surface_upward_sensible_heat_flux')
    LH = iris.load_cube(pf, 'surface_upward_latent_heat_flux')
    LW_down = iris.load_cube(pf, 'surface_downwelling_longwave_flux')
    LW_up = iris.load_cube(pf, 'upwelling_longwave_flux_in_air')
    SW_up = iris.load_cube(pf, 'upwelling_shortwave_flux_in_air')
    if config == 'DeMott_2015' or 'f150_DeMott' or 'Cooper':
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
        i.coord('time').convert_units('hours since 2011-01-15 00:00')
    LH = 0 - LH.data
    SH = 0 - SH.data
    if config == 'DeMott_2015' or 'f150_DeMott' or 'Cooper':
        var_dict = {'real_lon': real_lon, 'real_lat': real_lat, 'SW_up': SW_up, 'SW_down': SW_down,
                    'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down, 'LW_net': LW_net, 'SW_net': SW_net}
    else:
        var_dict = {'real_lon': real_lon, 'real_lat':real_lat,  'SW_up': SW_up, 'SW_down': SW_down,
                    'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down, 'LW_net': LW_net, 'SW_net': SW_net, 'Ts': Ts}
    return var_dict

RA1M_mod_SEB = load_SEB(config = 'RA1M_mods_f150', flight_date= '20110115T0000Z')
Cooper_SEB = load_SEB(config = 'Cooper', flight_date = '20110115T0000')
DeMott_SEB = load_SEB(config = 'f150_DeMott', flight_date = '20110115T0000')
#DeMott_2015_SEB = load_model(config = 'DeMott_2015', flight_date = '20110115T0000')

def load_obs():
    ## ----------------------------------------------- SET UP VARIABLES --------------------------------------------------##
    ## Load core data
    print('\nYes yes cuzzy, pretty soon you\'re gonna have some nice core data...')
    bsl_path_core = '/data/mac/ellgil82/cloud_data/core_data/core_masin_20110115_r001_flight150_50hz.nc'
    cubes = iris.load(bsl_path_core)
    core_temp = cubes[-12] #de-iced temperature
    core_temp = core_temp -273.15
    plane_lat = iris.load_cube(bsl_path_core, 'latitude')
    plane_lon = iris.load_cube(bsl_path_core, 'longitude')
    plane_alt = iris.load_cube(bsl_path_core, 'altitude')#plane_alt = plane_alt.data[72015:-30046]
    core_time =  iris.load_cube(bsl_path_core, 'time')
    # subsample core data
    core_temp = core_temp.data[0::50]
    plane_lat = plane_lat.data[0::50]
    plane_alt = plane_alt.data[0::50]
    plane_lon = plane_lon.data[0::50]
    core_time = core_time.data[0::50]
    # Trim times to match shortest data length (= CIP)
    core_time = core_time[1455:8667]
    plane_alt = plane_alt[1455:8667]
    plane_lon = plane_lon[1455:8667]
    plane_lat = plane_lat[1455:8667]
    core_temp = core_temp[1455:8667]
    ## Load CIP data
    # Load CIP from .npz
    print('\nOi mate, right now I\'m loading some siiiiick CIP data...')
    path = '/data/mac/ellgil82/cloud_data/Constantino_Oasis_Peninsula/'
    s_file = 'flight150_s_v2.npz'
    npz_s=np.load(path+s_file)
    m_file = 'flight150_m_v2.npz'
    npz_m = np.load(path + m_file)
    n_file = 'flight150_n_v2.npz'
    npz_n = np.load(path + n_file)
    CIP_time = npz_m['time']
    CIP_bound = npz_s['TestPlot_all_y']
    m_all = npz_m['TestPlot_all_y']
    IWC = npz_m['TestPlot_HI_y']+ npz_m['TestPlot_MI_y']
    S_LI = npz_m['TestPlot_LI_y']+npz_m['TestPlot_S_y']
    n_drop_CIP = npz_n['TestPlot_LI_y']+npz_n['TestPlot_S_y']
    # Load CAS data
    CAS_file = '/data/mac/ellgil82/cloud_data/netcdfs/flight150_cas.nc'
    # Create variables
    print ('\nOn dis CAS ting...')
    LWC_cas = iris.load_cube(CAS_file, 'liquid water content calculated from CAS ')
    LWC_cas = LWC_cas.data[1294:8506]
    CAS_time = iris.load_cube(CAS_file, 'time')
    CAS_time = CAS_time[1294:8506]
    aer = iris.load_cube(CAS_file, 'Aerosol concentration spectra measured by cas ')
    aer = aer[:, 1294:8506]
    n_drop_CAS = np.sum(aer[8:,:].data, axis=0)
    n_drop =  n_drop_CAS[:15348] #n_drop_CIP[1:] +
    ## ----------------------------------------- PERFORM CALCULATIONS ON DATA --------------------------------------------##
    # Find number concentrations of ice only
    n_ice = npz_s['TestPlot_HI_z']+npz_s['TestPlot_MI_z'] # Consider high and mid-irregular particles to be ice
    n_ice = n_ice * 2. # correct data (as advised by TLC and done by Constantino for their 2016 and 2017 papers)
    n_ice = n_ice/1000 #in cm-3
    n_ice = n_ice[1:]
    CIP_mean_ice = []
    j = np.arange(64)
    for i in j:#
        m = np.mean(n_ice[:,i])
        CIP_mean_ice = np.append(CIP_mean_ice,m)
    nconc_ice_all = np.sum(n_ice, axis=1) # sum of high and mid-irregular
    # Convert times
    unix_time = 1295049600
    CIP_real_time = CIP_time + unix_time
    s = pd.Series(CIP_real_time)
    CIP_time = pd.to_datetime(s, unit='s')
    core_time = core_time/1000
    core_time = core_time + unix_time
    core_time = pd.Series(core_time)
    core_time = pd.to_datetime(core_time, unit='s')
    CAS_time = np.ndarray.astype(CAS_time.data, float)
    CAS_time = CAS_time / 1000
    CAS_real_time = CAS_time + unix_time
    s = pd.Series(CAS_real_time)
    CAS_time = pd.to_datetime(s, unit='s')
    ## ------------------------------------- COMPUTE WHOLE-FLIGHT STATISTICS ---------------------------------------------##
    # FIND IN-CLOUD LEGS
    # Find only times when flying over ice shelf
    print('\nYEAH BUT... IS IT CLOUD DOE BRUH???')
    idx = np.where(plane_lon.data > -64.3) # only in the region of interest (away from mountains, in 'cloud box')
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
    temp_array_ice = []
    temp_array_liq = []
    alt_array_ice = []
    alt_array_liq = []
    lon_array_liq = []
    lon_array_ice = []
    nconc_ice = []
    drop_array = []
    # Calculate number concentration averages only when the instrument detects liquid/ice particles
    for i in cloud_idx:
        if nconc_ice_all[i] > 0.00000001:
            IWC_array.append(IWC[i])
            alt_array_ice.append(plane_alt[i])
            nconc_ice.append(nconc_ice_all[i])
            lon_array_ice.append(plane_lon[i])
            temp_array_ice.append(core_temp[i])
        elif n_drop[i] > 1.0:  # same threshold as in Lachlan-Cope et al. (2016)
            drop_array.append(n_drop[i])
            LWC_array.append(LWC_cas[i])
            alt_array_liq.append(plane_alt[i])
            lon_array_liq.append(plane_lon[i])
            temp_array_liq.append(core_temp[i])
    ## Create longitudinal transects
    # Calculate mean values at each height in the model
    # Create bins from model data
    print('\nbinning by altitude...')
    #Load model data to get altitude/longitude bins
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/f150/')
    T = iris.load_cube('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/f150/20110115T0000Z_Peninsula_1p5km_RA1M_mods_f150_pa000.pp', 'air_temperature')
    alt_bins = T.coord('level_height').points.tolist()
    real_lon, real_lat = rotate_data(T, 2,3)
    lon_bins = T.coord('longitude')[130:185].points.tolist()
    # Find index of model longitude bin to which aircraft data would belong
    # Turn data into pandas dataframe
    d_liq = {'LWC': LWC_array, 'lon_idx': np.digitize(lon_array_liq, bins=lon_bins), 'alt_idx': np.digitize(alt_array_liq, bins=alt_bins), 'alt': alt_array_liq, 'temp': temp_array_liq, 'lons': lon_array_liq,  'n_drop': drop_array}
    d_ice = {'IWC': IWC_array,'n_ice': nconc_ice, 'lon_idx': np.digitize(lon_array_ice, bins=lon_bins), 'lons': lon_array_ice, }
    df_liq = pd.DataFrame(data = d_liq)
    df_ice = pd.DataFrame(data= d_ice)
    print('\ncreating observed transects...')
    # Use groupby to group by longitude index and mean over the groups
    grouped_liq = df_liq.set_index(['lon_idx'])
    grouped_ice = df_ice.set_index(['lon_idx'])
    # Calculate ice water contents and means only when ice is present
    # Separate into outward and return legs
    IWC_leg1 = grouped_ice['IWC'][:70]# outward
    IWC_leg2 = grouped_ice['IWC'][70:]
    LWC_leg1 = grouped_liq['LWC'][:520] # outward leg
    LWC_leg2 = grouped_liq['LWC'][520:]
    nice_leg1 = grouped_ice['n_ice'][:70]
    nice_leg2 = grouped_ice['n_ice'][70:]
    drop_leg1 = grouped_liq['n_drop'][:520]
    drop_leg2 = grouped_liq['n_drop'][520:]
    ice_lons_leg1 = grouped_ice[:70]['lons']
    ice_lons_leg2 = grouped_ice[70:]['lons']
    liq_lons_leg1 = grouped_liq[:520]['lons']
    liq_lons_leg2 = grouped_liq[520:]['lons']
    # Calculate means of non-zero points for each variable for the transects
    IWC_transect1 = grouped_ice[:70].groupby(['lon_idx']).mean()['IWC'] #25, 31,32, 35, 59, 60
    LWC_transect1 = grouped_liq[:520].groupby(['lon_idx']).mean()['LWC']
    ice_transect1 = grouped_ice[:70].groupby(['lon_idx']).mean()['n_ice']
    drop_transect1 = grouped_liq[:520].groupby(['lon_idx']).mean()['n_drop']
    IWC_transect2 = grouped_ice[70:].groupby(['lon_idx']).mean()['IWC'] #25, 31,32, 35, 59, 60
    LWC_transect2 = grouped_liq[520:].groupby(['lon_idx']).mean()['LWC']
    ice_transect2 = grouped_ice[70:].groupby(['lon_idx']).mean()['n_ice']
    drop_transect2 = grouped_liq[520:].groupby(['lon_idx']).mean()['n_drop']
    temp_transect = grouped_liq.groupby(['lon_idx']).mean()['temp']
        # Add in some zeros at correct place to make mean transect the right shape for plotting
    def append_1(transect):
        transect = np.append(np.zeros(9), transect)
        transect = np.append(transect, np.zeros(10))# [24, 30,31,34], [0])
        transect = np.insert(transect,[25],[0,0,0,0])
        transect = np.insert(transect, [31], [0,0,0,0,0,0,0])
        return transect
    LWC_transect2, drop_transect2 = np.append(LWC_transect2, np.zeros(9)), np.append(drop_transect2, np.zeros(9))
    LWC_transect1, drop_transect1 = append_1(LWC_transect1), append_1(drop_transect1)
    temp_transect = np.append(temp_transect, np.zeros(10)+np.nan)
    IWC_transect1 = np.append(IWC_transect1, np.zeros(12))
    IWC_transect1 = np.insert(IWC_transect1, [15], np.zeros(14))
    ## Create vertical profiles
    grouped_liq = df_liq.set_index(['alt_idx'])
    LWC_profile = grouped_liq[:520].groupby(['alt_idx']).mean()['LWC']
    LWC_profile = np.append(np.zeros(10), LWC_profile)
    LWC_profile = np.insert(LWC_profile, [21], [0,0,0,0,0])
    return aer, IWC_array, LWC_array, lon_bins, IWC_transect1, LWC_transect1, drop_transect1, alt_array_liq, temp_transect, temp_array_liq,\
           ice_transect1, nice_leg1, drop_leg1, ice_lons_leg1, ice_lons_leg2, IWC_leg1, LWC_leg1, IWC_transect2, \
           LWC_transect2, drop_transect2, ice_transect2, nice_leg2, drop_leg2,liq_lons_leg1, liq_lons_leg2, IWC_leg2, LWC_leg2, LWC_profile, IWC_transect1

aer, IWC_array, LWC_array,lon_bins, IWC_transect1, LWC_transect1, drop_transect1,  alt_array_liq,  temp_transect, temp_array_liq,ice_transect1, nice_leg1, drop_leg1, ice_lons_leg1, ice_lons_leg2, IWC_leg1, \
LWC_leg1, IWC_transect2, LWC_transect2, drop_transect2, ice_transect2, nice_leg2, drop_leg2,liq_lons_leg1, liq_lons_leg2, IWC_leg2, LWC_leg2, LWC_profile, IWC_transect1 = load_obs()

def print_stats(): # Add drop means etc.
    model_mean = pd.DataFrame()
    for run in model_runs:
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

AWS_loc = pd.read_csv('/data/clivarm/wip/ellgil82/AWS/AWS_loc.csv', header = 0)
AWS_list = ['AWS15', 'AWS14']#, 'OFCAP']


def load_AWS(station):
    ## --------------------------------------------- SET UP VARIABLES ------------------------------------------------##
    ## Load data from AWS 14 and AWS 15 for January 2011
    print('\nDayum grrrl, you got a sweet AWS...')
    os.chdir('/data/clivarm/wip/ellgil82/AWS/')
    for file in os.listdir('/data/clivarm/wip/ellgil82/AWS/'):
        if fnmatch.fnmatch(file, '%(station)s_Jan_2011*' % locals()):
            AWS = pd.read_csv(str(file), header = 0)
            print(AWS.shape)
    Jan18 = AWS.loc[(AWS['Day'] == 15)]# & (AWS['Hour'] >= 12)]
    #Jan18 = Jan18.append(AWS.loc[(AWS['Day'] == 19) & (AWS['Hour'] == 0)])
    Day_mean = Jan18.mean(axis=0) # Calculates averages for whole day
    Flight = Jan18.loc[(Jan18['Hour'] >=17) &  (Jan18['Hour'] <= 20)]
    Flight_mean = Flight.mean(axis=0) # Calculates averages over the time period sampled (17:00 - 20:00)
    return Flight_mean, Day_mean, Jan18

AWS14_flight_mean, AWS14_day_mean, AWS14_Jan = load_AWS('AWS14')
AWS15_flight_mean, AWS15_day_mean, AWS15_Jan = load_AWS('AWS15')
AWS14_SEB_flight_mean, AWS14_SEB_day_mean, AWS14_SEB_Jan  = load_AWS('AWS14_SEB')

def shift_AWS(station):
    os.chdir('/data/clivarm/wip/ellgil82/AWS/')
    for file in os.listdir('/data/clivarm/wip/ellgil82/AWS/'):
        if fnmatch.fnmatch(file, '%(station)s_Jan_2011*' % locals()):
            AWS = pd.read_csv(str(file), header=0)
            print(AWS.shape)
    Jan18 = AWS.loc[(AWS['Day'] == 15)  & (AWS['Hour'] >= 5)]
    Jan18 = Jan18.append(AWS.loc[(AWS['Day'] == 16) & (AWS['Hour'] < 5)])
    Day_mean = Jan18.mean(axis=0)  # Calculates averages for whole day
    Flight = Jan18.loc[(Jan18['Hour'] >= 13) & (Jan18['Hour'] <= 16)]
    Flight_mean = Flight.mean(axis=0)  # Calculates averages over the time period sampled (17:00 - 20:00)
    return Flight_mean, Day_mean, Jan18

# Shift AWS 14 data to match observations = cloud is simulated too early (by around 5 hours), so need to calculate means
# when the cloud fields are more comparable
AWS14_SEB_flight_mean, AWS14_SEB_day_mean, AWS14_SEB_Jan  = shift_AWS('AWS14_SEB')

## ----------------------------------------------- COMPARE MODEL & AWS ---------------------------------------------- ##

real_lat = Cooper_SEB['real_lat']
real_lon = Cooper_SEB['real_lon']

## Finds closest model gridbox to specified point in real lat, lon coordinates (not indices)
def find_gridloc(x,y):
    lat_loc = np.argmin((real_lat-y)**2) #take whole array and subtract lat you want from each point, then find the smallest difference
    lon_loc = np.argmin((real_lon-x)**2)
    return lon_loc, lat_loc

AWS14_lon, AWS14_lat = find_gridloc(AWS_loc['AWS14'][1], AWS_loc['AWS14'][0])
AWS14_real_lon = real_lon[AWS14_lon]
AWS14_real_lat = real_lat[AWS14_lat]
AWS15_lon, AWS15_lat = find_gridloc(AWS_loc['AWS15'][1], AWS_loc['AWS15'][0])
AWS15_real_lon = real_lon[AWS15_lon]
AWS15_real_lat = real_lat[AWS15_lat]

## -------------------------------------------- CALCULATE TOTAL SEB ------------------------------------------------- ##

os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/f150/')

def calc_SEB(run, times):
    AWS14_SEB_flight = AWS14_SEB_flight_mean['SWnet_corr'] + AWS14_SEB_flight_mean['LWnet_corr'] + AWS14_SEB_flight_mean['Hsen'] + AWS14_SEB_flight_mean['Hlat'] - AWS14_SEB_flight_mean['Gs']
    AWS14_melt_flight = AWS14_SEB_flight_mean['melt_energy']
    AWS14_SEB_day = AWS14_SEB_day_mean['SWnet_corr'] + AWS14_SEB_day_mean['LWnet_corr'] + AWS14_SEB_day_mean['Hsen'] + AWS14_SEB_day_mean['Hlat']
    AWS14_melt_day = AWS14_SEB_day_mean['melt_energy']
    Model_SEB_flight_AWS14 = np.mean(run['LW_net'][times[0]:times[1],(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1,2)) + \
                         np.mean(run['SW_net'][times[0]:times[1],(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1,2)) + \
                         np.mean(run['SH'][times[0]:times[1],(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1,2)) + \
                         np.mean(run['LH'][times[0]:times[1],(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1,2))
    Model_SEB_day_AWS14 = np.mean(run['LW_net'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1,2)) + \
                             np.mean(run['SW_net'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1,2)) + \
                          np.mean(run['SH'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1,2)) + \
                           np.mean(run['LH'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1,2))
    Model_SEB_flight_AWS15 = np.mean(run['LW_net'][times[0]:times[1],(AWS15_lon-1):(AWS15_lon+1), (AWS15_lat-1):(AWS15_lat+1)].data, axis = (1,2)) + \
                             np.mean(run['SW_net'][times[0]:times[1],(AWS15_lon-1):(AWS15_lon+1), (AWS15_lat-1):(AWS15_lat+1)].data, axis = (1,2)) + \
                             np.mean(run['SH'][times[0]:times[1],(AWS15_lon-1):(AWS15_lon+1), (AWS15_lat-1):(AWS15_lat+1)], axis = (1,2)) + \
                             np.mean(run['LH'][times[0]:times[1],(AWS15_lon-1):(AWS15_lon+1), (AWS15_lat-1):(AWS15_lat+1)], axis = (1,2))
    Model_SEB_day_AWS15 = np.mean(run['LW_net'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1,2)) + \
                             np.mean(run['SW_net'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1,2)) + \
                          np.mean(run['SH'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1,2)) + \
                           np.mean(run['LH'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1,2))
    Time = run['LW_net'].coord('time')
    # Time = Time + 0.5 # model data = hourly means, centred on the half-hour, so account for this
    Model_time = Time.units.num2date(Time.points)
    #melt_masked_day = np.ma.masked_where(RA1M_mod_SEB['Ts'] < -0.025, Model_SEB_day_AWS14)
    #melt_masked_day = melt_masked_day.clip(min=0)
    #melt_masked_flight = np.ma.masked_where(RA1M_mod_SEB['Ts']  < -0.025, Model_SEB_flight_AWS14)
    return Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15, AWS14_SEB_flight, AWS14_SEB_day,# AWS14_melt_flight, AWS14_melt_day, melt_masked_day, melt_masked_flight,

Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15,  \
obs_SEB_AWS14_flight,  obs_SEB_AWS14_day,  = calc_SEB(Cooper_SEB, times = (69,79)) #obs_melt_AWS14_flight, obs_melt_AWS14_day, melt_masked_day, melt_masked_flight,

def calc_bias(run, times, day): # times should be in tuple format, i.e. (start, end) and day should be True or False
    AWS14_bias = []
    AWS15_bias = []
    for i, j in zip([run['LW_down'].data, run['LW_up'][:,0,:,:].data, run['LW_net'].data, run['SW_down'].data,run['SW_up'][:,0,:,:].data, run['SW_net'].data, run['LH'], run['SH']],
                    ['LWin', 'LWout_corr', 'LWnet_corr','SWin_corr', 'SWout','SWnet_corr','Hlat','Hsen']):
        if day == True:
            AWS14_bias.append((np.mean(i[:,  (AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)])) - AWS14_SEB_day_mean[j])
        elif day == False:
            AWS14_bias.append((np.mean(i[times[0]:times[1], (AWS14_lon - 1):(AWS14_lon + 1), (AWS14_lat - 1):(AWS14_lat + 1)])) - AWS14_SEB_flight_mean[j])
        else:
            print('\'day\' must be set to True or False')
    if day == True:
        AWS14_bias.append(np.mean(melt_masked_day) - AWS14_SEB_day_mean['melt_energy'])
    #elif day == False:
    #    AWS14_bias.append(np.mean(melt_masked_flight) - AWS14_SEB_flight_mean['melt_energy'])
    for i, j in zip([run['LW_down'].data, run['LW_up'][:,0,:,:].data,  run['SW_down'].data, run['SW_up'][:,0,:,:].data],['Lin', 'Lout', 'Sin', 'Sout']):
        if day == True:
            AWS15_bias.append((np.mean(i[:, (AWS15_lon - 1):(AWS15_lon + 1), (AWS15_lat - 1):(AWS15_lat + 1)])) - AWS15_day_mean[j])
        elif day == False:
            AWS15_bias.append((np.mean(i[times[0]:times[1], (AWS15_lon - 1):(AWS15_lon + 1), (AWS15_lat - 1):(AWS15_lat + 1)])) - AWS15_flight_mean[j])
    return AWS14_bias, AWS15_bias

Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15, melt_masked_day, melt_masked_flight, \
obs_SEB_AWS14_flight,  obs_SEB_AWS14_day, obs_melt_AWS14_flight, obs_melt_AWS14_day = calc_SEB(Cooper_SEB, times = (69,79))

AWS14_bias, AWS15_bias = calc_bias(Cooper_SEB, times = (69,79), day = False)


# Plot liquid vs. ice for observations only
def obs_var_scatter():
    fig = plt.figure(figsize=(17, 9))
    ax = fig.add_subplot(121)
    aer, IWC_array, LWC_array, alt_array, lon_array, lon_bins, IWC_transect, LWC_transect = load_obs()
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    ax2 = plt.twiny(ax)
    sc = ax2.scatter(IWC_array, (np.array(alt_array)/1000), marker = 'x', s = 80, color = 'gray', zorder = 1, alpha = 0.6 )
    p_QCF = ax.plot(IWC_profile, altitude, color='k', linewidth=2.5, label='ice', zorder = 10)
    ax.set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize = 28, labelpad = 35)
    ax.set_ylabel('Altitude (km)', fontname='SegoeUI semibold', fontsize=28, labelpad=20)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(0, 0.0040)
    ax.set_ylim(0, 5.0)
    plt.setp(ax.get_xticklabels()[0], visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylim(0, 5.0)
    ax2.set_xlim(0, 0.0040)
    ax.xaxis.get_offset_text().set_fontsize(24)
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    ax.axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
    ax2.axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    [l.set_visible(False) for (w,l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    ax = fig.add_subplot(122)
    ax2 = plt.twiny(ax)
    sc = ax2.scatter(LWC_array, (np.array(alt_array)/1000), marker = 'x', s = 80, color = 'gray', zorder = 1, alpha = 0.6 )
    p_QCL = ax.plot(LWC_profile, altitude, color='k', linewidth=2.5, label='ice', zorder = 10)
    ax.set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize = 28, labelpad = 35)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_xlim(0, 0.40)
    ax.set_ylim(0, 5.0)
    plt.setp(ax.get_xticklabels()[0], visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax2.set_ylim(0, 5.0)
    ax2.set_xlim(0, 0.40)
    ax.xaxis.get_offset_text().set_fontsize(24)
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    ax.axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
    ax2.axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    [i.set_linewidth(2) for i in ax2.spines.itervalues()]
    plt.subplots_adjust(bottom = 0.18, wspace = 0.15, top = 0.9, left = 0.1, right = 0.95)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/f150/vertical_profile_obs_with_var.eps')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/f150/vertical_profile_obs_with_var.png')
    plt.show()

#obs_var_scatter()

def nconc_transect():
    fig = plt.figure(figsize=(18, 7))
    ax2 = fig.add_subplot(111)
    ax3 = plt.twiny(ax2)
    ax3.axis('off')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax3.spines.values(), linewidth=3, color='dimgrey')
    plt.setp(ax2.spines.values(), linewidth=3, color='dimgrey')
    ax2.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    [l.set_visible(False) for (w, l) in enumerate(ax2.yaxis.get_ticklabels()) if w % 2 != 0]
    # mean_IWC = ax.plot(lon_bins, IWC_transect2, linewidth = 2, color = '#7570b3', label = 'Mean ice')
    mean_LWC = ax2.plot(lon_bins, drop_transect1, linewidth=2, color='#1b9e77', label='Mean liquid')
    scatter_LWC = ax3.scatter(liq_lons_leg1, drop_leg1, marker='s', color='#1b9e77', label='All liquid', alpha=0.65)
    DeMott_LWC = ax2.plot(lon_bins, DeMott_vars['CDNC_transect'], lw=2, color='#EA580F', label='DeMott (2010)')
    DeMott_15_LWC = ax2.plot(lon_bins, DeMott_2015_vars['CDNC_transect'], lw=2, color = '#328B49', label = 'DeMott (2015)')
    Cooper_LWC = ax2.plot(lon_bins, Cooper_vars['CDNC_transect'], lw=2, color='#5D13E8', label='Cooper')
    mean_obs = ax2.axhline(y=np.mean(drop_leg1), linestyle='--', color='#222222', lw=2, label='Observed transect mean')
    #ax2.fill_between(lon_bins, RA1M_mod_vars['liq_5'], RA1M_mod_vars['liq_95'], facecolor='#1f78b4', alpha=0.5)
    #ax2.fill_between(lon_bins, Cooper_vars['liq_5'], Cooper_vars['liq_95'], facecolor='#EA580F', alpha=0.5)
    #ax2.fill_between(lon_bins, DeMott_vars['liq_5'], DeMott_vars['liq_95'], facecolor='#5D13E8', alpha=0.5)
    # scatter_IWC = ax4.scatter(lons_leg2, IWC_leg2, marker = 'o', color = '#7570b3', label = 'All ice')
    ax2.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax3.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax2.set_ylim(0, 350)
    ax3.set_ylim(0, 350)
    ax2.yaxis.get_offset_text().set_fontsize(24)
    ax2.yaxis.get_offset_text().set_color('dimgrey')
    plt.setp(ax2.get_yticklabels()[-1], visible=False)
    ax2.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax2.set_ylabel('Cloud droplet \nnumber \nconcentration \n(cm$^{-3}$)', rotation=0, fontname='SegoeUI semibold', color='dimgrey',
                   fontsize=28, labelpad=10)
    ax2.set_xlabel('Longitude', fontname='SegoeUI semibold', fontsize=28, color='dimgrey', labelpad=10)
    ax2.yaxis.set_label_coords(-0.22, 0.4)
    ax3.xaxis.set_visible(False)
    ax2.set_xticks([-64, -63.5, -63, -62.5])
    plt.subplots_adjust(bottom=0.15, right=0.85, left=0.24)
    lgd = plt.legend([scatter_LWC, mean_LWC[0], mean_obs,  DeMott_LWC[0], DeMott_15_LWC[0], Cooper_LWC[0]],
                     ['All observed droplets', 'Mean observed CDNC', 'Observed transect \nmean CDNC',  'DeMott (2010) CDNC', 'DeMott (2015) CDNC',
                     'Cooper CDNC'], markerscale=2, bbox_to_anchor=(0.85, 1.1),  fontsize=20)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
        lgd.get_frame().set_linewidth(0.0)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/nconc_transect_leg1_liquid.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/nconc_transect_leg1_liquid.png', transparent=True)
    #plt.show()

#nconc_transect()

def mfrac_transect():
    fig = plt.figure(figsize=(18, 7))
    ax2 = fig.add_subplot(111)
    ax3 = plt.twiny(ax2)
    ax3.axis('off')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.setp(ax3.spines.values(), linewidth=3, color='dimgrey')
    plt.setp(ax2.spines.values(), linewidth=3, color='dimgrey')
    ax2.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    [l.set_visible(False) for (w, l) in enumerate(ax2.yaxis.get_ticklabels()) if w % 2 != 0]
    #mean_IWC = ax.plot(lon_bins, IWC_transect2, linewidth = 2, color = '#7570b3', label = 'Mean ice')
    mean_LWC = ax2.plot(lon_bins, LWC_transect1, linewidth = 2, color = '#1b9e77', label = 'Mean liquid')
    scatter_LWC = ax3.scatter(liq_lons_leg1, LWC_leg1, marker = 's', color = '#1b9e77', label = 'All liquid', alpha=0.65)
    RA1M_LWC = ax2.plot(lon_bins, RA1M_mod_vars['QCL_transect'], lw = 2, color='#1f78b4', label = 'RA1M_mod')
    DeMott_LWC = ax2.plot(lon_bins, DeMott_vars['QCL_transect'], lw=2, color='#EA580F', label='DeMott')
    Cooper_LWC = ax2.plot(lon_bins, Cooper_vars['QCL_transect'], lw=2, color='#5D13E8', label='Cooper')
    mean_obs = ax2.axhline(y = np.mean(LWC_leg1), linestyle = '--', color = '#222222', lw = 2, label = 'Observed transect mean')
    ax2.fill_between(lon_bins, RA1M_mod_vars['liq_5'], RA1M_mod_vars['liq_95'], facecolor='#1f78b4', alpha = 0.5)
    ax2.fill_between(lon_bins, Cooper_vars['liq_5'], Cooper_vars['liq_95'], facecolor='#EA580F', alpha=0.5)
    ax2.fill_between(lon_bins, DeMott_vars['liq_5'], DeMott_vars['liq_95'], facecolor='#5D13E8', alpha=0.5)
    #scatter_IWC = ax4.scatter(lons_leg2, IWC_leg2, marker = 'o', color = '#7570b3', label = 'All ice')
    ax2.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax3.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax2.set_ylim(0,0.1)
    ax3.set_ylim(0,0.1)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1d'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1d'))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
    ax2.yaxis.get_offset_text().set_fontsize(24)
    ax2.yaxis.get_offset_text().set_color('dimgrey')
    #plt.setp(ax.get_yticklabels()[-2], visible=False)
    ax2.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax2.set_ylabel('Liquid mass \nfraction \n(g kg$^{-1}$)', rotation = 0, fontname='SegoeUI semibold', color = 'dimgrey', fontsize = 28, labelpad = 10)
    #ax.set_ylabel('Ice mass \nfraction \n(g kg$^{-1}$)', rotation = 0, fontname='SegoeUI semibold', color = 'dimgrey', fontsize = 28, labelpad = 10)
    ax2.set_xlabel('Longitude', fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 10)
    ax2.yaxis.set_label_coords(-0.2, 0.4)
    ax3.xaxis.set_visible(False)
    ax2.set_xticks([-64, -63.5, -63, -62.5])
    plt.subplots_adjust(bottom = 0.15, right= 0.85, left = 0.2)
    #lgd = plt.legend([ mean_obs, RA1M_LWC[0], DeMott_LWC[0], Cooper_LWC[0]],
    #                 [ 'Observed transect mean', 'RA1M_mod', 'DeMott',
    #                  'Cooper'], markerscale=2, bbox_to_anchor=(1.25, 1.1), loc='best', fontsize=20)
    lgd = plt.legend([scatter_LWC, mean_LWC[0], mean_obs, RA1M_LWC[0], DeMott_LWC[0], Cooper_LWC[0]], ['All observed liquid', 'Mean observed liquid', 'Observed transect mean', 'RA1M_mod', 'DeMott', 'Cooper'], markerscale=2, bbox_to_anchor = (1.25, 1.1), loc='best', fontsize=20)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
        lgd.get_frame().set_linewidth(0.0)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/mfrac_transect_liquid.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/mfrac_transect_liquid.png', transparent=True)
    plt.show()

#mfrac_transect()


from itertools import chain
import scipy

#model_runs = [CASIM_vars]

def mp_correl_plot():
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
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/Microphysics/mp_correlations_f150.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/Microphysics/mp_correlations_f150.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/Microphysics/mp_correlations_f150.pdf', transparent=True)
    plt.show()

#mp_correl_plot()

def temp_v_ice():
    fig = plt.figure(figsize=(20, 7))
    ax2 = fig.add_subplot(111)
    ax3 = plt.twinx(ax2)
    ax4 = plt.twiny(ax2)
    for ax in ax2, ax3:
        ax.spines['top'].set_visible(False)
    [l.set_visible(False) for (w, l) in enumerate(ax2.yaxis.get_ticklabels()) if w % 2 != 0]
    ax2.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax4.axis('off')
    plt.setp(ax3.spines.values(), linewidth=3, color='dimgrey')
    plt.setp(ax2.spines.values(), linewidth=3, color='dimgrey')
    ax2.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax3.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ice_obs = ax2.plot(lon_bins, IWC_transect1, linewidth=2, color='#1b9e77', label='Observed mean ice')
    ice_mod = ax2.plot(lon_bins, RA1M_mod_vars['QCF_transect'], linewidth=2, linestyle = '--', color='#1b9e77', label='Model mean ice')
    #scatter_IWC = ax4.scatter(ice_lons_leg1, IWC_leg1, marker='s', color='#1b9e77', label='All liquid', alpha=0.65)
    T_obs = ax3.plot(lon_bins, temp_transect, lw= 2, color = 'darkred', label = 'Air temperature')
    T_mod = ax3.plot(lon_bins, np.mean(RA1M_mod_vars['transect_box_T'][69:79, 10:25, :, :], axis=(0, 1, 2)),  linestyle = '--',color = 'darkred', label = 'Model mean temperature')
    ax2.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax3.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax2.set_ylim(0, 0.0101)
    ax3.set_ylim(-22,0)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
    ax2.yaxis.get_offset_text().set_fontsize(24)
    ax2.yaxis.get_offset_text().set_color('dimgrey')
    plt.setp(ax2.get_yticklabels()[-1], visible=False)
    ax2.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax2.set_ylabel('Cloud ice \nmass fraction \n(g kg$^{-1}$)', rotation=0, fontname='SegoeUI semibold',
                   color='dimgrey',fontsize=28, labelpad=10)
    ax2.set_xlabel('Longitude', fontname='SegoeUI semibold', fontsize=28, color='dimgrey', labelpad=10)
    ax3.set_ylabel('Air temperature\n ($^{\circ}$C)', rotation=0, fontname='SegoeUI semibold',
                   color='dimgrey',fontsize=28, labelpad=10)
    ax2.yaxis.set_label_coords(-0.2, 0.4)
    ax3.yaxis.set_label_coords(1.25, 0.6)
    ax3.xaxis.set_visible(False)
    ax2.set_xticks([-64, -63.5, -63, -62.5])
    plt.subplots_adjust(bottom=0.15, right=0.76, left=0.2)
    lgd = plt.legend([ice_obs[0], ice_mod[0], T_obs[0], T_mod[0]], ['Observed ice', 'Modelled ice', 'Observed temperature', 'Modelled temperature'], markerscale=2, bbox_to_anchor = (1.45, 1.1), loc='best', fontsize=20)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
        lgd.get_frame().set_linewidth(0.0)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/ice_v_temp_transect.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/ice_v_temp_transect.png', transparent=True)
    plt.show()



#temp_v_ice()

def SEB_correl_plot():
    fig, ax = plt.subplots(len(model_runs), 2, figsize=(18, len(model_runs * 5) + 3))  # , squeeze=False)
    ax = ax.flatten()
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l'}
    plot = 0
    var_names = ['LW$_{\downarrow}$', 'SW$_{\downarrow}$']
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set(adjustable='box-forced', aspect='equal')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    for run in model_runs:
        slope, intercept, r2, p, sterr = scipy.stats.linregress(AWS14_Jan['Sin'], np.mean(run['SW_down'][:,161:163, 182:184].data, axis = (1,2))[::4])
        if p <= 0.01:
            ax[plot].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight = 'bold', transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        else:
            ax[plot].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        ax[plot].scatter(AWS15_Jan['Sin'], np.mean(run['SW_down'][:,133:207, 182:184].data, axis = (1,2))[::4], color = '#f68080', s = 50)
        #ax[plot].set_xlim(min(chain(AWS15_Jan['Sin'], np.mean(run['SW_down'][:,199:201, 199:201].data, axis = (1,2)))), max(chain(AWS15_Jan['Sin'], np.mean(run['SW_down'][:,199:201, 199:201].data, axis = (1,2)))))
        #ax[plot].set_ylim(min(chain(AWS15_Jan['Sin'], np.mean(run['SW_down'][:,199:201, 199:201].data, axis = (1,2)))), max(chain(AWS15_Jan['Sin'], np.mean(run['SW_down'][:,199:201, 199:201].data, axis = (1,2)))))
        ax[plot].set_xlim(100,800)
        ax[plot].set_ylim(100, 800)
        ax[plot].plot(ax[plot].get_xlim(), ax[plot].get_ylim(), ls="--", c = 'k', alpha = 0.8)
        slope, intercept, r2, p, sterr = scipy.stats.linregress(AWS15_Jan['Lin'], np.mean(run['LW_down'][:,161:163, 182:184].data, axis = (1,2))[::4])
        if p <= 0.01:
            ax[plot+1].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight='bold', transform=ax[plot+1].transAxes,
                          size=24,
                          color='dimgrey')
        else:
            ax[plot+1].text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                          s='r$^{2}$ = %s' % np.round(r2, decimals=2), transform=ax[plot+1].transAxes, size=24,
                          color='dimgrey')
        ax[plot+1].scatter(AWS15_Jan['Lin'], np.mean(run['LW_down'][:,161:163, 182:184].data, axis = (1,2))[::4], color='#f68080', s=50)
        ax[plot+1].set_xlim(240, 310)
        ax[plot+1].set_ylim(240, 310)
        #ax[plot+1].set_xlim(min(chain(AWS15_Jan['Lin'], np.mean(run['LW_down'][:,199:201, 199:201].data, axis = (1,2)))), max(chain(AWS15_Jan['Lin'], np.mean(run['LW_down'][:,199:201, 199:201].data, axis = (1,2)))))
        #ax[plot+1].set_ylim(min(chain(AWS15_Jan['Lin'], np.mean(run['LW_down'][:, 199:201, 199:201].data, axis=(1, 2)))), max(chain(AWS15_Jan['Lin'], np.mean(run['LW_down'][:, 199:201, 199:201].data, axis=(1, 2)))))
        ax[plot+1].plot(ax[plot+1].get_xlim(), ax[plot+1].get_ylim(), ls="--", c='k', alpha=0.8)
         #'r$^{2}$ = %s' % r2,
        lab = ax[plot].text(0.1, 0.85, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        lab2 = ax[plot+1].text(0.1, 0.85, transform = ax[plot+1].transAxes, s=lab_dict[plot+1], fontsize=32, fontweight='bold', color='dimgrey')
        titles = ['    RA1M','    RA1M', 'RA1M_mod', 'RA1M_mod', '     fl_av','     fl_av', '    RA1T', '    RA1T', 'RA1T_mod','RA1T_mod', '   CASIM', '   CASIM']
        ax[plot].text(0.82, 1.06, transform=ax[plot].transAxes, s=titles[plot], fontsize=28,
                      color='dimgrey')
        plt.setp(ax[plot].get_yticklabels()[-1], visible=False)
        plt.setp(ax[plot].get_xticklabels()[-1], visible=False)
        plt.setp(ax[plot+1].get_xticklabels()[-1], visible=False)
        ax[plot+1].yaxis.tick_right()
        [l.set_visible(False) for (w, l) in enumerate(ax[plot + 1].yaxis.get_ticklabels()) if w % 2 != 0]
        ax[plot].yaxis.set_label_coords(-0.6, 0.5)
        ax[plot+1].yaxis.set_label_coords(1.6, 0.5)
        ax[plot].spines['right'].set_visible(False)
        ax[plot+1].spines['left'].set_visible(False)
        plot = plot + 2
        plt.subplots_adjust(top = 0.98, hspace = 0.15, bottom = 0.15, wspace = 0.17, left = 0.25, right = 0.75)
    ax[11].set_xlabel('Observed %s' % var_names[0], size=24, color='dimgrey', rotation=0, labelpad=10)
    ax[4].set_ylabel('Modelled %s' % var_names[1], size=24, color='dimgrey', rotation=0, labelpad=80)
    ax[10].set_xlabel('Observed %s' % var_names[1], size=24, color='dimgrey', rotation=0, labelpad=10)
    ax[5].set_ylabel('Modelled %s' % var_names[0], size=24, color='dimgrey', rotation=0, labelpad=80)
    plt.setp(ax[10].get_xticklabels()[-2], visible=False)
    #plt.setp(ax[5].get_xticklabels()[-2], visible=False)
    #plt.setp(ax[6].get_xticklabels()[-2], visible=False)
    #plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    #plt.setp(ax[2].get_xticklabels()[-3], visible=False)
    #plt.setp(ax[2].get_yticklabels()[-1], visible=False)
    #plt.setp(ax[5].get_yticklabels()[-2], visible=False)
    #plt.setp(ax[6].get_yticklabels()[-2], visible=False)
    #plt.setp(ax[1].get_yticklabels()[-3], visible=False)
    #plt.setp(ax[2].get_yticklabels()[-3], visible=False)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/correlations_rad_AWS15.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/correlations_rad_AWS15.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/correlations_rad_AWS15.pdf', transparent=True)
    plt.show()

def rad_time_srs():
    #model_mp = [RA1M_mod_vars, Cooper_vars, DeMott_vars]#, RA1T_vars, RA1T_mod_vars]#, CASIM_vars]#[RA1M_mod_vars]#
    model_SEB = [RA1M_mod_SEB, Cooper_SEB, DeMott_SEB]
    fig, ax = plt.subplots(len(model_SEB),2, sharex='col', figsize=(16,len(model_SEB*5)+2))#(1,2, figsize = (18,8))##, squeeze=False)#
    ax = ax.flatten()
    ax2 = np.empty_like(ax)
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        #[l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        #[l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        axs.axvspan(17.25, 19.75, edgecolor = 'dimgrey', facecolor='dimgrey', alpha=0.5)
    def my_fmt(x,p):
        return {0}.format(x) + ':00'
    plot = 0
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    for SEB in model_SEB:
        #AWS14_flight_mean, AWS14_day_mean, AWS14_Jan = load_AWS('AWS14')
        AWS14_SEB_flight_mean, AWS14_SEB_day_mean, AWS14_SEB_Jan = shift_AWS('AWS14_SEB')
        AWS15_flight_mean, AWS15_day_mean, AWS15_Jan = load_AWS('AWS15')
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/f150/')
        print('\nPLOTTING DIS BIATCH...')
        ## 1st column = downwelling SW. Dashed lines = model, solid = obs. Red = 14, Blue = 15.
        ax[plot].plot(SEB['LW_down'].coord('time').points[47:], (np.mean(SEB['SW_down'][47:, 199:201, 199:201].data, axis = (1,2))), label = 'AWS14, modelled', linewidth = 3, linestyle = '--', color = 'darkred')
        ax2[plot] = plt.twiny(ax[plot])
        ax[plot].spines['right'].set_visible(False)
        ax2[plot].axis('off')
        ax2[plot].plot(np.arange(12,24, 0.5), AWS14_SEB_Jan['SWin'][24:], label='AWS14, observed', linewidth=3, color='darkred')
        ax[plot].plot(SEB['LW_down'].coord('time').points[47:], np.mean(SEB['SW_down'][47:,161:163, 182:184].data, axis = (1,2)), label='AWS15, modelled', linewidth=3, linestyle='--',color='darkblue')
        ax2[plot].plot(AWS15_Jan['Hour'], AWS15_Jan['Sin'], label='AWS15, observed', linewidth=3, color='darkblue')
        lab = ax[plot].text(0.1, 0.85, transform=ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold',color='dimgrey')
        #plt.setp(ax[plot].get_yticklabels()[-1], visible=False)
        #ax[plot].set_xlim(12, 23)
        ax[plot].set_ylim(0,850)
        ax2[plot].set_ylim(0,850)
        ax[plot].set_yticks([0,400, 800])
        ax2[plot].set_xlim(AWS15_Jan['Hour'].values[12], AWS15_Jan['Hour'].values[-1]) ##
        ax[plot].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        ## 2nd column = downwelling LW. As above.
        ax2[plot+1] = plt.twiny(ax[plot+1])
        #ax[plot+1].set_xlim(12, 23)
        ax2[plot+1].set_xlim(AWS15_Jan['Hour'].values[12], AWS15_Jan['Hour'].values[-1])
        ax[plot+1].set_ylim(220,320)
        ax2[plot+1].set_ylim(220,320)
        ax[plot+1].set_yticks([250,300])
        ax2[plot + 1].axis('off')
        ax[plot+1].yaxis.set_label_position("right")
        ax[plot + 1].spines['left'].set_visible(False)
        ax[plot+1].yaxis.tick_right()
        ax[plot+1].plot(SEB['LW_down'].coord('time').points[47:], np.mean(SEB['LW_down'][47:,199:201, 199:201].data, axis = (1,2)),  label = 'AWS14, modelled', linewidth = 3, linestyle = '--', color = 'darkred') ##
        ax[plot+1].plot(SEB['LW_down'].coord('time').points[47:],np.mean(SEB['LW_down'][47:,161:163, 182:184].data, axis = (1,2)), label = 'AWS15, modelled', linewidth = 3, linestyle = '--', color = 'darkblue') ##
        obs14 = ax2[plot+1].plot(np.arange(12,24, 0.5),  AWS14_SEB_Jan['LWin'][24:], label='AWS14, observed', linewidth=3, color='darkred') ##
        obs15 = ax2[plot+1].plot(AWS15_Jan['Hour'], AWS15_Jan['Lin'], label='AWS15, observed', linewidth=3, color='darkblue') ##
        lab = ax[plot+1].text(0.1, 0.85, transform=ax[plot+1].transAxes, s=lab_dict[plot+1], fontsize=32, fontweight='bold',color='dimgrey')
        ax[plot+1].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        #[l.set_visible(False) for (i, l) in enumerate(ax[plot+1].yaxis.get_ticklabels()) if i % 2 != 0]
        #[l.set_visible(False) for (i, l) in enumerate(ax[plot + 1].xaxis.get_ticklabels()) if i % 2 != 0]
        [w.set_linewidth(2) for w in ax[plot].spines.itervalues()]
        [w.set_linewidth(2) for w in ax[plot+1].spines.itervalues()]
        #ax[plot+1].set_xlim(run['LW_down'].coord('time').points[1], run['LW_down'].coord('time').points[-1])
        #ax2[plot+1].set_xlim(AWS15_Jan['Hour'].values[0], AWS15_Jan['Hour'].values[-1]) ##
        plt.setp(ax2[plot].get_xticklabels(), visible=False)
        plt.setp(ax2[plot+1].get_xticklabels(), visible=False)
        titles = ['RA1M_mod', 'RA1M_mod', '   Cooper', '   Cooper', '   DeMott','   DeMott']# '   CASIM', '   CASIM']
        ax[plot].text(0.83, 1.05, transform=ax[plot].transAxes, s=titles[plot], fontsize=28, color='dimgrey')
        print('\nDONE!')
        print('\nNEEEEEXT')
        plot = plot + 2
    # ax[plot+1].set_xlim(12,23)
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d:00"))
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d:00"))
    lns = [Line2D([0], [0], color='darkred', linewidth=3),
           Line2D([0], [0], color='darkred', linestyle = '--', linewidth=3),
           Line2D([0], [0], color='darkblue', linewidth=3),
           Line2D([0], [0], color='darkblue', linestyle = '--', linewidth=3)]
    labs = ['AWS 14, observed', 'AWS 14, modelled','AWS 15, observed', 'AWS 15, modelled']#  '                      ','                      '
    lgd = plt.legend(lns, labs, ncol=2, bbox_to_anchor=(0.9, -0.2), borderaxespad=0., loc='best', prop={'size': 24})
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    fig.text(0.5, 0.04, 'Time (hours)', fontsize=24, fontweight = 'bold', ha = 'center', va = 'center', color = 'dimgrey')
    fig.text(0.1, 0.55, 'Downwelling \nshortwave \nflux \n(W m$^{-2}$)', fontsize=30, ha= 'center', va='center', rotation = 0, color = 'dimgrey')
    fig.text(0.92, 0.55, 'Downwelling \nlongwave \nflux \n(W m$^{-2}$)', fontsize=30, ha='center', va='center', color = 'dimgrey', rotation=0)
    plt.subplots_adjust(left=0.25, bottom=0.2, right=0.78, top=0.97, wspace=0.15, hspace=0.2)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/vn11_SEB_time_srs_LWd_SWd_f150_shifted_RA1M.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/vn11_SEB_time_srs_LWd_SWd_f150_shifted_RA1M.eps', transparent = True)
    plt.show()

rad_time_srs()


#SEB_correl_plot()

#Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15, melt_masked_day, melt_masked_flight, \
#obs_SEB_AWS14_flight,  obs_SEB_AWS14_day, obs_melt_AWS14_flight, obs_melt_AWS14_day = calc_SEB(RA1T_vars, times = (47,95))

def SEB_diff(run):
    fig, axs = plt.subplots(1, 1, figsize=(22, 8), frameon=False)
    hrs = mdates.HourLocator(interval=2)
    hrfmt = mdates.DateFormatter('%H:%M')
    plt.setp(axs.spines.values(), linewidth=2, color='dimgrey')
    axs.set_ylabel('Energy flux \n (W m$^{-2}$)', rotation=0, fontsize=44, labelpad=70, color='dimgrey')
    [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
    [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    axs.set_ylim(-100, 150)
    # Plot differences
    ax3 = axs.twinx()
    ax3.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.set_ylim(-100, 150)
    ax3.fill_between(x=AWS14_SEB_Jan['Time'][24:], y1=AWS14_SEB_Jan['SWnet_corr'][24:], y2=np.mean(
        run['SW_net'][48:, (AWS14_lon - 1):(AWS14_lon + 1), (AWS14_lat - 1):(AWS14_lat + 1)].data, axis=(1, 2))[0::2],
                     where=(np.mean(
                         run['SW_net'][48:, (AWS14_lon - 1):(AWS14_lon + 1), (AWS14_lat - 1):(AWS14_lat + 1)].data,
                         axis=(1, 2)))[0::2] > AWS14_SEB_Jan['SWnet_corr'][24:], facecolor='#375869', interpolate=True,
                     zorder=2)
    ax3.fill_between(x=AWS14_SEB_Jan['Time'][24:], y1=AWS14_SEB_Jan['SWnet_corr'][24:], y2=np.mean(
        run['SW_net'][48:, (AWS14_lon - 1):(AWS14_lon + 1), (AWS14_lat - 1):(AWS14_lat + 1)].data, axis=(1, 2))[0::2],
                     where=np.mean(
                         run['SW_net'][48:, (AWS14_lon - 1):(AWS14_lon + 1), (AWS14_lat - 1):(AWS14_lat + 1)].data,
                         axis=(1, 2))[0::2] < AWS14_SEB_Jan['SWnet_corr'][24:], facecolor='#B7D7E8', interpolate=True,
                     zorder=2)
    ax3.fill_between(x=AWS14_SEB_Jan['Time'][24:], y1=AWS14_SEB_Jan['LWnet_corr'][24:], y2=np.mean(
        run['LW_net'][48:, (AWS14_lon - 1):(AWS14_lon + 1), (AWS14_lat - 1):(AWS14_lat + 1)].data, axis=(1, 2))[0::2],
                     where=(np.mean(
                         run['LW_net'][48:, (AWS14_lon - 1):(AWS14_lon + 1), (AWS14_lat - 1):(AWS14_lat + 1)].data,
                         axis=(1, 2)))[0::2] > AWS14_SEB_Jan['LWnet_corr'][24:], facecolor='#50673b', interpolate=True,
                     zorder=2)
    ax3.fill_between(x=AWS14_SEB_Jan['Time'][24:], y1=AWS14_SEB_Jan['LWnet_corr'][24:], y2=np.mean(
        run['LW_net'][48:, (AWS14_lon - 1):(AWS14_lon + 1), (AWS14_lat - 1):(AWS14_lat + 1)].data, axis=(1, 2))[0::2],
                     where=np.mean(
                         run['LW_net'][48:, (AWS14_lon - 1):(AWS14_lon + 1), (AWS14_lat - 1):(AWS14_lat + 1)].data,
                         axis=(1, 2))[0::2] < AWS14_SEB_Jan['LWnet_corr'][24:], facecolor='#b6cda1', interpolate=True,
                     zorder=2)
    ax3.fill_between(x=AWS14_SEB_Jan['Time'][24:], y1=AWS14_SEB_Jan['melt_energy'][24:], y2=melt_masked_day[48:][0::2],
                     where=melt_masked_day[48:][0::2] > AWS14_SEB_Jan['melt_energy'][24:], facecolor='#934c4c', interpolate=True,
                     zorder=2)
    ax3.fill_between(x=AWS14_SEB_Jan['Time'][24:], y1=AWS14_SEB_Jan['melt_energy'][24:], y2=melt_masked_day[48:][0::2],
                     where=melt_masked_day[48:][0::2] < AWS14_SEB_Jan['melt_energy'][24:], facecolor='#f9b2b2', interpolate=True,
                     zorder=2)
    # Plot observed SEB
    axs.plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['SWnet_corr'][24:], color='#6fb0d2', lw=5, label='Net shortwave flux', zorder = 9)
    axs.plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['LWnet_corr'][24:], color='#86ad63', lw=5, label='Net longwave flux', zorder = 9)
    #axs.plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['Hsen'][24:], color='#1f78b4', lw=5, label='Sensible heat flux')
    #axs.plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['Hlat'][24:], color='#33a02c', lw=5, label='Latent heat flux')
    axs.plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['melt_energy'][24:], color='#f68080', lw=5, label='Melt flux')
    axs.set_xlim(AWS14_SEB_Jan['Time'][840], max(AWS14_SEB_Jan['Time']))
    axs.axvspan(18.614583, 18.7083, edgecolor='dimgrey', facecolor='dimgrey', alpha=0.5)
    # Plot model SEB
    ax = axs.twiny()
    ax.tick_params(axis='both', which='both', labelsize=44, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=50)
    ax.axes.get_xaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    Model_time = run['SW_net'].coord('time').units.num2date(run['SW_net'].coord('time').points)
    ax.set_xlim(Model_time[48], Model_time[-1])
    ax.plot(Model_time[48:], np.mean(run['SW_net'][48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1, 2)), linestyle = '--',color='#6fb0d2', lw=5, label='Net shortwave flux', zorder = 10)
    ax.plot(Model_time[48:], np.mean(run['LW_net'][48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1, 2)), linestyle = '--',color='#86ad63', lw=5, label='Net longwave flux', zorder = 10)
    #ax.plot(Model_time[48:], np.mean(run['SH'][48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1, 2)), color='#1f78b4', linestyle = '--',lw=5, label='Sensible heat flux')
    #ax.plot(Model_time[48:], np.mean(run['LH'][48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1, 2)), color='#33a02c',linestyle = '--', lw=5, label='Latent heat flux')
    ax.plot(Model_time[48:], melt_masked_day[48:], color='#f68080', lw=5, linestyle = '--',label='Melt flux')
    lgd = plt.legend(fontsize=36, bbox_to_anchor = (1.15, 1.15))
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    #plt.setp(axs.get_yticklabels()[-2], visible=False)
    plt.subplots_adjust(left=0.22, top = 0.92, bottom=0.1, right=0.9, hspace = 0.1, wspace = 0.1)
    for axes in [axs, ax, ax3]:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        axes.xaxis.set_major_formatter(hrfmt)
        axes.xaxis.set_major_locator(hrs)
        axes.tick_params(axis='both', which='both', labelsize=44, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axes.axhline(y=0, xmin=0, xmax=1, linestyle='--', linewidth=2)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_SEB_difs_RA1T.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_SEB_difs_RA1T.png', transparent=True)
    plt.show()


#SEB_diff(RA1T_vars)
