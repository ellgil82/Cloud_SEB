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


# Load model data
def load_model(config, flight_date, times): #times should be a range in the format 11,21
    pa = []
    pb = []
    pf = []
    print('\nimporting data from %(config)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/f150'):
            if fnmatch.fnmatch(file, flight_date + '*%(config)s_pb*' % locals()):
                pb.append(file)
            elif fnmatch.fnmatch(file, flight_date + '*%(config)s_pa*' % locals()):
                pa.append(file)
            elif fnmatch.fnmatch(file, flight_date) & fnmatch.fnmatch(file, '*%(config)s_pf*' % locals()):
                pf.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/f150/')
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
    ## ---------------------------------------- CREATE MODEL VERTICAL PROFILES ------------------------------------------ ##
    # Create mean vertical profiles for region of interest
    # region of interest = ice shelf. Longitudes of ice shelf along transect =
    # OR: region of interest = only where aircraft was sampling layer cloud: time 53500 to 62000 = 14:50 to 17:00
    # Define box: -62 to -61 W, -66.9 to -68 S
    # Coord: lon = 188:213, lat = 133:207, time = 4:6 (mean of preceding hours)
    print('\ncreating transects geez...')
    box_QCF = ice_mass_frac[times[0]:times[1], :40, 115:150, 130:185].data
    box_QCL = liq_mass_frac[times[0]:times[1], :40, 115:150, 130:185].data
    transect_box_QCF = ice_mass_frac[times[0]:times[1], :40, 127:137, 130:185].data
    transect_box_QCL = liq_mass_frac[times[0]:times[1], :40, 127:137, 130:185].data
    box_mean_IWP = np.mean(IWP[times[0]:times[1], 115:150, 130:185].data)#, axis = (0,1,2))
    box_mean_LWP = np.mean(LWP[times[0]:times[1], 115:150, 130:185].data)#, axis =(0,1,2))
    QCF_transect = np.mean(ice_mass_frac[:,:,127:137, 130:185 ].data, axis = (0,1,2))
    QCL_transect = np.mean(liq_mass_frac[:, :, 127:137, 130:185 ].data, axis=(0, 1, 2))
    IWP_transect = np.mean(IWP[:, 127:137, 130:185 ].data, axis = (0,1))
    LWP_transect = np.mean(LWP[:, 127:137, 130:185  ].data, axis = (0,1))
    QCL_profile = np.mean(ice_mass_frac[:,:,127:137, 130:185].data, axis = (0,2,3))
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
    var_dict = {'real_lon': real_lon, 'real_lat':real_lat,   'lsm': lsm, 'orog': orog,  'IWP': IWP, 'LWP':LWP, 'ice_5': ice_5,
                'ice_95': ice_95, 'liq_5': liq_5, 'liq_95': liq_95, 'box_QCF': box_QCF, 'box_QCL': box_QCL, 'vert_5': vert_5,
                 'vert_95': vert_95, 'LWP_transect': LWP_transect,'IWP_transect': IWP_transect, 'QCL_profile': QCL_profile,
                'QCF_transect': QCF_transect, 'QCL_transect': QCL_transect, 'QCF': ice_mass_frac, 'QCL': liq_mass_frac}
    return  var_dict

# Load models in for times of interest: (59, 68) for time of flight, (47, 95) for midday-midnight (discard first 12 hours as spin-up)
#HM_vars = load_model('Hallett_Mossop', (11,21))
RA1M_mod_vars = load_model(config = 'RA1M_mods_f150', flight_date = '20110115T1200', times = (0,11))
#RA1T_mod_vars = load_model('RA1T_mod_24',(59,68))
#RA1T_vars = load_model('RA1T_24', (59,68)) # 1 hr means
#RA1M_vars = load_model('RA1M_24', (59,68))
Cooper_vars = load_model(config = 'Cooper', flight_date = '20110115T0000', times = (59,68))
DeMott_vars = load_model(config = 'DeMott', flight_date = '20110115T0000',  times = (59,68))
#fl_av_vars = load_model('fl_av')
#model_runs = [RA1M_vars, RA1M_mod_vars,RA1T_vars, RA1T_mod_vars]#, CASIM_vars fl_av_vars, ]

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
        elif n_drop[i] > 1.0:  # same threshold as in Lachlan-Cope et al. (2016)
            drop_array.append(n_drop[i])
            LWC_array.append(LWC_cas[i])
            alt_array_liq.append(plane_alt[i])
            lon_array_liq.append(plane_lon[i])
    ## Create longitudinal transects
    # Calculate mean values at each height in the model
    # Create bins from model data
    print('\nbinning by altitude...')
    #Load model data to get altitude/longitude bins
    lon_bins = RA1M_mod_vars['QCF'].coord('longitude')[130:185].points.tolist()
    alt_bins = RA1M_mod_vars['QCF'].coord('level_height').points.tolist()
    # Find index of model longitude bin to which aircraft data would belong
    # Turn data into pandas dataframe
    d_liq = {'LWC': LWC_array, 'lon_idx': np.digitize(lon_array_liq, bins=lon_bins), 'alt_idx': np.digitize(alt_array_liq, bins=alt_bins), 'alt': alt_array_liq, 'lons': lon_array_liq,  'n_drop': drop_array}
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
    LWC_leg1 = grouped_liq['LWC'][:70] # outward leg
    LWC_leg2 = grouped_liq['LWC'][70:]
    nice_leg1 = grouped_ice['n_ice'][:70]
    nice_leg2 = grouped_ice['n_ice'][70:]
    drop_leg1 = grouped_liq['n_drop'][:70]
    drop_leg2 = grouped_liq['n_drop'][70:]
    ice_lons_leg1 = grouped_ice[:70]['lons']
    ice_lons_leg2 = grouped_ice[70:]['lons']
    liq_lons_leg1 = grouped_liq[:70]['lons']
    liq_lons_leg2 = grouped_liq[70:]['lons']
    # Calculate means of non-zero points for each variable for the transects
    IWC_transect1 = grouped_ice[:70].groupby(['lon_idx']).mean()['IWC'] #25, 31,32, 35, 59, 60
    LWC_transect1 = grouped_liq[:520].groupby(['lon_idx']).mean()['LWC']
    ice_transect1 = grouped_ice[:70].groupby(['lon_idx']).mean()['n_ice']
    drop_transect1 = grouped_liq[:520].groupby(['lon_idx']).mean()['n_drop']
    IWC_transect2 = grouped_ice[70:].groupby(['lon_idx']).mean()['IWC'] #25, 31,32, 35, 59, 60
    LWC_transect2 = grouped_liq[520:].groupby(['lon_idx']).mean()['LWC']
    ice_transect2 = grouped_ice[70:].groupby(['lon_idx']).mean()['n_ice']
    drop_transect2 = grouped_liq[520:].groupby(['lon_idx']).mean()['n_drop']
        # Add in some zeros at correct place to make mean transect the right shape for plotting
    def append_1(transect):
        transect = np.append(np.zeros(9), transect)
        transect = np.append(transect, np.zeros(10))# [24, 30,31,34], [0])
        transect = np.insert(transect,[25],[0,0,0,0])
        transect = np.insert(transect, [31], [0,0,0,0,0,0,0])
        return transect
    LWC_transect2, drop_transect2 = np.append(LWC_transect2, np.zeros(9)), np.append(drop_transect2, np.zeros(9))
    LWC_transect1, drop_transect1 = append_1(LWC_transect1), append_1(drop_transect1)
    ## Create vertical profiles
    grouped_liq = df_liq.set_index(['alt_idx'])
    LWC_profile = grouped_liq[:520].groupby(['alt_idx']).mean()['LWC']
    LWC_profile = np.append(np.zeros(10), LWC_profile)
    LWC_profile = np.insert(LWC_profile, [21], [0,0,0,0,0])
    return aer, IWC_array, LWC_array, lon_bins, IWC_transect1, LWC_transect1, drop_transect1, \
           ice_transect1, nice_leg1, drop_leg1, ice_lons_leg1, ice_lons_leg2, IWC_leg1, LWC_leg1, IWC_transect2, \
           LWC_transect2, drop_transect2, ice_transect2, nice_leg2, drop_leg2,liq_lons_leg1, liq_lons_leg2, IWC_leg2, LWC_leg2, LWC_profile

aer, IWC_array, LWC_array,lon_bins, IWC_transect1, LWC_transect1, drop_transect1, ice_transect1, nice_leg1, drop_leg1, ice_lons_leg1, ice_lons_leg2, IWC_leg1, \
LWC_leg1, IWC_transect2, LWC_transect2, drop_transect2, ice_transect2, nice_leg2, drop_leg2,liq_lons_leg1, liq_lons_leg2, IWC_leg2, LWC_leg2, LWC_profile = load_obs()


print 'Drop array means:\n\n Leg 1:'
print np.mean(drop_leg1)
print 'Drop array means:\n\n Leg 2:'
print np.mean(drop_leg2)

print 'Drop transect means:\n\n Leg 1:'
print np.mean(drop_transect1)
print 'Drop transect means:\n\n Leg 2:'
print np.mean(drop_transect2)

print 'Ice array means:\n\n Leg 1:'
print np.mean(nice_leg1)
print 'Ice array means:\n\n Leg 2:'
print np.mean(nice_leg2)

print 'Ice transect means:\n\n Leg 1:'
print np.mean(ice_transect1)
print 'Ice transect means:\n\n Leg 2:'
print np.mean(ice_transect2)

altitude = np.arange(0,3000)

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
    ax = fig.add_subplot(111)
    ax2 = plt.twinx(ax)
    ax3 = plt.twiny(ax2)
    ax4 = plt.twiny(ax)
    ax4.axis('off')
    ax3.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
    plt.setp(ax2.spines.values(), linewidth=3, color='dimgrey')
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    #[l.set_visible(False) for (w, l) in enumerate(ax2.yaxis.get_ticklabels()) if w % 2 != 0]
    #mean_IWC = ax.plot(lon_bins, ice_transect2, linewidth = 2, color = '#7570b3', label = 'Mean ice')
    mean_LWC = ax2.plot(lon_bins, drop_transect1, linewidth = 2, color = '#1b9e77', label = 'Mean droplet')
    scatter_LWC = ax3.scatter(liq_lons_leg1, drop_leg1, marker = 's', color = '#1b9e77', label = 'All droplets', alpha=0.65)
    #scatter_IWC = ax4.scatter(lons_leg2, nice_leg2, marker = 'o', color = '#7570b3', label = 'All ice')
    ax.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax2.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax3.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax4.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax.set_ylim(0, 0.0008)
    ax4.set_ylim(0, 0.0008)
    ax2.set_ylim(0,500.001)
    ax3.set_ylim(0,500.001)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-3, 2))
    ax2.yaxis.get_offset_text().set_fontsize(24)
    ax2.yaxis.get_offset_text().set_color('dimgrey')
    ax.yaxis.get_offset_text().set_color('dimgrey')
    plt.setp(ax.get_yticklabels()[-2], visible=False)
    ax2.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax2.set_ylabel('Liquid droplet \nnumber \nconcentration \n(cm$^{-3}$)', rotation = 0, fontname='SegoeUI semibold', color = 'dimgrey', fontsize = 28, labelpad = 10)
    #ax.set_ylabel('Ice particle \nnumber \nconcentration \n(cm$^{-3}$)', rotation = 0, fontname='SegoeUI semibold', color = 'dimgrey', fontsize = 28, labelpad = 10)
    ax.set_xlabel('Longitude', fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 10)
    ax2.yaxis.set_label_coords(1.22, 0.6)
    ax.yaxis.set_label_coords(-0.17, 0.28)
    ax3.xaxis.set_visible(False)
    ax4.xaxis.set_visible(False)
    plt.subplots_adjust(bottom = 0.15, right= 0.77, left = 0.18)
    lgd = plt.legend()#[mean_IWC[0], mean_LWC[0], scatter_IWC, scatter_LWC], ['Mean ice', 'Mean droplets', 'All ice', 'All droplets'], markerscale=2, bbox_to_anchor = (1.4, 1.1), loc='best', fontsize=24)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
        lgd.get_frame().set_linewidth(0.0)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/nconc_transect_leg1_liquid.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/nconc_transect_leg1_liquid.png', transparent=True)
    plt.show()

nconc_transect()

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
    #[l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    #[l.set_visible(False) for (w, l) in enumerate(ax2.yaxis.get_ticklabels()) if w % 2 != 0]
    #mean_IWC = ax.plot(lon_bins, IWC_transect2, linewidth = 2, color = '#7570b3', label = 'Mean ice')
    mean_LWC = ax2.plot(lon_bins, LWC_transect1, linewidth = 2, color = '#1b9e77', label = 'Mean liquid')
    scatter_LWC = ax3.scatter(liq_lons_leg1, LWC_leg1, marker = 's', color = '#1b9e77', label = 'All liquid', alpha=0.65)
    model_LWC = ax2.plot(lon_bins, RA1M_mod_vars['QCL_transect'], lw = 2, color = '#fb9a99', label = 'Modelled liquid')
    ax2.axhline(y = np.mean(LWC_transect1), linestyle = '--', lw = 1, label = 'Mean observed liquid')
    ax2.fill_between(lon_bins, RA1M_mod_vars['liq_5'], RA1M_mod_vars['liq_95'], facecolor='#fb9a99', alpha = 0.5)
    #scatter_IWC = ax4.scatter(lons_leg2, IWC_leg2, marker = 'o', color = '#7570b3', label = 'All ice')
    ax2.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax3.set_xlim(np.min(lon_bins), np.max(lon_bins))
    ax2.set_ylim(0,0.1)
    ax3.set_ylim(0,0.1)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1g'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1g'))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 2))
    ax2.yaxis.get_offset_text().set_fontsize(24)
    ax2.yaxis.get_offset_text().set_color('dimgrey')
    #plt.setp(ax.get_yticklabels()[-2], visible=False)
    ax2.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax2.set_ylabel('Liquid mass \nfraction \n(g kg$^{-1}$)', rotation = 0, fontname='SegoeUI semibold', color = 'dimgrey', fontsize = 28, labelpad = 10)
    #ax.set_ylabel('Ice mass \nfraction \n(g kg$^{-1}$)', rotation = 0, fontname='SegoeUI semibold', color = 'dimgrey', fontsize = 28, labelpad = 10)
    ax2.set_xlabel('Longitude', fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 10)
    ax2.yaxis.set_label_coords(1.22, 0.6)
    #ax.yaxis.set_label_coords(-0.15, 0.4)
    ax3.xaxis.set_visible(False)
    plt.subplots_adjust(bottom = 0.15, right= 0.73, left = 0.15)
    lgd = plt.legend()#[mean_IWC[0], mean_LWC[0], scatter_IWC, scatter_LWC], ['Mean ice', 'Mean droplets', 'All ice', 'All droplets'], markerscale=2, bbox_to_anchor = (1.47, 1.1), loc='best', fontsize=24)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
        lgd.get_frame().set_linewidth(0.0)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/mfrac_transect_leg1_liquid.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/mfrac_transect_leg1_liquid.png', transparent=True)
    plt.show()

mfrac_transect()



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
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/Microphysics/correlations_f150.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/Microphysics/correlations_f150.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f150/Microphysics/correlations_f150.pdf', transparent=True)
    plt.show()

#correl_plot()
