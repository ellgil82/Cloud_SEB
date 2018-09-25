## ------------------------------------------------ CREATE MEAN VERTICAL PROFILES OF ALL MODEL RUNS VS. OBSERVATIONS ------------------------------------------------------ ##
# File management: make sure all model runs are in one containing folder. Presently, this is /data/mac/ellgil82/cloud_data/um/six_hrly

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

os.chdir('/data/mac/ellgil82/cloud_data/um/means')

# Load model data
def load_model(var):
    pc = []
    microphys = []
    print('\nimporting data from %(var)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/means'):
            if fnmatch.fnmatch(file, '*%(var)s_pc*.pp' % locals()):
                pc.append(file)
            elif fnmatch.fnmatch(file, '*%(var)s_pg*.pp' % locals()):
                microphys.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/means')
    cubes = iris.load(microphys)
    cl_drop = cubes[2]
    cl_drop = cl_drop/1000000 # convert to cm-3
    ice_mass_frac = iris.load_cube(pc, 'mass_fraction_of_cloud_ice_in_air')
    liq_mass_frac = iris.load_cube(pc, 'mass_fraction_of_cloud_liquid_water_in_air')
    spec_hum = iris.load_cube(pc, 'specific_humidity')
    Var_4D = [ice_mass_frac, liq_mass_frac, spec_hum, cl_drop]
    ## set up parameters for rotated projection
    pole_lon = ice_mass_frac.coord('grid_longitude').coord_system.grid_north_pole_longitude
    pole_lat = ice_mass_frac.coord('grid_latitude').coord_system.grid_north_pole_latitude
    ## Rotate projection
    #create numpy arrays of coordinates
    rotated_lat = ice_mass_frac.coord('grid_latitude').points
    rotated_lon = ice_mass_frac.coord('grid_longitude').points
    # Perform rotation
    real_lon, real_lat = iris.analysis.cartography.unrotate_pole(rotated_lon, rotated_lat, pole_lon, pole_lat)
    # Create new lat/lon variables
    print ('\nunrotating pole, innit...')
    model_lat = ice_mass_frac.coord('grid_latitude')
    model_lon = ice_mass_frac.coord('grid_longitude')
    New_lat = iris.coords.DimCoord(real_lat, standard_name='latitude',long_name="grid_latitude",var_name="lat",units=model_lat.units)
    New_lon= iris.coords.DimCoord(real_lon, standard_name='longitude',long_name="grid_longitude",var_name="lon",units=model_lon.units)
    # Remove old lat/lons, add new ones
    for var in Var_4D:
        var.remove_coord('grid_latitude')
        var.add_dim_coord(New_lat, data_dim=2)
        var.remove_coord('grid_longitude')
        var.add_dim_coord(New_lon, data_dim=3)
    # Convert model data to g kg-1
    ice_mass_frac = ice_mass_frac * 1000
    liq_mass_frac = liq_mass_frac * 1000
    ## ---------------------------------------- CREATE MODEL VERTICAL PROFILES ------------------------------------------ ##
    # Create mean vertical profiles for region of interest
    # region of interest = ice shelf. Longitudes of ice shelf along transect =
    # OR: region of interest = only where aircraft was sampling layer cloud: time 53500 to 62000 = 14:50 to 17:00
    # Define box: -62 to -61 W, -66.9 to -68 S
    # Coord: lon = 188:213, lat = 133:207, time = 4:6 (mean of preceding hours)
    print('\ncreating vertical profiles geez...')
    box_QCF = ice_mass_frac[4:6,:,133:207,188:213].data
    box_QCL = liq_mass_frac[4:6,:,133:207,188:213].data
    box_q = spec_hum[4:6,:,133:207,188:213].data
    box_cl_drop = cl_drop[4:6,:,133:207,188:213].data
    mean_q = np.mean(box_q, axis=(0,2,3))
    mean_QCF = np.mean(box_QCF, axis=(0,2,3))
    mean_QCL = np.mean(box_QCL, axis=(0,2,3))
    mean_drop = np.mean(box_cl_drop, axis=(0,2,3))
    AWS14_mean_QCF = np.mean(ice_mass_frac[4:6, :, 199:201, 199:201].data, axis =(0,2,3))
    AWS14_mean_QCL = np.mean(liq_mass_frac[4:6, :, 199:201, 199:201].data, axis =(0,2,3))
    AWS15_mean_QCF = np.mean(ice_mass_frac[4:6, :, 161:163, 182:184].data, axis =(0,2,3)) # AWS 15 = 161:163, but trying for now to get out of this cloud band
    AWS15_mean_QCL = np.mean(liq_mass_frac[4:6, :, 161:163, 182:184].data, axis =(0,2,3))
    # Find max and min values at each model level
    #time_mean_QCF = np.mean(box_QCF, axis=0)
    array = pd.DataFrame()
    for each_lat in np.arange(74):
        for each_lon in np.arange(25):
            for each_time in np.arange(2):
                m = pd.DataFrame(box_QCF[each_time,:,each_lat,each_lon])
                array = pd.concat([m, array], axis=1)
    max_QCF = array.max(axis=1)
    min_QCF = array.min(axis=1)
    std_QCF = array.std(axis=1)
    mean_ice = array.mean(axis=0) # for PDF
    # Calculate 95th percentile
    ice_95 = np.percentile(array, 95,axis=1)
    ice_5 = np.percentile(array, 5,axis=1)
    # Find max and min values at each model level
    #time_mean_QCL = np.mean(box_QCL, axis=0)
    array = pd.DataFrame()
    for each_lat in np.arange(74):
        for each_lon in np.arange(25):
            for each_time in np.arange(2):
                m = pd.DataFrame(box_QCL[each_time,:,each_lat,each_lon])
                array = pd.concat([m, array], axis=1)
    max_QCL = array.max(axis=1)
    min_QCL = array.min(axis=1)
    std_QCL = array.std(axis=1)
    mean_liq = array.mean(axis=0) # for PDF
    # Calculate 95th percentile
    liq_95 = np.percentile(array, 95,axis=1)
    liq_5 = np.percentile(array, 5,axis=1)
    # Calculate PDF of ice and liquid water contents
    #liq_PDF = mean_liq.plot.density(color = 'k', linewidth = 1.5)
    #ice_PDF = mean_ice.plot.density(linestyle = '--', linewidth=1.5, color='k')
    altitude = ice_mass_frac.coord('level_height').points / 1000
    return mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, mean_drop

def print_stats():
    model_mean = pd.DataFrame()
    for run in ['Smith', 'Smith_tnuc', 'PC2', 'PC2_tnuc']:
        mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, mean_drop = load_model(run)
        print('\n\nMean cloud box QCL of %(run)s is: '% locals()+str(np.mean(mean_QCL)) )
        print('\n\nMean cloud box QCF of %(run)s is: '% locals()+str(np.mean(mean_QCF)) )
        print('\n\nMean QCL at AWS 14 and 15 is ' + str(np.mean(AWS14_mean_QCL))+ ' and ' + str(np.mean(AWS15_mean_QCL)) + ', respectively in %(run)s' % locals())
        print ('\n\nMean QCF at AWS 14 and 15 is '+str(np.mean(AWS14_mean_QCF))+' and '+str( np.mean(AWS15_mean_QCF))+', respectively in %(run)s' % locals())
        m = pd.DataFrame({'mean QCL': np.mean(mean_QCL), 'mean_QCF': np.mean(mean_QCF), 'AWS 14 QCL': np.mean(AWS14_mean_QCL), 'AWS 15 QCL': np.mean(AWS15_mean_QCL), 'AWS 14 QCF' : np.mean(AWS14_mean_QCF), 'AWS 15 QCF' : np.mean(AWS15_mean_QCF)}, index = [0])
        model_mean = pd.concat([model_mean, m])
    means = model_mean.mean(axis=0)
    print means

#print_stats()

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

#IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs()

## ---------------------------------------------------- PLOTTING ---------------------------------------------------- ##

## Caption: Mean vertical profiles of ice (left, panels a, c, e, g and i) and liquid (right, panels b, d, f, h and j) mass fraction within the 'cloud' box. Panels a) and b) show
## observed profiles, while each following row corresponds to one model run, such that c) and d) are profiles of ice and liquid mass fraction for 'Smith', respectively; e) and f)
## are for 'Smith_tnuc'; g) and h) are for 'PC2'; and i) and j) are for 'PC2_tnuc'. In panels c) to j), the thick black line indicates the mean of the entire 'cloud' box, the shaded
## area shows the full range of values for every individual grid cell in the 'cloud' box, and the dashed lines show the 5th and 95th percentiles of profiles for all grid cells within
## the box. Note that: firstly, the scale of the x axes for the panels showing liquid mass fraction are an order of magnitude higher than those for panels showing ice mass fraction,
## secondly, all model configurations overestimate ice mass fraction and underestimate liquid mass fraction by around an order of magnitude compared with observations, and lastly,
## the full range of modelled ice mass fractions is not shown because the maximum is so far above the 95th percentile that it would make the detail invisible.

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans','Verdana']

model_runs = ['Smith', 'Smith_tnuc', 'PC2', 'PC2_tnuc']
def model_distr():
    model_runs = ['Smith', 'Smith_tnuc', 'PC2', 'PC2_tnuc']
    fig, ax = plt.subplots(5,2, figsize=(16,38))
    ax = ax.flatten()
    lab_dict = {2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8:'i', 9:'j'}
    plot = 2
    for run in model_runs:
        mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, cl_drop = load_model('Smith_tnuc')
        IWC_profile, LWC_profile, aer = load_obs() 
        ax[0].plot(IWC_profile, altitude, color = 'k', linewidth = 1.5)
        ax[0].set_xlim(0, 0.020)
        ax[0].set_ylim(0, max(altitude))
        ax[0].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', fontsize=28, labelpad=20)
        ax[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)) #use scientific notation on axes
        ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.setp(ax[0].get_xticklabels()[0], visible=False)
        ax[0].xaxis.get_offset_text().set_fontsize(24)
        ax[0].text(x=0.003, y = 4.8, s='a', fontsize=32, fontweight='bold', color='k')
        [w.set_linewidth(2) for w in ax[0].spines.itervalues()] # thick box outline
        ax[0].axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
        ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[1].plot(LWC_profile, altitude, color = 'k', linewidth = 1.5)
        ax[1].set_xlim(0, 0.20)
        ax[1].set_ylim(0, max(altitude))
        ax[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)) #use scientific notation on axes
        ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.setp(ax[1].get_xticklabels()[0], visible=False)
        ax[1].tick_params(labelleft='off')
        ax[1].xaxis.get_offset_text().set_fontsize(24)
        [w.set_linewidth(2) for w in ax[1].spines.itervalues()] # thick box outline
        ax[1].axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
        ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[1].plot(LWC_profile, altitude, color = 'k', linewidth = 1.5)
        ax[1].text(x=0.03, y=4.8, s='b', fontsize=32, fontweight='bold', color='k')
        print('\n PLOTTING DIS BIATCH...')
        ax[plot].plot(mean_QCF, altitude, color='k', linewidth=3, label='Ice')
        ax[plot].fill_betweenx(altitude, min_QCF, max_QCF, facecolor = 'k', alpha=0.2) # Shaded region between maxima and minima
        ax[plot].plot(ice_5, altitude, color='k', linestyle='--', linewidth=1) # Plot 5th and 95th percentiles
        ax[plot].plot(ice_95, altitude, color='k', linestyle='--', linewidth=1)
        ax[plot].set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize=28, labelpad=35)
        ax[plot].set_xlim(0, 0.020)
        ax[plot].set_ylim(0, max(altitude))
        ax[plot].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', fontsize=28, labelpad=20)
        ax[plot].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)) #use scientific notation on axes
        ax[plot].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.setp(ax[plot].get_xticklabels()[0], visible=False)
        ax[plot].text(x=0.003, y=4.8, s=lab_dict[plot], fontsize=32, fontweight='bold', color='k')
        ax[plot].xaxis.get_offset_text().set_fontsize(24)
        [w.set_linewidth(2) for w in ax[plot].spines.itervalues()] # thick box outline
        ax[plot].axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
        ax[plot].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax[plot+1].plot(mean_QCL, altitude, color='k', linewidth=3, label='Liquid')
        ax[plot+1].fill_betweenx(altitude, min_QCL, max_QCL, facecolor = 'k', alpha=0.2)
        ax[plot+1].plot(liq_95, altitude, color='k', linewidth=1, linestyle = '--')
        ax[plot+1].plot(liq_5, altitude, color='k', linewidth=1, linestyle = '--')
        ax[plot+1].set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize=28, labelpad=35)
        ax[plot+1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax[plot+1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax[plot+1].set_xlim(0,0.20)
        ax[plot+1].tick_params(labelleft='off')
        ax[plot+1].text(x=0.03, y=4.8, s=lab_dict[plot+1], fontsize=32, fontweight='bold', color='k')
        ax[plot+1].xaxis.get_offset_text().set_fontsize(24)
        plt.setp(ax[plot+1].get_xticklabels()[0], visible=False)
        ax[plot+1].set_ylim(0, max(altitude))
        [w.set_linewidth(2) for w in ax[plot+1].spines.itervalues()]
        #[l.set_visible(False) for (w,l) in enumerate(ax[plot+1].xaxis.get_ticklabels()) if w % 2 != 0]
        ax[plot+1].axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
        plot = plot+2
        print('\nDONE!')
        print('\nNEEEEEXT')
    plt.subplots_adjust(bottom=0.2, top=0.95, left=0.12, right=0.95, wspace=0.08)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/distr_vertical_profile_all_mod.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/distr_vertical_profile_all_mod.eps')
    #plt.show()

#model_distr()

# Plot model vs observations for QCF and QCL
def obs_mod_profile():
    fig, ax = plt.subplots(1,2, figsize=(16, 9))
    ax = ax.flatten()
    mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, mean_drop = load_model('Smith')
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs()
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        axs.set_ylim(0, max(altitude))
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    #m_QCF = ax[0].plot(mean_QCF, altitude, color = 'k', linestyle = '--', linewidth = 2.5)
    p_QCF = ax[0].plot(IWC_profile, altitude, color = 'k', linewidth = 2.5)
    ax[0].set_xlabel('Cloud ice mass mixing ratio \n(g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize = 28, labelpad = 35, color = 'dimgrey')
    ax[0].set_ylabel('Altitude \n(km)', rotation = 0, fontname='SegoeUI semibold', fontsize = 28, color = 'dimgrey', labelpad = 80)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax[0].set_xlim(0,0.0401)
    ax[0].xaxis.get_offset_text().set_fontsize(24)
    ax[0].xaxis.get_offset_text().set_color('dimgrey')
    ax[0].fill_betweenx(altitude, ice_5, ice_95, facecolor='#BDC8DC')  # Shaded region between maxima and minima
    #ax[0].plot(ice_5, altitude, color='darkslateblue', linestyle=':', linewidth=2)
    #ax[0].plot(ice_95, altitude, color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
    ax[0].text(x=0.003, y=4.8, s='a', fontsize=32, fontweight='bold', color='dimgrey')
    #m_14 = ax[0].plot(AWS14_mean_QCF, altitude, color='darkred', linestyle='--', linewidth=3)
    #m_15= ax[0].plot(AWS15_mean_QCF, altitude, color='darkblue', linestyle='--', linewidth=3)
    ax[1].set_xlabel('Cloud liquid mass mixing ratio \n(g kg$^{-1}$)',  fontname='SegoeUI semibold', color= 'dimgrey', fontsize = 28, labelpad = 35)
    p_QCL = ax[1].plot(LWC_profile, altitude, color = 'k', linewidth = 2.5, label = 'Observations')
    #m_QCL = ax[1].plot(mean_QCL, altitude, color = 'k', linestyle = '--', linewidth = 2.5, label = 'Model: \'cloud\' box mean')
    ax[1].fill_betweenx(altitude, liq_5, liq_95, facecolor='#BDC8DC', label = 'Model: 5$^{th}$ & 95$^{th}$ \npercentiles of \'cloud\' box')  # Shaded region between maxima and minima
    #ax[1].plot(liq_5, altitude, color='darkslateblue', linestyle=':', linewidth=2, label='5$^{th}$ & 95$^{th}$ percentiles')
    #ax[1].plot(liq_95, altitude, color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
    #m_14 = ax[1].plot(AWS14_mean_QCL, altitude, color='darkred', linestyle='--', linewidth=3, label='Model: AWS 14')
    #m_15 = ax[1].plot(AWS15_mean_QCL, altitude, color='darkblue', linestyle='--', linewidth=3, label='Model: AWS 15')
    ax[1].axes.tick_params(axis = 'both', which = 'both', direction = 'in', length = 5, width = 1.5,  labelsize = 24, pad = 10)
    ax[1].tick_params(labelleft = 'off')
    ax[1].set_xlim(0,0.401)
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[1].xaxis.get_offset_text().set_fontsize(24)
    ax[1].xaxis.get_offset_text().set_color('dimgrey')
    plt.setp(ax[0].get_xticklabels()[-3], visible=False)
    plt.setp(ax[1].get_xticklabels()[-3], visible=False)
    ax[1].text(x=0.03, y=4.8, s='b', fontsize=32, fontweight='bold', color='dimgrey')
    plt.subplots_adjust(wspace = 0.1, bottom = 0.23, top = 0.95, left = 0.17, right = 0.98)
    handles, labels = ax[1].get_legend_handles_labels()
    handles = [handles[0], handles[-1]]#, handles[1], handles[2],  handles[3] ]
    labels =  [labels[0], labels[-1]]#, labels[1], labels[2], labels[3]]
    lgd = plt.legend(handles, labels, fontsize=20, markerscale=2)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profiles_Smith_OBS_RANGE.eps')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profiles_Smith_OBS_RANGE.png')
    plt.show()

obs_mod_profile()

## Caption: Observed vertical profiles of ice and liquid mass fraction within the 'cloud' box. The solid black line indicates the liquid contents of the entire box, while the dashed
## line shows the ice mass fraction. Note that the ice and liquid mass fractions are plotted on different scales (differing by two orders of magnitude).

# Plot liquid vs. ice for observations only
def obs_profile():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL = load_model('Smith')
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array = load_obs()
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    ax2 = plt.twiny(ax)
    p_QCF = ax.plot(IWC_profile, altitude, color = 'k', linestyle = '--', linewidth = 1.5, label = 'ice')
    p_QCL = ax2.plot(LWC_profile, altitude, color = 'k', linewidth = 1.5, label = 'liquid')
    ax.set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize = 28, labelpad = 35)
    ax2.set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize = 28, labelpad = 35)
    ax.set_ylabel('Altitude (km)', fontname='SegoeUI semibold', fontsize = 28, labelpad = 20)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.setp(ax.get_xticklabels()[0], visible=False)
    plt.setp(ax2.get_xticklabels()[0], visible=False)
    ax.set_xlim(0,0.002)
    ax2.set_xlim(0, 0.2)
    ax.set_ylim(0, max(altitude))
    ax.xaxis.get_offset_text().set_fontsize(24)
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    ax.axes.tick_params(axis = 'both', which = 'both', direction = 'in', length = 5, width = 1.5,  labelsize = 24, pad = 10)
    ax2.axes.tick_params(axis = 'both', which = 'both', direction = 'in', length = 5, width = 1.5,  labelsize = 24, pad = 10)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    lns = p_QCF + p_QCL
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, markerscale= 2, loc=1, fontsize=24)
    plt.subplots_adjust(bottom = 0.15, top = 0.85, left = 0.15, right = 0.95)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/vertical_profile_obs.eps')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/vertical_profile_obs.png')

# Plot liquid vs. ice for observations only
def obs_var_scatter():
    fig = plt.figure(figsize=(17, 9))
    ax = fig.add_subplot(121)
    mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL = load_model('Smith_tnuc')
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array = load_obs()
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
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profile_obs_with_var.eps')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/vertical_profile_obs_with_var.png')
    #plt.show()

#obs_var_scatter()

## Caption: Modelled vertical profiles of ice and liquid mass fraction within the 'cloud' box. Panel a) shows the modelled ice mass fraction profile and panel b) shows the modelled
## liquid mass fraction profile. In both panels, the thick black line indicates the mean of the entire box, the dashed lines show the 5th and 95th percentiles, and the shaded area
## shows the full range of values. Note that the full range of ice mass fraction is not shown in panel a because the maximum is so far above the 95th percentile that it would make
## the detail invisible.

# 2 subplots, one each for ice and liquid: a) QCF, b) QCL. dashed lines are the 5th and 95th percentile, filled region shows full range of values (every gridbox in the 'cloud' region)

def mod_profile():
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, mean_drop = load_model('Smith_tnuc')
    ice_mass_frac = iris.load_cube('/data/mac/ellgil82/cloud_data/um/means/20110118T0000Z_Peninsula_km1p5_Smith_tnuc_pc000.pp','mass_fraction_of_cloud_ice_in_air')
    liq_mass_frac = iris.load_cube('/data/mac/ellgil82/cloud_data/um/means/20110118T0000Z_Peninsula_km1p5_Smith_tnuc_pc000.pp','mass_fraction_of_cloud_liquid_water_in_air')
    IWC_profile, LWC_profile, aer = load_obs()
    m_QCF = ax.plot(mean_QCF, (ice_mass_frac.coord('level_height').points / 1000), color='k', linewidth=3, label='ice')
    ax.fill_betweenx((ice_mass_frac.coord('level_height').points / 1000), min_QCF, max_QCF, facecolor = 'k', alpha=0.2)
    ax.plot(ice_5, (ice_mass_frac.coord('level_height').points / 1000), color='k', linestyle='--', linewidth=1)
    ax.plot(ice_95, (ice_mass_frac.coord('level_height').points / 1000), color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize=28, labelpad=35)
    ax.set_xlim(0, 0.025)
    ax.set_ylim(0, max((ice_mass_frac.coord('level_height').points / 1000)))
    ax.set_ylabel('Altitude (km)', fontname='SegoeUI semibold', fontsize=28, labelpad=20)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1g'))
    plt.setp(ax.get_xticklabels()[0], visible=False)
    ax.xaxis.get_offset_text().set_fontsize(24)
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    ax.axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
    at = AnchoredText("a", prop=dict(size=32), frameon=False, loc=1)
    ax.add_artist(at)
    ax2 = fig.add_subplot(122)
    ax2.fill_betweenx((ice_mass_frac.coord('level_height').points / 1000), min_QCL, max_QCL, facecolor = 'k', alpha=0.2)
    m_QCL = ax2.plot(mean_QCL, (ice_mass_frac.coord('level_height').points / 1000), color='k', linewidth=3, label='liquid')
    ax2.plot(liq_95, (ice_mass_frac.coord('level_height').points / 1000), color='k', linewidth=1, linestyle = '--')
    ax2.plot(liq_5, (ice_mass_frac.coord('level_height').points / 1000), color='k', linewidth=1, linestyle = '--')
    ax2.set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize=28, labelpad=35)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    #ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1g'))
    ax2.set_xlim(0,0.15)
    ax2.tick_params(labelleft='off')
    ax2.xaxis.get_offset_text().set_fontsize(24)
    plt.setp(ax2.get_xticklabels()[0], visible=False)
    ax2.set_ylim(0, max((ice_mass_frac.coord('level_height').points / 1000)))
    [i.set_linewidth(2) for i in ax2.spines.itervalues()]
    [l.set_visible(False) for (i,l) in enumerate(ax2.xaxis.get_ticklabels()) if i % 2 != 0]
    at = AnchoredText("b", prop=dict(size=32), frameon=False, loc=2)
    ax2.add_artist(at)
    ax2.axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.12, right=0.95)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/distr_vertical_profile_mod_Smith_mean.eps')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/distr_vertical_profile_mod_Smith_tnuc_mean.png')

def QCF_plot():
    model_runs = ['Smith', 'Smith_tnuc', 'PC2', 'PC2_tnuc']
    fig, ax = plt.subplots(2,2, figsize=(16,16))
    ax = ax.flatten()
    lab_dict = {0:'a', 1:'b', 2: 'c', 3: 'd'}
    plot = 0
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.set_xlim(0, 0.02)
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    for run in model_runs:
        mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, mean_drop = load_model(run)
        IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array, drop_profile = load_obs()
        ax2 = plt.twiny(ax[plot])
        ax2.set_xlim(0,0.02)
        ax2.axis('off')
        ax2.axes.tick_params(axis='both', which='both', tick1On=False, tick2On=False,  pad=10)
        plt.setp(ax2.get_yticklabels()[0], visible=False)
        plt.setp(ax2.get_xticklabels()[0], visible=False)
        ax2.axes.tick_params(labeltop='off')
        p_QCF = ax2.plot(IWC_profile, altitude, color='k', linewidth=3, label='Observations')
        #m_QCF = ax[plot].plot(mean_QCF, altitude, color='k', linestyle = '--', linewidth=3, label='Model: Cloud box')
        #m_14 = ax[plot].plot(AWS14_mean_QCF, altitude, color='darkred', linestyle = ':', linewidth=3, label='Model: AWS 14')
        #m_15= ax[plot].plot(AWS15_mean_QCF, altitude, color='darkred', linestyle='--', linewidth=3, label='Model: AWS 15')
        ax[plot].fill_betweenx(altitude, min_QCF, max_QCF, facecolor='lightslategrey', alpha = 0.5)# Shaded region between maxima and minima
        ax[plot].plot(ice_5, altitude, color='darkslateblue', linestyle=':', linewidth=2)
        ax[plot].plot(ice_95, altitude, color='darkslateblue', linestyle=':', linewidth=2)# Plot 5th and 95th percentiles
        ax[plot].set_xlim(0, 0.02)
        ax[plot].set_ylim(0, max(altitude))
        plt.setp(ax[plot].get_xticklabels()[0], visible=False)
        ax[plot].axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
        ax[plot].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        lab = ax[plot].text(x=0.002, y=4.8, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        print('\n PLOTTING DIS BIATCH...')
        plot = plot+1
        print('\nDONE!')
        print('\nNEEEEEXT')
    ax[0].axes.tick_params(labelbottom='off')
    ax[1].axes.tick_params(labelbottom = 'off', labelleft='off')
    ax[3].axes.tick_params(labelleft='off')
    ax[0].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', color='dimgrey',fontsize=28, labelpad=20)
    ax[2].set_ylabel('Altitude (km)', fontname='SegoeUI semibold',color='dimgrey',fontsize=28, labelpad=20)
    ax[2].set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold',color='dimgrey', fontsize=28, labelpad=35)
    ax[3].set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold',color='dimgrey', fontsize=28, labelpad=35)
    ax[2].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[3].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[3].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[2].xaxis.get_offset_text().set_fontsize(24)
    ax[3].xaxis.get_offset_text().set_fontsize(24)
    ax[2].xaxis.get_offset_text().set_color('dimgrey')
    ax[3].xaxis.get_offset_text().set_color('dimgrey')
    for axs in ax:
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    lns = p_QCF #+ m_QCF + m_14 + m_15 # create labels for legend
    labs = [l.get_label() for l in lns]
    ax[plot-1].legend(lns, labs, markerscale=2, loc=1, fontsize=24)
    lgd = ax[plot-1].legend(lns, labs, markerscale=2, loc=1, fontsize=24)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.12, right=0.95, hspace = 0.12, wspace=0.08)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/QCF_obs_v_all_mod_no_AWS_range_only.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/QCF_obs_v_all_mod_no_AWS_range_only.eps')
    #plt.show()

def QCL_plot():
    model_runs = ['Smith', 'Smith_tnuc', 'PC2', 'PC2_tnuc']
    fig, ax = plt.subplots(2,2, figsize=(16,16))
    ax = ax.flatten()
    lab_dict = {0:'a', 1:'b', 2: 'c', 3: 'd'}
    plot = 0
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.set_xlim(0,150)
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    for run in model_runs:
        mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, mean_drop = load_model(run)
        IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array, drop_profile = load_obs()
        ax2 = plt.twiny(ax[plot])
        ax2.set_xlim(0, 0.2)
        ax2.axis('off')
        ax2.axes.tick_params(axis='both', which='both',tick1On=False, tick2On=False,)
        plt.setp(ax2.get_yticklabels()[0], visible=False)
        plt.setp(ax2.get_xticklabels()[0], visible=False)
        ax2.axes.tick_params(labeltop='off')
        p_QCL = ax2.plot(LWC_profile, altitude, color='k', linewidth=3, label='Observations')
        m_QCL = ax[plot].plot(mean_QCL, altitude, color='k', linestyle = '--', linewidth=3, label='Model: Cloud box')
        #m_14 = ax[plot].plot(AWS14_mean_QCL, altitude, color='darkred', linestyle = ':', linewidth=3, label='Model: AWS 14')
        #m_15= ax[plot].plot(AWS15_mean_QCL, altitude, color='darkred', linestyle='--', linewidth=3, label='Model: AWS 15')
        ax[plot].fill_betweenx(altitude, min_QCL, max_QCL, facecolor='lightslategrey', alpha = 0.5)  # Shaded region between maxima and minima
        ax[plot].plot(liq_5, altitude, color='darkslateblue', linestyle=':', linewidth=2)
        ax[plot].plot(liq_95, altitude, color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
        ax[plot].set_xlim(0, 0.20)
        ax[plot].set_ylim(0, max(altitude))
        plt.setp(ax[plot].get_xticklabels()[0], visible=False)
        ax[plot].axes.tick_params(axis='both', which='both', tick1On=False, tick2On=False,)
        ax[plot].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        lab = ax[plot].text(x=0.025, y=4.8, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        print('\n PLOTTING DIS BIATCH...')
        plot = plot + 1
        print('\nDONE!')
        print('\nNEEEEEXT')
    ax[0].axes.tick_params(labelbottom='off')
    ax[1].axes.tick_params(labelbottom = 'off', labelleft='off')
    ax[3].axes.tick_params(labelleft='off')
    ax[0].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', color = 'dimgrey', fontsize=28, labelpad=20)
    ax[2].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', color = 'dimgrey',  fontsize=28, labelpad=20)
    ax[2].set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold',color = 'dimgrey', fontsize=28, labelpad=35)
    ax[2].xaxis.get_offset_text().set_fontsize(24)
    ax[3].xaxis.get_offset_text().set_fontsize(24)
    ax[2].xaxis.get_offset_text().set_color('dimgrey')
    ax[3].xaxis.get_offset_text().set_color('dimgrey')
    ax[3].set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', color = 'dimgrey', fontsize=28, labelpad=35)
    ax[2].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[3].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[3].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[2].xaxis.get_offset_text().set_fontsize(24)
    ax[3].xaxis.get_offset_text().set_fontsize(24)
    for axs in ax:
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    lns = p_QCL + m_QCL #+ m_14 + m_15
    labs = [l.get_label() for l in lns]
    lgd = ax[plot-1].legend(lns, labs, markerscale=2, loc=1, fontsize=24)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.12, right=0.95, hspace=0.12, wspace=0.08)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/QCL_obs_v_all_mod_no_AWS.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/QCL_obs_v_all_mod_no_AWS.eps')
    #plt.show()

def QCF_QCL_plot():
    model_runs = ['Smith', 'Smith_tnuc', 'PC2', 'PC2_tnuc']
    fig, ax = plt.subplots(4, 2, figsize=(16, 28))
    ax = ax.flatten()
    lab_dict = {0:'a', 1:'b', 2: 'c', 3: 'd', 4:'e', 5:'f', 6:'g', 7:'h'}
    plot = 0
    for run in model_runs:
        mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL = load_model(run)
        IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array = load_obs()
        ax2 = plt.twiny(ax[plot])
        ax2.set_xlim(0,0.02)
        ax2.axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
        plt.setp(ax2.get_yticklabels()[0], visible=False)
        plt.setp(ax2.get_xticklabels()[0], visible=False)
        ax2.axes.tick_params(labeltop='off')
        p_QCF = ax2.plot(IWC_profile, altitude, color='k', linewidth=3, label='Observations')
        m_QCF = ax[plot].plot(mean_QCF, altitude, color='k', linestyle = '--', linewidth=3, label='Model: Cloud box')
        m_14 = ax[plot].plot(AWS14_mean_QCF, altitude, color='darkred', linestyle = ':', linewidth=3, label='Model: AWS 14')
        m_15= ax[plot].plot(AWS15_mean_QCF, altitude, color='darkred', linestyle='--', linewidth=3, label='Model: AWS 15')
        ax[plot].fill_betweenx(altitude, min_QCF, max_QCF, facecolor='lightslategrey', alpha = 0.5)# Shaded region between maxima and minima
        ax[plot].plot(ice_5, altitude, color='darkslateblue', linestyle=':', linewidth=2)
        ax[plot].plot(ice_95, altitude, color='darkslateblue', linestyle=':', linewidth=2)# Plot 5th and 95th percentiles
        ax[plot].set_xlim(0, 0.02)
        ax[plot].set_ylim(0, max(altitude))
        plt.setp(ax[plot].get_xticklabels()[0], visible=False)
        [i.set_linewidth(2) for i in ax[plot].spines.itervalues()]
        ax[plot].axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
        ax[plot].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        [w.set_linewidth(2) for w in ax[plot].spines.itervalues()] # thick box outline
        lab = ax[plot].text(x=0.002, y=4.8, s=lab_dict[plot], fontsize=32, fontweight='bold', color='k')
        print('\n PLOTTING DIS BIATCH...')
        ax3 = plt.twiny(ax[plot+4])
        ax3.set_xlim(0, 0.2)
        ax3.axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
        plt.setp(ax3.get_yticklabels()[0], visible=False)
        plt.setp(ax3.get_xticklabels()[0], visible=False)
        ax3.axes.tick_params(labeltop='off')
        p_QCL = ax3.plot(LWC_profile, altitude, color='k', linewidth=3, label='Observations')
        m_QCL = ax[plot+4].plot(mean_QCL, altitude, color='k', linestyle='--', linewidth=3, label='Model: Cloud box')
        m_14 = ax[plot+4].plot(AWS14_mean_QCL, altitude, color='darkred', linestyle=':', linewidth=3,
                             label='Model: AWS 14')
        m_15 = ax[plot+4].plot(AWS15_mean_QCL, altitude, color='darkred', linestyle='--', linewidth=3,
                             label='Model: AWS 15')
        ax[plot+4].fill_betweenx(altitude, min_QCL, max_QCL, facecolor='lightslategrey',
                               alpha=0.5)  # Shaded region between maxima and minima
        ax[plot+4].plot(liq_5, altitude, color='darkslateblue', linestyle=':', linewidth=2)
        ax[+4].plot(liq_95, altitude, color='darkslateblue', linestyle=':',
                      linewidth=2)  # Plot 5th and 95th percentiles
        ax[plot+4].set_xlim(0, 0.2)
        ax[plot+4].set_ylim(0, max(altitude))
        plt.setp(ax[plot+4].get_xticklabels()[0], visible=False)
        [i.set_linewidth(2) for i in ax[plot+4].spines.itervalues()]
        ax[plot+4].axes.tick_params(axis='both', which='both', direction='in', length=5, width=1.5, labelsize=24, pad=10)
        ax[plot+4].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        [w.set_linewidth(2) for w in ax[plot+4].spines.itervalues()]  # thick box outline
        lab = ax[plot+4].text(x=0.002, y=4.8, s=lab_dict[plot+4], fontsize=32, fontweight='bold', color='k')
        print('\nDONE!')
        print('\nNEEEEEXT')
        plot = plot + 1
    ax[0].axes.tick_params(labelbottom='off')
    ax[1].axes.tick_params(labelbottom = 'off', labelleft='off')
    ax[3].axes.tick_params(labelleft='off')
    ax[0].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', fontsize=28, labelpad=20)
    ax[2].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', fontsize=28, labelpad=20)
    ax[2].set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize=28, labelpad=35)
    ax[3].set_xlabel('Ice mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize=28, labelpad=35)
    ax[2].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[3].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[3].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[2].xaxis.get_offset_text().set_fontsize(24)
    ax[3].xaxis.get_offset_text().set_fontsize(24)
    ax[4].axes.tick_params(labelbottom='off', labelleft='off')
    ax[5].axes.tick_params(labelbottom='off', labelleft = 'off')
    ax[6].axes.tick_params( labelleft='off')
    ax[7].axes.tick_params(labelleft='off')
    ax[6].set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize=28, labelpad=35)
    ax[6].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[6].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[7].set_xlabel('Liquid mass fraction (g kg$^{-1}$)', fontname='SegoeUI semibold', fontsize=28, labelpad=35)
    ax[7].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[7].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    lns = p_QCF + m_QCF + m_14 + m_15 # create labels for legend
    labs = [l.get_label() for l in lns]
    ax[plot-1].legend(lns, labs, markerscale=2, loc=1, fontsize=24)
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.12, right=0.95, hspace = 0.12, wspace=0.08)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/QCF_QCL_obs_v_all_mod_mean_with_AWS.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/QCF_QCL_obs_v_all_mod_mean_with_AWS.eps')
    #plt.show()

def drop_plot():
    model_runs = ['Smith', 'Smith_tnuc', 'PC2', 'PC2_tnuc']
    fig, ax = plt.subplots(2,2, figsize=(16,16))
    ax = ax.flatten()
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.set_xlim(0,200)
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    lab_dict = {0:'a', 1:'b', 2: 'c', 3: 'd'}
    plot = 0
    for run in model_runs:
        mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, mean_drop = load_model(run)
        IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs()
        ax2 = plt.twiny(ax[plot])
        ax2.set_xlim(0, 250)
        ax2.axis('off')
        plt.setp(ax2.get_yticklabels()[0], visible=False)
        plt.setp(ax2.get_xticklabels()[0], visible=False)
        ax2.axes.tick_params(labeltop='off')
        ax2.tick_params(axis = 'both', which='both', tick1On=False, tick2On=False, )
        p_drop = ax2.plot(drop_profile, altitude, color='k', linewidth=3, label='Observations')
        m_drop = ax[plot].plot(mean_drop, altitude, color='k', linestyle = '--', linewidth=3, label='Model: Cloud box')
        #m_14 = ax[plot].plot(AWS14_mean_QCL, altitude, color='darkred', linestyle = ':', linewidth=3, label='Model: AWS 14')
        #m_15= ax[plot].plot(AWS15_mean_QCL, altitude, color='darkred', linestyle='--', linewidth=3, label='Model: AWS 15')
        #ax[plot].fill_betweenx(altitude, min_QCL, max_QCL, facecolor='lightslategrey', alpha = 0.5)  # Shaded region between maxima and minima
        #ax[plot].plot(liq_5, altitude, color='darkslateblue', linestyle=':', linewidth=2)
        #ax[plot].plot(liq_95, altitude, color='darkslateblue', linestyle=':', linewidth=2)  # Plot 5th and 95th percentiles
        #ax[plot].set_xlim(0, 0.20)
        ax[plot].set_ylim(0, max(altitude))
        plt.setp(ax[plot].get_xticklabels()[0], visible=False)
        [i.set_linewidth(2) for i in ax[plot].spines.itervalues()]
        ax[plot].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        lab = ax[plot].text(x=25, y=4.8, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        print('\n PLOTTING DIS BIATCH...')
        plot = plot + 1
        print('\nDONE!')
        print('\nNEEEEEXT')
    ax[0].axes.tick_params(labelbottom='off')
    ax[1].axes.tick_params(labelbottom = 'off', labelleft='off')
    ax[3].axes.tick_params(labelleft='off')
    ax[0].set_ylabel('Altitude (km)', fontname='SegoeUI semibold', color = 'dimgrey', fontsize=28, labelpad=20)
    ax[2].set_ylabel('Altitude (km)', fontname='SegoeUI semibold',  color = 'dimgrey', fontsize=28, labelpad=20)
    ax[2].set_xlabel('Droplet number \nconcentration (cm$^{-3}$)', fontname='SegoeUI semibold',  color = 'dimgrey', fontsize=28, labelpad=35)
    ax[2].xaxis.get_offset_text().set_fontsize(24)
    ax[3].xaxis.get_offset_text().set_fontsize(24)
    ax[3].set_xlabel('Droplet number \nconcentration (cm$^{-3}$)',  color = 'dimgrey', fontname='SegoeUI semibold', fontsize=28, labelpad=35)
    ax[2].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[3].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax[3].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax[2].xaxis.get_offset_text().set_fontsize(24)
    ax[3].xaxis.get_offset_text().set_fontsize(24)
    ax[2].xaxis.get_offset_text().set_color('dimgrey')
    ax[3].xaxis.get_offset_text().set_color('dimgrey')
    for axs in ax:
        axs.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    lns = p_drop + m_drop# + m_14 + m_15
    labs = [l.get_label() for l in lns]
    lgd = ax[plot-1].legend(lns, labs, markerscale=2, loc=1, fontsize=24)
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.12, right=0.95, hspace=0.12, wspace=0.08)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/droplet_obs_v_all_mod.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/droplet_obs_v_all_mod.eps')
    #plt.show()

#drop_plot()

def ice_plot():
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
    [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs()
    mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, mean_drop = load_model('Smith')
    ax.set_xlim(0, 0.003)
    ax.tick_params(axis = 'both', which='both', tick1On=False, tick2On=False, )
    ax.plot(n_ice_profile, altitude, color='k', linewidth=3, label='Observations')
    ax.set_ylim(0, max(altitude))
    plt.setp(ax.get_xticklabels()[0], visible=False)
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel('Altitude \n(km)', fontname='SegoeUI semibold', color = 'dimgrey', fontsize=28, rotation = 0, labelpad=20)
    ax.set_xlabel('Ice particle number \nconcentration (cm$^{-3}$)', fontname='SegoeUI semibold',  color = 'dimgrey', fontsize=28, labelpad=35)
    ax.xaxis.get_offset_text().set_fontsize(24)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_color('dimgrey')
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    plt.subplots_adjust(bottom=0.23, top=0.95, left=0.23, right=0.95)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/n_ice_obs_v_all_mod.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/n_ice_obs_v_all_mod.eps')
    plt.show()


def obs_drop_plot():
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
    [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    IWC_profile, LWC_profile, aer, IWC_array, LWC_array, alt_array_ice, alt_array_liq, drop_profile, drop_array, nconc_ice, box_IWC, box_LWC, box_nconc_ice, box_nconc_liq, n_ice_profile = load_obs()
    mean_QCF, mean_QCL, altitude, liq_5, liq_95, max_QCF, max_QCL, min_QCF, min_QCL, ice_5, ice_95, AWS14_mean_QCF, AWS14_mean_QCL, AWS15_mean_QCF, AWS15_mean_QCL, mean_drop = load_model('Smith')
    ax.set_xlim(0, 250)
    ax.tick_params(axis = 'both', which='both', tick1On=False, tick2On=False, )
    ax.plot(drop_profile, altitude, color='k', linewidth=3, label='Observations')
    ax.set_ylim(0, max(altitude))
    plt.setp(ax.get_xticklabels()[0], visible=False)
    [i.set_linewidth(2) for i in ax.spines.itervalues()]
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel('Altitude \n(km)', fontname='SegoeUI semibold', color = 'dimgrey', fontsize=28, rotation = 0, labelpad=20)
    ax.set_xlabel('Droplet number \nconcentration (cm$^{-3}$)', fontname='SegoeUI semibold',  color = 'dimgrey', fontsize=28, labelpad=35)
    ax.xaxis.get_offset_text().set_fontsize(24)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))  # use scientific notation on axes
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_color('dimgrey')
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    plt.subplots_adjust(bottom=0.23, top=0.95, left=0.23, right=0.95)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/n_drop_obs.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Vertical profiles/n_drop_obs.eps')
    plt.show()

#obs_drop_plot()
#ice_plot()

#mod_profile()
#QCL_plot()
#QCF_plot()
#obs_profile()


