## -------------------------- COMPARE RADIATIVE BIASES IN THE MODEL VS. OBSERVATIONS -------------------------------- ##
# File management: make sure all model runs are in one containing folder. Presently, this is /data/mac/ellgil82/cloud_data/um/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import iris
import os
import fnmatch
import matplotlib.patheffects as PathEffects
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib import rcParams

## Define functions
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

def load_model(var):
    pa = []
    pf = []
    print('\nimporting data from %(var)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/'):
            if fnmatch.fnmatch(file, '*%(var)s_pf*' % locals()):
                pf.append(file)
            elif fnmatch.fnmatch(file, '*%(var)s_pa*' % locals()):
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
    if var == 'CASIM_24' or var == 'CASIM_24_DeMott' or var == 'CASIM_f152_ice_off':
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
    if var =='CASIM_24' or var == 'CASIM_24_DeMott' or var == 'CASIM_f152_ice_off':
        var_dict = {'real_lon': real_lon, 'real_lat': real_lat, 'SW_up': SW_up, 'SW_down': SW_down,
                    'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down, 'LW_net': LW_net, 'SW_net': SW_net}
    else:
        var_dict = {'real_lon': real_lon, 'real_lat':real_lat,  'SW_up': SW_up, 'SW_down': SW_down,
                    'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down, 'LW_net': LW_net, 'SW_net': SW_net, 'Ts': Ts}
    return var_dict


#t24_vars = load_model('24')
RA1M_vars = load_model('RA1M_24')
RA1M_mod_vars = load_model('RA1M_mod_24')
RA1T_vars = load_model('RA1T_24')
RA1T_mod_vars = load_model('RA1T_mod_24')
#fl_av_vars = load_model('fl_av')
DeMott_vars = load_model('CASIM_24_DeMott')
Cooper_vars = load_model('CASIM_24')
ice_off_vars = load_model('CASIM_f152_ice_off')
model_runs = [RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars, Cooper_vars,  ice_off_vars]# fl_av_vars,

## Load AWS metadata: data are formatted so that row [0] is the latitude, row [1] is the longitude, and each AWS is in a
## separate column, so it can be indexed in the pandas dataframe
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
    Jan18 = AWS.loc[(AWS['Day'] == 18)]# & (AWS['Hour'] >= 12)]
    #Jan18 = Jan18.append(AWS.loc[(AWS['Day'] == 19) & (AWS['Hour'] == 0)])
    Day_mean = Jan18.mean(axis=0) # Calculates averages for whole day
    Flight = Jan18.loc[(Jan18['Hour'] >=15) &  (Jan18['Hour'] <= 17)]#[(Jan18['Hour'] >= 12)]#
    Flight_mean = Flight.mean(axis=0) # Calculates averages over the time period sampled (15:00 - 17:00)
    return Flight_mean, Day_mean, Jan18

AWS14_flight_mean, AWS14_day_mean, AWS14_Jan = load_AWS('AWS14')
AWS15_flight_mean, AWS15_day_mean, AWS15_Jan = load_AWS('AWS15')
AWS14_SEB_flight_mean, AWS14_SEB_day_mean, AWS14_SEB_Jan  = load_AWS('AWS14_SEB')

## ----------------------------------------------- COMPARE MODEL & AWS ---------------------------------------------- ##

real_lat = RA1M_mod_vars['real_lat']
real_lon = RA1M_mod_vars['real_lon']

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


## ------------------------------------------- CALCULATE BIASES ----------------------------------------------------- ##
def calc_bias(run, times, day): # times should be in tuple format, i.e. (start, end) and day should be True or False
    AWS14_bias = []
    AWS15_bias = []
    for i, j in zip([run['LW_down'].data, run['LW_up'][:,0,:,:].data, run['LW_net'].data, run['SW_down'].data,run['SW_up'][:,0,:,:].data, run['SW_net'].data, run['LH'], run['SH']],
                    ['LWin', 'LWout_corr', 'LWnet_corr','SWin_corr', 'SWout','SWnet_corr','Hlat','Hsen']):
        if day == True:
            AWS14_bias.append(np.mean(np.mean(i[::2,  (AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1,2)) - AWS14_SEB_Jan[j]))
        elif day == False:
            AWS14_bias.append(np.mean(np.mean(i[times[0]:times[1]:2, (AWS14_lon - 1):(AWS14_lon + 1), (AWS14_lat - 1):(AWS14_lat + 1)], axis = (1,2)) - AWS14_SEB_Jan[j][(times[0]/2):(times[-1]/2)]))
        else:
            print('\'day\' must be set to True or False')
    if day == True:
        AWS14_bias.append(np.mean(melt_masked_day - AWS14_SEB_Jan['melt_energy'][::2]))
    elif day == False:
        AWS14_bias.append(np.mean(np.mean(melt_masked_flight) - AWS14_SEB_Jan['melt_energy'][(times[0]/2):(times[-1]/2)]))
    for i, j in zip([run['LW_down'].data, run['LW_up'][:,0,:,:].data,  run['SW_down'].data, run['SW_up'][:,0,:,:].data],['Lin', 'Lout', 'Sin', 'Sout']):
        if day == True:
            AWS15_bias.append((np.mean(i[:, (AWS15_lon - 1):(AWS15_lon + 1), (AWS15_lat - 1):(AWS15_lat + 1)])) - AWS15_day_mean[j])
        elif day == False:
            AWS15_bias.append((np.mean(i[times[0]:times[1], (AWS15_lon - 1):(AWS15_lon + 1), (AWS15_lat - 1):(AWS15_lat + 1)])) - AWS15_flight_mean[j])
    return AWS14_bias, AWS15_bias

def calc_vals(run):
    AWS14_vals = []
    AWS15_vals = []
    for i in [run['LW_down'], run['LW_up'][:,0,:,:], run['SW_down'], run['SW_up'][:,0,:,:]]:
        AWS14_vals.append(np.mean(i[59:68,  (AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data))
        AWS15_vals.append(np.mean(i[59:68,  (AWS15_lon-1):(AWS15_lon+1), (AWS15_lat-1):(AWS15_lat+1)].data))

names = ['RA1M_mod', 'RA1M_mod']#['RA1M', 'RA1M_mod', 'RA1T', 'RA1T_mod', 'Cooper', 'DeMott']
#model_runs = [RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars, CASIM_vars]
model_runs = [RA1M_mod_vars, RA1M_mod_vars]

for run, name in model_runs, names:
    Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15, melt_masked_day, melt_masked_flight, \
    obs_SEB_AWS14_flight,  obs_SEB_AWS14_day, obs_melt_AWS14_flight, obs_melt_AWS14_day = calc_SEB(run, times = (68,80))
    AWS14_bias, AWS15_bias = calc_bias(run, times = (68,80), day = False)
    print '\n\n'+name + ' bias:\n\n'
    print AWS14_bias, AWS15_bias




## -------------------------------------------------- PLOTTING ------------------------------------------------------ ##
## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans',
                               'Verdana']
def plot_Ts():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('$T_s$ \n\n($^{\circ}$C)', fontsize=32, color='dimgrey', rotation = 0, labelpad = 50)
    ax.tick_params(axis='both', which='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax.set_xticks([0,12,24])
    ax2 = ax.twiny()
    ax2.axis('off')
    ax.set_xlim(0, 24)
    ax2.set_xlim(816, 863)
    ax.axhline(y=0, xmin=0, xmax=1, color='dimgrey', linestyle='--', linewidth=1)
    #ax.axes.get_xaxis().set_visible(False)
    T_srs = np.mean(run['Ts'][:, (AWS14_lat - 1):(AWS14_lat + 1), (AWS14_lon - 1):(AWS14_lon + 1)].data, axis=(1, 2))
    ax.plot(T_srs, color='#400C60', linestyle='--', lw=2.5, label='model')
    ax2.plot(AWS14_SEB_Jan['Tsobs'], color='#400C60', label='observed', lw=2.5)
    ax.axvspan(17, 20, edgecolor='dimgrey', facecolor='dimgrey', alpha=0.5)
    ax.axvline(17.25, linestyle=':', lw=2.5, color='dimgrey')
    ax.axvline(14.75, linestyle=':', lw=2.5, color='dimgrey')
    ax.tick_params(axis='both', which='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d:00"))
    plt.subplots_adjust(left = 0.25)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_Ts_RA1M_mod.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_Ts_RA1M_mod.png', transparent=True)
    plt.show()

plot_Ts()

def E_melt_plot():
    fig, ax = plt.subplots(2,1, figsize = (10,10))
    ax = ax.flatten()
    for axs in ax:
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey')
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.tick_params(axis='both', which='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax2 = ax[0].twiny()
    ax2.axis('off')
    ax[0].set_xlim(0,96)
    ax2.set_xlim(816,863)
    ax[0].axhline(y=0, xmin=0, xmax=1, color = 'dimgrey', linestyle='--', linewidth=1)
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].plot(Model_SEB_day_AWS14,color = '#400C60', linestyle = '--', lw = 2.5, label = 'model')
    ax2.plot(obs_SEB_AWS14_day,color = '#400C60', label = 'observed', lw = 2.5)
    ax[0].axvspan(68,80, edgecolor='dimgrey', facecolor='dimgrey', alpha=0.5)
    ax[0].axvline(59, linestyle = ':', lw = 2.5, color='dimgrey')
    ax[0].axvline(69, linestyle=':', lw=2.5, color='dimgrey')
    ax2 = ax[1].twiny()
    ax2.axis('off')
    ax[1].set_xlim(0,24)
    ax2.set_xlim(816,863)
    ax[1].set_xticks([0,12,24])
    ax[1].plot(melt_masked_day,color='#f68080', linestyle = '--', label = 'model', lw=2.5)
    ax2.plot(obs_melt_AWS14_day,color='#f68080', label = 'observed', lw = 2.5)
    ax[0].set_ylabel('$E_{tot}$ \n(W m$^{-2}$)', fontsize = 28, color = 'dimgrey', labelpad = 60, rotation = 0)
    ax[1].set_ylabel('$E_{melt}$ \n(W m$^{-2}$)', fontsize = 28, color = 'dimgrey', labelpad = 80, rotation =0)
    ax[1].tick_params(axis='both', which='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax[1].axvspan(17,20, edgecolor='dimgrey', facecolor='dimgrey', alpha=0.5)
    ax[1].axvline(14.75, linestyle=':', lw=2.5, color='dimgrey')
    ax[1].axvline(17.25, linestyle=':', lw=2.5, color='dimgrey')
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d:00"))
    plt.subplots_adjust(left = 0.3, right = 0.95, top = 0.98, bottom = 0.15)
    lns = [Line2D([0], [0], color='dimgrey', linewidth=3),
           Line2D([0], [0], color='dimgrey', linestyle='--', linewidth=3)]
    labs = ['Observed',
            'Modelled']  # ,'AWS 15, observed', 'AWS 15, modelled']#  '                      ','                      '
    lgd = plt.legend(lns, labs, bbox_to_anchor=(1., 1.34), borderaxespad=0., loc='best', prop={'size': 24})
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_melt+Etot_RA1M_mod.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_melt+Etot_RA1M_mod.png', transparent=True)
    plt.show()

E_melt_plot()

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
    axs[0].set_xlim(AWS14_SEB_Jan['Time'].values[24], AWS14_SEB_Jan['Time'].values[-1])
    axs[0].axvspan(18.7083, 18.83333, edgecolor='dimgrey', facecolor='dimgrey', alpha=0.5)
    axs[0].axvline(18.614583, linestyle = ':', lw = 5, color='dimgrey')
    axs[0].axvline(18.7083, linestyle=':', lw=5, color='dimgrey')
    # Plot model SEB
    Model_time = run['SW_net'].coord('time')
    Model_time.convert_units('seconds since 1970-01-01 00:00:00')
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
    ax2.plot(melt_masked_day[12:], color='#f68080', lw=5, label='Melt flux')
    axs[1].axvspan(Model_time[68], Model_time[80], edgecolor='dimgrey', facecolor='dimgrey', alpha=0.5)
    axs[1].axvline(Model_time[68], linestyle = ':', lw = 5, color='dimgrey')
    axs[1].axvline(Model_time[59], linestyle=':', lw=5, color='dimgrey')
    axs[1].set_xlim(Model_time[48], Model_time[-1])
    axs[1].set_xticks([Model_time[48], Model_time[59], Model_time[68], Model_time[80], Model_time[-1]])
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
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_SEB__RA1M_mod.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_SEB_RA1M_mod.png', transparent=True)
    plt.show()

total_SEB(RA1M_mod_vars)

def rad_time_srs():
    model_runs = [RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars, Cooper_vars, DeMott_vars, ice_off_vars, ]#[RA1M_mod_vars]#
    fig, ax = plt.subplots(len(model_runs),2, sharex='col', figsize=(16,len(model_runs*5)+3), squeeze=False)#(18,8))#
    ax = ax.flatten()
    ax2 = np.empty_like(ax)
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        #[l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        #[l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        axs.axvline(x = 14.75, color = '#222222', alpha = 0.5, linestyle = ':', linewidth = 3)
        axs.axvline(x=17, color='#222222', alpha=0.5, linestyle=':', linewidth=3)
        axs.axvspan(14.75,17, edgecolor = 'dimgrey', facecolor = 'dimgrey', alpha = 0.2,)
        axs.axvspan(17, 20, edgecolor = 'dimgrey', facecolor='dimgrey', alpha=0.5) #shifted  = 17,20 / normal = 14.75, 17
        axs.arrow(x=0.4, y=0.95, dx = .4, dy = 0., linewidth = 3, color='k', zorder = 10)
    def my_fmt(x,p):
        return {0}.format(x) + ':00'
    plot = 0
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o'}
    for run in model_runs:
        AWS14_flight_mean, AWS14_day_mean, AWS14_Jan = load_AWS('AWS14')
        AWS15_flight_mean, AWS15_day_mean, AWS15_Jan = load_AWS('AWS15')
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
        ax[plot].text(x=13, y=750, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
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
        ax[plot+1].text(x=13, y=300, s=lab_dict[plot+1], fontsize=32, fontweight='bold', color='dimgrey')
        ax[plot+1].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        #[l.set_visible(False) for (i, l) in enumerate(ax[plot+1].yaxis.get_ticklabels()) if i % 2 != 0]
        #[l.set_visible(False) for (i, l) in enumerate(ax[plot + 1].xaxis.get_ticklabels()) if i % 2 != 0]
        [w.set_linewidth(2) for w in ax[plot].spines.itervalues()]
        [w.set_linewidth(2) for w in ax[plot+1].spines.itervalues()]
        #ax[plot+1].set_xlim(run['LW_down'].coord('time').points[1], run['LW_down'].coord('time').points[-1])
        #ax2[plot+1].set_xlim(AWS15_Jan['Hour'].values[0], AWS15_Jan['Hour'].values[-1]) ##
        plt.setp(ax2[plot].get_xticklabels(), visible=False)
        plt.setp(ax2[plot+1].get_xticklabels(), visible=False)
        titles = ['    RA1M','    RA1M', 'RA1M_mod', 'RA1M_mod']#, '     fl_av','     fl_av', '    RA1T', '    RA1T', 'RA1T_mod','RA1T_mod', '   CASIM', '   CASIM']
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
    plt.subplots_adjust(left=0.22, bottom=0.12, right=0.78, top=0.97, wspace=0.15, hspace=0.15)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/vn11_SEB_time_srs_LWd_SWd_all_runs_shifted_AWS14.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/vn11_SEB_time_srs_LWd_SWd_all_runs_shifted_AWS14.eps', transparent = True)
    plt.show()

rad_time_srs()


def cl_cover():
    model_runs = ['Smith', 'Smith_tnuc', 'PC2', 'PC2_tnuc']
    fig, ax = plt.subplots(2,2, sharex='col', figsize=(12,16))#, squeeze=False)
    ax = ax.flatten()
    ax2 = np.empty_like(ax)
    plot = 0
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    for run in model_runs:
        LW_net_surf, SW_net_surf, toa_outgoing_LW, SW_down, LW_dif, SW_dif, lsm, orog, real_lon, real_lat, time_srs15, time_srs14, v_low_cl, low_cl, mid_cl, high_cl = load_model(run)
        AWS14_flight_mean, AWS14_day_mean, AWS14_Jan = load_AWS('AWS14')
        AWS15_flight_mean, AWS15_day_mean, AWS15_Jan = load_AWS('AWS15')
        print('\nPLOTTING DIS BIATCH...')
        # Calculate total cloud fraction
        #low_cl = ax[plot].plot(np.arange(12,24), time_srs14[])
        mod14 = ax[plot].plot(np.arange(12,24), time_srs14[2], label = 'AWS14, modelled', linewidth = 1.5, linestyle = '--', color = 'darkred')
        ax2[plot] = plt.twiny(ax[plot])
        obs14 = ax2[plot].plot(AWS14_Jan['Hour'], AWS14_Jan['Cloudcover'], label='AWS14, observed', linewidth=1.5, color='darkred')
        ax[plot].text(x=13, y=0.9, s=lab_dict[plot], fontsize=24, fontweight='bold', color='k')
        ax[plot].axvspan(15, 17, facecolor='grey', alpha=0.5)
        ax[plot].tick_params(axis='both', labelsize=24, tick1On=True, tick2On=True, length=5, direction='in', pad=10)
        [l.set_visible(False) for (i, l) in enumerate(ax[plot].yaxis.get_ticklabels()) if i % 2 != 0]
        [l.set_visible(False) for (i, l) in enumerate(ax[plot].xaxis.get_ticklabels()) if i % 2 != 0]
        plt.setp(ax[plot].get_xticklabels()[-1], visible=False)
        [w.set_linewidth(2) for w in ax[plot].spines.itervalues()]
        plt.setp(ax2[plot].get_xticklabels(), visible=False)
        print('\nDONE!')
        print('\nNEEEEEXT')
        plot = plot + 1
    lns = obs14 + mod14
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, ncol=2, bbox_to_anchor=(1.4, -0.4), borderaxespad=0., loc='best', prop={'size': 18})
    plt.subplots_adjust(left=0.18, bottom=0.17, right=0.8, top=0.98, wspace = 0.08, hspace = 0.08)
    fig.text(0.5, 0.11, 'Time (hours)', fontsize=24, fontweight = 'bold', ha = 'center', va = 'center')
    fig.text(0.05, 0.55, 'Cloud cover', fontsize=24, fontweight = 'bold', ha= 'center', va='center', rotation = 90)
    fig.text(0.95, 0.55, 'Cloud cover', fontsize=24, ha='center', fontweight = 'bold', va='center', rotation=270)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/Cl_cover_time_srs_all_mod.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/Cl_cover_time_srs_all_mod.eps')
    plt.show()

from itertools import chain
import scipy

def correl_plot():
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



#correl_plot()

Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15, melt_masked_day, melt_masked_flight, \
obs_SEB_AWS14_flight,  obs_SEB_AWS14_day, obs_melt_AWS14_flight, obs_melt_AWS14_day = calc_SEB(RA1M_mod_vars, times = (68,80))

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
    axs.plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['SWnet_corr'][24:], color='#6fb0d2', lw=5, label='SW$_{net}$', zorder = 9)
    axs.plot(AWS14_SEB_Jan['Time'][24:], AWS14_SEB_Jan['LWnet_corr'][24:], color='#86ad63', lw=5, label='LW$_{net}$', zorder = 9)
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
    ax.plot(Model_time[48:], np.mean(run['SW_net'][48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1, 2)), linestyle = '--',color='#6fb0d2', lw=5, label='SW$_{net}$', zorder = 10)
    ax.plot(Model_time[48:], np.mean(run['LW_net'][48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1, 2)), linestyle = '--',color='#86ad63', lw=5, label='LW$_{net}$', zorder = 10)
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
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_SEB_difs_RA1M_mod.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_SEB_difs_RA1M_mod.png', transparent=True)
    plt.show()


#SEB_diff(RA1M_mod_vars)
#
def melt_map():
    fig, axs  = plt.subplots(1,2, figsize = (16, 8))#, figsize=(20, 12), frameon=False)
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l'}
    run = [RA1M_mod_vars, RA1M_mod_vars]#[RA1M_SEB, RA1M_mod_SEB, RA1T_SEB, RA1T_mod_SEB, CASIM_SEB, DeMott_SEB]
    # Import data to plot LSM and orography
    lsm = iris.load_cube('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/20110118T0000Z_Peninsula_1p5km_RA1M_mod_24_pa000.pp', 'land_binary_mask')
    orog = iris.load_cube('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/t24/20110118T0000Z_Peninsula_1p5km_RA1M_mod_24_pa000.pp', 'surface_altitude')
    for i in [lsm, orog]:
        real_lat, real_lon = rotate_data(i, 0,1)
    for a in [0,1]:#,2,3,5,6]:
        # Set up plot
        axs[a].spines['right'].set_visible(False)
        axs[a].spines['left'].set_visible(False)
        axs[a].spines['top'].set_visible(False)
        axs[a].spines['bottom'].set_visible(False)
        # Plot LSM and orography
        axs[a].contour(real_lon, real_lat, lsm.data, colors='#535454', linewidths=2.5,zorder=5)  # , transform=RH.coord('grid_longitude').coord_system.as_cartopy_projection())
        axs[a].contour(real_lon, real_lat, orog.data, colors='#535454', levels=[15], linewidths=2.5, zorder=6)
        lsm_masked = np.ma.masked_where(lsm.data == 1, lsm.data)
        orog_masked = np.ma.masked_where(orog.data < 15, orog.data)
        # Mask orography above 15 m
        axs[a].contourf(real_lon, real_lat, orog_masked, colors = 'w', zorder = 3)
        # Make the sea blue
        axs[a].contourf(real_lon, real_lat, lsm_masked, colors='#a6cee3', zorder=2)
        # Sort out ticks
        axs[a].tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=15, size=0, tick1On=False, tick2On=False)
        PlotLonMin = np.min(real_lon)
        PlotLonMax = np.max(real_lon)
        PlotLatMin = np.min(real_lat)
        PlotLatMax = np.max(real_lat)
        XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
        XTickLabels = [None] * len(XTicks)
        for i, XTick in enumerate(XTicks):
            if XTick < 0:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
            else:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')
        plt.sca(axs[a])
        plt.xticks(XTicks, XTickLabels)
        axs[a].set_xlim(PlotLonMin, PlotLonMax)
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
        YTickLabels = [None] * len(YTicks)
        for i, YTick in enumerate(YTicks):
            if YTick < 0:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
            else:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
        plt.sca(axs[a])
        plt.yticks(YTicks, YTickLabels)
        axs[a].set_ylim(PlotLatMin, PlotLatMax)
        # Add plot labels
        lab = axs[a].text(-80, -61.5, zorder=10,  s=lab_dict[a], fontsize=32, fontweight='bold', color='dimgrey')
        # Calculate model melt rates
        Model_E = np.mean((run[a]['LW_net'][68:80,:,:].data + run[a]['SW_net'][68:80,:,:].data + run[a]['SH'][68:80,:,:] + run[a]['LH'][68:80,:,:]), axis=0)
        melt_spatial = np.ma.masked_where((np.mean(run[a]['Ts'][17:20,:,:].data, axis=0) < -0.025) & lsm.data == 1, Model_E)
        Ts_mean= np.mean(run[a]['Ts'][17:20,:,:].data, axis=0)
        melt_spatial[(Ts_mean < -0.025) & (Model_E > 0)] = 0
        melt_spatial[(lsm.data == 0)] = 0
        # Convert to mmwe
        Lf = 334000  # J kg-1
        rho_H2O = 999.7  # kg m-3
        melt_mmwe = ((melt_spatial / (Lf * rho_H2O))*10800)*1000
        melt_mmwe = np.ma.masked_where(lsm == 0, melt_mmwe)
        # Calculate observed melt
        melt_obs = (( obs_melt_AWS14_flight/ (Lf * rho_H2O))*10800)*1000
        # Plot model melt rates
        x, y = np.meshgrid(real_lon, real_lat)
        c = axs[a].pcolormesh(x,y , melt_mmwe, cmap='viridis', vmin=0, vmax=3, zorder=1)
        # Plot observed melt rate at AWS 14
        # Hacky fix to plot melt at right colour
        axs[a].scatter(-67.01,-61.50, c = melt_obs, s = 100,marker='o', edgecolors = 'w',vmin=0, vmax=3, zorder = 100, cmap = matplotlib.cm.viridis )
        #axs[a].plot(-67.01,-61.50,  markerfacecolor='#3d4d8a', markersize=15, marker='o', markeredgecolor='w', zorder=100)#
    # Add colourbar
    CBarXTicks = [0,3]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.15, 0.15, 0.3, 0.04])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks)  #
    CBar.set_label('Melt rate (mm w.e.)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    # Sort out plot and save
    plt.subplots_adjust(bottom = 0.3, top = 0.95, wspace = 0.25, hspace = 0.25)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/melt_map.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/melt_map.eps', transparent=True)
    plt.show()

melt_map()

