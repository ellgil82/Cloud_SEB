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
    if var == 'CASIM_24':
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
    if var =='CASIM_24':
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
fl_av_vars = load_model('fl_av')
CASIM_vars = load_model('CASIM_24')
model_runs = [RA1M_vars, RA1M_mod_vars, RA1T_vars, RA1T_mod_vars, CASIM_vars]# fl_av_vars,

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
    Flight = Jan18.loc[(Jan18['Hour'] >= 12)]#[(Jan18['Hour'] >=15) &  (Jan18['Hour'] <= 17)]
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
    AWS14_SEB_flight = AWS14_SEB_flight_mean['SWnet_corr'] + AWS14_SEB_flight_mean['LWnet_corr'] + AWS14_SEB_flight_mean['Hsen'] + AWS14_SEB_flight_mean['Hlat'] - AWS14_SEB_flight_mean['Gs']
    AWS14_melt_flight = AWS14_SEB_flight_mean['melt_energy']
    AWS14_SEB_day = AWS14_SEB_day_mean['SWnet_corr'] + AWS14_SEB_day_mean['LWnet_corr'] + AWS14_SEB_day_mean['Hsen'] + AWS14_SEB_day_mean['Hlat']
    AWS14_melt_day = AWS14_SEB_day_mean['melt_energy']
    Model_SEB_flight_AWS14 = np.mean(run['LW_net'][times[0]:times[1],(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data) + \
                         np.mean(run['SW_net'][times[0]:times[1],(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data) + \
                         np.mean(run['SH'][times[0]:times[1],(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)]) + \
                         np.mean(run['LH'][times[0]:times[1],(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)])
    Model_SEB_day_AWS14 = np.mean(run['LW_net'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data) + \
                             np.mean(run['SW_net'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data) + \
                          np.mean(run['SH'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)]) + \
                           np.mean(run['LH'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)])
    Model_SEB_flight_AWS15 = np.mean(run['LW_net'][times[0]:times[1],(AWS15_lon-1):(AWS15_lon+1), (AWS15_lat-1):(AWS15_lat+1)].data) + \
                             np.mean(run['SW_net'][times[0]:times[1],(AWS15_lon-1):(AWS15_lon+1), (AWS15_lat-1):(AWS15_lat+1)].data) + \
                             np.mean(run['SH'][times[0]:times[1],(AWS15_lon-1):(AWS15_lon+1), (AWS15_lat-1):(AWS15_lat+1)]) + \
                             np.mean(run['LH'][times[0]:times[1],(AWS15_lon-1):(AWS15_lon+1), (AWS15_lat-1):(AWS15_lat+1)])
    Model_SEB_day_AWS15 = np.mean(run['LW_net'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data) + \
                             np.mean(run['SW_net'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data) + \
                          np.mean(run['SH'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)]) + \
                           np.mean(run['LH'][:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)])
    Time = run['LW_net'].coord('time')
    # Time = Time + 0.5 # model data = hourly means, centred on the half-hour, so account for this
    Model_time = Time.units.num2date(Time.points)
    melt_masked_day = np.ma.masked_where(run['Ts'] < -0.025, Model_SEB_day_AWS14)
    #melt_masked_day = melt_masked_day.clip(min=0)
    melt_masked_flight = np.ma.masked_where(run['Ts']  < -0.025, Model_SEB_flight_AWS14)
    return Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15, melt_masked_day, melt_masked_flight, AWS14_SEB_flight, AWS14_SEB_day, AWS14_melt_flight, AWS14_melt_day



## ------------------------------------------- CALCULATE BIASES ----------------------------------------------------- ##
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
        AWS14_bias.append(melt_masked_day - AWS14_SEB_day_mean['melt_energy'])
    elif day == False:
        AWS14_bias.append(melt_masked_flight - AWS14_SEB_flight_mean['melt_energy'])
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

Model_SEB_day_AWS14, Model_SEB_day_AWS15, Model_SEB_flight_AWS14, Model_SEB_flight_AWS15, melt_masked_day, melt_masked_flight, \
obs_SEB_AWS14_flight,  obs_SEB_AWS14_day, obs_melt_AWS14_flight, obs_melt_AWS14_day = calc_SEB(RA1T_vars, times = (47,95))

AWS14_bias, AWS15_bias = calc_bias(RA1T_vars, times = (47,95), day = False)

print AWS14_bias, AWS15_bias

## -------------------------------------------------- PLOTTING ------------------------------------------------------ ##

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
        [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
        ax.set_ylim(-100, 300)
    # Plot observed SEB
    axs[0].plot(AWS14_SEB_Jan['Time'], AWS14_SEB_Jan['SWnet_corr'], color='#6fb0d2', lw=5, label='Net shortwave flux')
    axs[0].plot(AWS14_SEB_Jan['Time'], AWS14_SEB_Jan['LWnet_corr'], color='#86ad63', lw=5, label='Net longwave flux')
    axs[0].plot(AWS14_SEB_Jan['Time'], AWS14_SEB_Jan['Hsen'], color='#1f78b4', lw=5, label='Sensible heat flux')
    axs[0].plot(AWS14_SEB_Jan['Time'], AWS14_SEB_Jan['Hlat'], color='#33a02c', lw=5, label='Latent heat flux')
    axs[0].plot(AWS14_SEB_Jan['Time'], AWS14_SEB_Jan['melt_energy'], color='#f68080', lw=5, label='Melt flux')
    #axs[0].axes.get_xaxis().set_visible(False)
    axs[0].set_xlim(AWS14_SEB_Jan['Time'].values[0], AWS14_SEB_Jan['Time'].values[-1])
    axs[0].axvspan(18.614583, 18.7083, edgecolor='dimgrey', facecolor='dimgrey', alpha=0.5)
    # Plot model SEB
    axs[1].set_xlim(Model_time[48], Model_time[-1])
    LH = 0 - run['LH'].data
    SH = 0- run['SH'].data
    axs[1].plot(Model_time[48:], np.mean(run['SW_net'][48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1, 2)), color='#6fb0d2', lw=5, label='Net shortwave flux')
    axs[1].plot(Model_time[48:], np.mean(run['LW_net'][48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)].data, axis = (1, 2)), color='#86ad63', lw=5, label='Net longwave flux')
    axs[1].plot(Model_time[48:], np.mean(SH[48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1, 2)), color='#1f78b4', lw=5, label='Sensible heat flux')
    axs[1].plot(Model_time[48:], np.mean(LH[48:,(AWS14_lon-1):(AWS14_lon+1), (AWS14_lat-1):(AWS14_lat+1)], axis = (1, 2)), color='#33a02c', lw=5, label='Latent heat flux')
    axs[1].plot(Model_time[48:], melt_day[48:], color='#f68080', lw=5, label='Melt flux')
    axs[1].axvspan(Model_time[59], Model_time[68], edgecolor='dimgrey', facecolor='dimgrey', alpha=0.5)
    lgd = plt.legend(fontsize=36, bbox_to_anchor = (1.15, 1.3))
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    #axs[0].text(x=AWS14_SEB_Jan['Time'].values[2], y=250, s='a', fontsize=44, fontweight='bold', color='dimgrey')
    #axs[1].text(x=Model_time[1], y=250, s='b', fontsize=32, fontweight='bold', color='dimgrey')
    plt.setp(axs[0].get_yticklabels()[-2], visible=False)
    plt.setp(axs[1].get_yticklabels()[-2], visible=False)
    plt.subplots_adjust(left=0.22, top = 0.95, bottom=0.1, right=0.9, hspace = 0.1, wspace = 0.1)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_SEB__RA1M_mod.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/AWS14_SEB_RA1M_mod.png', transparent=True)
    plt.show()

#total_SEB(RA1M_mod_vars)

def rad_time_srs():
    model_runs = [RA1M_vars, RA1M_mod_vars]#, RA1T_vars, RA1T_mod_vars]#, CASIM_vars]#[RA1M_mod_vars]#
    fig, ax = plt.subplots(len(model_runs),2, sharex='col', figsize=(16,len(model_runs*5)+3))#(1,2, figsize = (18,8))##, squeeze=False)#
    ax = ax.flatten()
    ax2 = np.empty_like(ax)
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        #[l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        axs.axvspan(14.75, 17, edgecolor = 'dimgrey', facecolor='dimgrey', alpha=0.5)
    def my_fmt(x,p):
        return {0}.format(x) + ':00'
    plot = 0
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
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
        ax[plot].plot(run['LW_down'].coord('time').points, np.mean(run['SW_down'][:,161:163, 182:184].data, axis = (1,2)), label='AWS15, modelled', linewidth=3, linestyle='--',color='darkblue')
        ax2[plot].plot(AWS15_Jan['Hour'], AWS15_Jan['Sin'], label='AWS15, observed', linewidth=3, color='darkblue')
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
        mod15 = ax[plot+1].plot(run['LW_down'].coord('time').points,np.mean(run['LW_down'][:,161:163, 182:184].data, axis = (1,2)), label = 'AWS15, modelled', linewidth = 3, linestyle = '--', color = 'darkblue') ##
        obs14 = ax2[plot+1].plot(AWS14_Jan['Hour'], AWS14_Jan['Lin'], label='AWS14, observed', linewidth=3, color='darkred') ##
        obs15 = ax2[plot+1].plot(AWS15_Jan['Hour'], AWS15_Jan['Lin'], label='AWS15, observed', linewidth=3, color='darkblue') ##
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
    fig.text(0.08, 0.55, 'Downwelling \nshortwave \nflux \n(W m$^{-2}$)', fontsize=30, ha= 'center', va='center', rotation = 0, color = 'dimgrey')
    fig.text(0.92, 0.55, 'Downwelling \nlongwave \nflux \n(W m$^{-2}$)', fontsize=30, ha='center', va='center', color = 'dimgrey', rotation=0)
    plt.subplots_adjust(left=0.22, bottom=0.35, right=0.78, top=0.97, wspace=0.15, hspace=0.15)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/vn11_SEB_time_srs_LWd_SWd_RA1M_v_RA1M_mod.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/f152/Radiation/vn11_SEB_time_srs_LWd_SWd_RA1M_v_RA1M_mod.eps', transparent = True)
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



correl_plot()
