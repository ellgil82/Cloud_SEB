## -------------------------------- LOAD AND PLOT MONTH-LONG TIME SERIES OF MODEL DATA ----------------------------------- ##

host = 'bsl'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import iris
import os
import fnmatch
import scipy
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
from matplotlib import rcParams
from matplotlib.lines import Line2D
import sys
if host == 'jasmin':
    sys.path.append('/group_workspaces/jasmin4/bas_climate/users/ellgil82/scripts/Tools/')
elif host == 'bsl':
    sys.path.append('/users/ellgil82/scripts/Tools/')

from tools import compose_date, compose_time, find_gridbox
from find_gridbox import find_gridbox
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap
import time
from sklearn.metrics import mean_squared_error
import datetime

if host == 'jasmin':
    os.chdir('/group_workspaces/jasmin4/bas_climate/users/ellgil82/OFCAP/netcdfs/')
elif host == 'bsl':
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/netcdfs/')

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans', 'Verdana']


def load_t_srs(var, domain):
    orog = iris.load_cube('OFCAP_orog.nc', 'surface_altitude')
    lsm = iris.load_cube('OFCAP_lsm.nc', 'land_binary_mask')
    orog = orog[0,0,:,:]
    lsm = lsm[0,0,:,:]
    for i in [orog, lsm]:
        real_lat, real_lon = rotate_data(i, 0,1)
    if var == 'met':
        # Load daily mean files in
        Tair = iris.load_cube('../netcdfs/OFCAP_Tair.nc', 'air_temperature')
        Ts = iris.load_cube('../netcdfs/OFCAP_Ts.nc', 'surface_temperature')
        q = iris.load_cube('../netcdfs/OFCAP_q.nc', 'specific_humidity') #?
        MSLP = iris.load_cube('../netcdfs/OFCAP_MSLP.nc', 'air_pressure_at_sea_level')
        u = iris.load_cube('../netcdfs/OFCAP_u_wind.nc', 'eastward_wind')
        v = iris.load_cube('../netcdfs/OFCAP_v_wind.nc', 'northward_wind')
        v = v[:,:, 1:, :]
        Model_time = u.coord('t')[:-1]
        Model_time = Model_time.units.num2date(Model_time.points)
        var_list = [ Tair, Ts, q, MSLP, u, v]
        for i in var_list:
            real_lat, real_lon = rotate_data(i, 2,3)
        if domain == 'ice shelf' or domain == 'Larsen C':
            # Create Larsen mask
            Larsen_mask = np.zeros((400, 400))
            lsm_subset = lsm.data[35:260, 90:230]
            Larsen_mask[35:260, 90:230] = lsm_subset
            Larsen_mask[orog.data > 25 ] = 0
            Larsen_mask = np.logical_not(Larsen_mask)
            Tair_srs = np.ma.masked_array(Tair[:-1,0,:,:].data, mask = np.broadcast_to(Larsen_mask, Tair[:,0,:,:].shape)).mean(axis = (1,2))
            Ts_srs = np.ma.masked_array(Ts[:-1,0,:,:].data, mask = np.broadcast_to(Larsen_mask, Ts[:,0,:,:].shape)).mean(axis = (1,2))
            q_srs = np.ma.masked_array(q[:-1,0,:,:].data, mask = np.broadcast_to(Larsen_mask, q[:,0,:,:].shape)).mean(axis = (1,2))
            u_srs = np.ma.masked_array(u[:-1,0,:,:].data, mask = np.broadcast_to(Larsen_mask, u[:,0,:,:].shape)).mean(axis = (1,2))
            v_srs = np.ma.masked_array(v[:-1,0,:,:].data, mask = np.broadcast_to(Larsen_mask, v[:,0,:,:].shape)).mean(axis = (1,2))
            ff_srs = np.sqrt((u_srs ** 2) + (v_srs ** 2))
        elif domain == 'AWS 14' or domain == 'AWS14':
            Tair_srs = np.mean(Tair[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            Ts_srs = np.mean(Ts[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            q_srs = np.mean(q[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            MSLP_srs = np.mean(MSLP[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            u_srs = np.mean(u[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            v_srs = np.mean(v[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            ff_srs = np.sqrt((u_srs ** 2) + (v_srs ** 2))
        elif domain == 'AWS 15' or domain == 'AWS15':
            Tair_srs = np.mean(Tair[:-1, 0, 161:163, 182:184].data, axis=(1, 2))
            Ts_srs = np.mean(Ts[:-1, 0, 161:163, 182:184].data, axis=(1, 2))
            q_srs = np.mean(q[:-1, 0, 161:163, 182:184].data, axis=(1, 2))
            MSLP_srs = np.mean(q[:-1, 0, 161:163, 182:184].data, axis=(1, 2))
            u_srs = np.mean(u[:-1, 0, 161:163, 182:184].data, axis=(1, 2))
            v_srs = np.mean(v[:-1, 0, 161:163, 182:184].data, axis=(1, 2))
            ff_srs = np.sqrt((u_srs**2)+(v_srs**2))
        var_dict = {'Tair_srs': Tair_srs-273.15, 'Ts_srs': Ts_srs-273.15, 'u_srs': u_srs, 'v_srs': v_srs, 'q_srs': q_srs,
                    'ff_srs': ff_srs, 'MSLP_srs': MSLP_srs, 'orog': orog, 'lsm': lsm, 'Model_time': Model_time}
    elif var == 'SEB':
        daymn_SWdown = iris.load_cube('OFCAP_SWdown.nc', 'surface_downwelling_shortwave_flux_in_air')
        daymn_SWup = iris.load_cube('OFCAP_SWup.nc', 'Net short wave radiation flux')
        daymn_SWnet = iris.load_cube('OFCAP_SWnet.nc', 'Net short wave radiation flux')
        daymn_LWnet = iris.load_cube('OFCAP_LWnet.nc', 'surface_net_downward_longwave_flux')
        daymn_LWdown = iris.load_cube('OFCAP_LWdown.nc', 'IR down')
        daymn_LWup = iris.load_cube('OFCAP_LWup.nc', 'surface_net_downward_longwave_flux')
        daymn_HL = iris.load_cube('OFCAP_HL.nc', 'Latent heat flux')
        daymn_HS = iris.load_cube('OFCAP_HS.nc', 'surface_upward_sensible_heat_flux') #../netcdfs/
        Ts = iris.load_cube('OFCAP_Ts.nc', 'surface_temperature')
        Ts.convert_units('celsius')
        Etot = iris.load_cube('OFCAP_Etot.nc', 'Net short wave radiation flux')
        Model_time = Etot.coord('time')[:-1]
        Model_time = Model_time.units.num2date(Model_time.points)
        if domain == 'ice shelf' or domain == 'Larsen C':
            # Create Larsen mask
            Larsen_mask = np.zeros((400, 400))
            lsm_subset = lsm.data[35:260, 90:230]
            Larsen_mask[35:260, 90:230] = lsm_subset
            Larsen_mask[orog.data > 25] = 0
            Larsen_mask = np.logical_not(Larsen_mask)
            SWdown_srs = np.ma.masked_array(daymn_SWdown[-1:, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, daymn_SWdown[:-1, 0, :, :].shape)).mean(axis=(1, 2))
            SWup_srs = np.ma.masked_array(daymn_SWup[:-1, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, daymn_SWup[:-1, 0, :, :].shape)).mean(axis=(1, 2))
            SWnet_srs = np.ma.masked_array(daymn_SWnet[:-1, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, daymn_SWnet[:-1, 0, :, :].shape)).mean(axis=(1, 2))
            LWnet_srs = np.ma.masked_array(daymn_LWnet[:-1, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, daymn_LWnet[:-1, 0, :, :].shape)).mean(axis=(1, 2))
            LWdown_srs = np.ma.masked_array(daymn_LWdown[:-1, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, daymn_LWdown[:-1, 0, :, :].shape)).mean(axis=(1, 2))
            LWup_srs = np.ma.masked_array(daymn_LWup[:-1, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, daymn_LWup[:-1, 0, :, :].shape)).mean(axis=(1, 2))
            HL_srs = np.ma.masked_array(daymn_HL[:-1, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, daymn_HL[:-1, 0, :, :].shape)).mean(axis=(1, 2))
            HS_srs = np.ma.masked_array(daymn_HS[:-1, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, daymn_HS[:-1, 0, :, :].shape)).mean(axis=(1, 2))
            Etot_srs = np.ma.masked_array(Etot[:-1, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, Etot[:-1, 0, :, :].shape)).mean(axis=(1, 2))
            Ts_srs = np.ma.masked_array(Ts[:-1, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, Ts[:-1, 0, :, :].shape)).mean(axis=(1, 2))
            melt_srs = Etot_srs
            melt_srs[Ts_srs < -0.025] = 0
        elif domain == 'AWS 14' or domain == 'AWS14':
            SWnet_srs = np.mean(daymn_SWnet[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            SWdown_srs = np.mean(daymn_SWdown[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            SWup_srs = np.mean(daymn_SWup[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            LWnet_srs = np.mean(daymn_LWnet[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            LWdown_srs = np.mean(daymn_LWdown[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            LWup_srs = np.mean(daymn_LWup[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            HL_srs = np.mean(daymn_HL[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            HS_srs = np.mean(daymn_HS[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            Etot_srs = np.mean(Etot[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            Ts_srs = np.mean(Ts[:-1, 0, 199:201, 199:201].data, axis=(1, 2))
            melt_srs = Etot_srs
            melt_srs[Ts_srs < -0.025] = 0
        elif domain == 'AWS 15' or domain == 'AWS15':
            SWnet_srs = np.mean(daymn_SWnet[:-1, 0,161:163, 182:184].data, axis=(1, 2))
            SWdown_srs = np.mean(daymn_SWdown[:-1, 0,161:163, 182:184].data, axis=(1, 2))
            SWup_srs = np.mean(daymn_SWup[:-1, 0,161:163, 182:184].data, axis=(1, 2))
            LWnet_srs = np.mean(daymn_LWnet[:-1, 0,161:163, 182:184].data, axis=(1, 2))
            LWdown_srs = np.mean(daymn_LWdown[:-1, 0,161:163, 182:184].data, axis=(1, 2))
            LWup_srs = np.mean(daymn_LWup[:-1, 0,161:163, 182:184].data, axis=(1, 2))
            HL_srs = np.mean(daymn_HL[:-1, 0,161:163, 182:184].data, axis=(1, 2))
            HS_srs = np.mean(daymn_HS[:-1, 0,161:163, 182:184].data, axis=(1, 2))
            Etot_srs = np.mean(Etot[:-1, 0, 161:163, 182:184].data, axis=(1, 2))
            Ts_srs = np.mean(Ts[:-1, 0, 161:163, 182:184].data, axis=(1, 2))
            melt_srs = Etot_srs
            melt_srs[Ts_srs < -0.025] = 0
        HL_srs = 0 - HL_srs
        HS_srs = 0 - HS_srs
        var_dict = {'SWnet_srs': SWnet_srs, 'SWdown_srs': SWdown_srs, 'SWup_srs': SWup_srs, 'LWnet_srs': LWnet_srs,
                    'LWdown_srs': LWdown_srs, 'LWup_srs': LWup_srs, 'HL_srs': HL_srs, 'HS_srs': HS_srs, 'Etot_srs': Etot_srs,
                    'melt_srs': melt_srs, 'orog': orog, 'lsm': lsm, 'Ts_srs': Ts_srs, 'Model_time': Model_time}
    return var_dict



SEB_dict = load_t_srs(var = 'SEB', domain = 'AWS14')
#met_dict = load_t_srs(var = 'met', domain = 'AWS14')

def time_srs_plot():
    fig, ax = plt.subplots(figsize = (30,14))
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.spines['right'].set_visible(False)
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]

#mean_obs = pd.read_csv('/data/mac/ellgil82/cloud_data/flights/OFCAP_flight_means.csv')

def plot_profile(var, domain, plot_obs):
    var_dict = {'theta': 'daymn_theta.nc', 'QCL': 'daymn_QCL.nc', 'QCF': 'daymn_QCF.nc', 'q': 'daymn_q.nc', 'Tair': 'daymn_Tair.nc' }
    cubes = iris.load(var_dict[var])
    profile_var = cubes[0]
    orog = iris.load_cube('OFCAP_orog.nc', 'surface_altitude')
    lsm = iris.load_cube('OFCAP_lsm.nc', 'land_binary_mask')
    orog = orog[0,0,:,:]
    lsm = lsm[0,0,:,:]
    if var == 'theta' or var == 'Tair':
        profile_var.convert_units('celsius')
        units = '($^{\circ}$C)'
    elif var == 'QCF' or var == 'QCL' or var == 'q':
        profile_var.convert_units('g kg-1')
        units = '(g kg$^{-1}$)'
    try:
        altitude = profile_var.coord('altitude').points
    except iris.exceptions.CoordinateNotFoundError:
        # Take orography data and use it to create hybrid height factory instance
        auxcoord = iris.coords.AuxCoord(orog.data, standard_name=str(orog.standard_name), long_name="orography", var_name="orog", units=orog.units)
        profile_var.add_aux_coord(auxcoord, (np.ndim(profile_var) - 2, np.ndim(profile_var) - 1))
        profile_var.coord('Hybrid height').convert_units('metres')
        factory = iris.aux_factory.HybridHeightFactory(delta=profile_var.coord("Hybrid height"), orography=profile_var.coord("surface_altitude"))
        profile_var.add_aux_factory(factory)  # this should produce a 'derived coordinate', 'altitude' (test this with >>> print theta)
        altitude = profile_var.coord('altitude').points[:, 0,0]
    if domain == 'ice shelf' or domain == 'Larsen C':
        # Create Larsen mask
        Larsen_mask = np.zeros((400, 400))
        lsm_subset = lsm.data[35:260, 90:230]
        Larsen_mask[35:260, 90:230] = lsm_subset
        Larsen_mask[orog.data > 25] = 0
        Larsen_mask = np.logical_not(Larsen_mask)
        profile_mn = np.ma.masked_array(data = profile_var.data, mask = np.broadcast_to(Larsen_mask,profile_var.shape)).mean(axis = (0,2, 3))
    elif domain == 'AWS 14' or domain == 'AWS14':
        profile_mn = np.mean(profile_var[:, :, 199:201, 199:201].data, axis=(0, 2, 3))
    elif domain == 'AWS 15' or domain == 'AWS15':
        profile_mn = np.mean(profile_var[:, :, 161:163, 182:184].data, axis=(0, 2, 3))
    elif domain == 'all':
        # Create Larsen mask
        Larsen_mask = np.zeros((400, 400))
        lsm_subset = lsm.data[35:260, 90:230]
        Larsen_mask[35:260, 90:230] = lsm_subset
        Larsen_mask[orog.data > 25] = 0
        Larsen_mask = np.logical_not(Larsen_mask)
        LCIS_mn = np.ma.masked_array(data=profile_var.data, mask=np.broadcast_to(Larsen_mask, profile_var.shape)).mean(axis=(0, 2, 3))
        AWS14_mn = np.mean(profile_var[:, :, 199:201, 199:201].data, axis=(0, 2, 3))
        AWS15_mn = np.mean(profile_var[:, :, 161:163, 182:184].data, axis=(0, 2, 3))
    fig, ax = plt.subplots(1,1, figsize=(10, 8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax.set_ylim(0, max(altitude/1000.))
    ax.set_xlabel(var + ' ' + units , fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=35)
    ax.set_ylabel('Altitude \n(km)', fontname='SegoeUI semibold', color='dimgrey', rotation = 0, fontsize=28, labelpad=75)
    try:
        ax.plot(profile_mn, altitude/1000., color = '#1f78b4', linestyle = '--', linewidth = 2.5, label = var)
    except:
        LCIS = ax.plot(LCIS_mn, altitude / 1000., color='#1f78b4', linestyle='--', linewidth=2.5, label='Ice shelf')
        AWS14 = ax.plot(AWS14_mn, altitude / 1000., color='darkred', linestyle='--', linewidth=2.5, label='AWS 14')
        AWS15 = ax.plot(AWS15_mn, altitude / 1000., color='darkblue', linestyle='--', linewidth=2.5, label='AWS 15')
        lgd = plt.legend(fontsize=20, markerscale=2)
        for ln in lgd.get_texts():
            plt.setp(ln, color='dimgrey')
        lgd.get_frame().set_linewidth(0.0)
    ax.axes.tick_params(axis = 'both', which = 'both', direction = 'in', length = 5, width = 1.5,  labelsize = 24, pad = 10)
    #ax.tick_params(labelleft = 'off')
    if var == 'QCF' or var == 'QCL':
        ax.set_xlim(0,0.02)
        [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
        if plot_obs == 'yes' or plot_obs == 'True':
            ax.plot(mean_obs[var], altitude/1000., color = 'k', lw = 2.5, label = 'Mean observations')
    if var == 'QCF' or var == 'QCL' or var == 'q':
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(24)
        ax.xaxis.get_offset_text().set_color('dimgrey')
    elif var == 'Tair':
        ax.set_xlim(-35,0)
    elif var == 'theta':
        ax.set_xlim(0,25)
    plt.subplots_adjust(wspace=0.1, bottom=0.23, top=0.95, left=0.25, right=0.9)
    #handles, labels = ax[1].get_legend_handles_labels()
    #handles = [handles[0], handles[1], handles[-1], handles[2],  handles[3] ]
    #labels = [labels[0], labels[1], labels[-1], labels[2], labels[3]]
    #lgd = plt.legend(handles, labels, fontsize=20, markerscale=2)
    plt.savefig('figs/'+domain+'_OFCAP_mean_vertical_profile_'+var+'.eps')
    plt.savefig('figs'+domain+'_OFCAP_mean_vertical_profile_'+var+'.png')
    plt.show()

#for i in ['QCL', 'QCF', 'theta', 'q', 'Tair']:
#    plot_profile(var = i, domain = 'all', plot_obs = 'no')

def plot_diurnal(var, domain):
    orog = iris.load_cube('OFCAP_orog.nc', 'surface_altitude')
    lsm = iris.load_cube('OFCAP_lsm.nc', 'land_binary_mask')
    orog = orog[0,0,:,:]
    lsm = lsm[0,0,:,:]
    # Load diurnal cycles in
    if var == 'SEB':
        diur_SWdown = iris.load_cube('diurnal_SWdown.nc', 'surface_downwelling_shortwave_flux_in_air')
        diur_SWup = iris.load_cube('diurnal_SWup.nc', 'Net short wave radiation flux')
        diur_SWnet = iris.load_cube('diurnal_SWnet.nc', 'Net short wave radiation flux')
        diur_LWnet = iris.load_cube('diurnal_LWnet.nc', 'surface_net_downward_longwave_flux')
        diur_LWdown = iris.load_cube('diurnal_LWdown.nc', 'IR down')
        diur_LWup = iris.load_cube('diurnal_LWup.nc', 'surface_net_downward_longwave_flux')
        diur_HL =iris.load_cube('diurnal_HL.nc', 'Latent heat flux')
        diur_HS = iris.load_cube('diurnal_HS.nc', 'surface_upward_sensible_heat_flux')
        var_dict = {'SWdown': diur_SWdown, 'SWnet': diur_SWnet, 'SWup': diur_SWup, 'LWdown': diur_LWdown, 'LWnet': diur_LWnet, 'LWup': diur_LWup, 'HL': diur_HL, 'HS':  diur_HS}
        colour_dict = {'SWdown': '#6fb0d2', 'SWnet': '#6fb0d2', 'SWup': '#6fb0d2', 'LWdown': '#86ad63', 'LWnet': '#86ad63', 'LWup': '#86ad63','HL': '#33a02c', 'HS': '#1f78b4'}
        UTC_3 = pd.DataFrame()
        for x in var_dict:
            UTC_3[x] = np.concatenate((np.mean(var_dict[x][5:, 0, 199:201, 199:201].data, axis=(1, 2)),
                                       np.mean(var_dict[x][:5, 0, 199:201, 199:201].data, axis=(1, 2))), axis=0)
    elif var == 'met':
        diur_Ts = iris.load_cube('diurnal_Ts.nc','surface_temperature')
        diur_Tair = iris.load_cube('diurnal_Tair.nc', 'air_temperature')
        for i in [diur_Tair, diur_Ts]:
            i.convert_units('celsius')
        diur_u = iris.load_cube('diurnal_u.nc', 'eastward_wind')
        diur_v = iris.load_cube('diurnal_v.nc', 'northward_wind')
        diur_v = diur_v[:,:,1::]
        real_lon, real_lat = rotate_data(diur_v, 2, 3)
        real_lon, real_lat = rotate_data(diur_u, 2, 3)
        diur_q = iris.load_cube('diurnal_q.nc', 'specific_humidity')
        var_dict = {'Ts': diur_Ts, 'Tair': diur_Tair, 'u': diur_u, 'v': diur_v,'q': diur_q}
        colour_dict = {'Ts': '#dd1c77', 'Tair': '#91003f', 'ff': '#238b45', 'q': '#2171b5'}
        if domain == 'ice shelf' or domain == 'Larsen C':
            # Create Larsen mask
            Larsen_mask = np.zeros((400, 400))
            lsm_subset = lsm.data[35:260, 90:230]
            Larsen_mask[35:260, 90:230] = lsm_subset
            Larsen_mask[orog.data > 25] = 0
            Larsen_mask = np.logical_not(Larsen_mask)
            UTC_3 = pd.DataFrame()
            for x in var_dict:
                UTC_3[x] = np.ma.concatenate((np.ma.masked_array(var_dict[x][5:, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, var_dict[x][5:, 0, :, :].shape)).mean(axis=(1, 2)),
                                              np.ma.masked_array(var_dict[x][:5, 0, :, :].data, mask=np.broadcast_to(Larsen_mask, var_dict[x][:5, 0, :, :].shape)).mean(axis=(1, 2))), axis=0)
        elif domain == 'AWS 14' or domain == 'AWS14':
            UTC_3 = pd.DataFrame()
            diur_ff = iris.cube.Cube(data = np.sqrt(var_dict['v'].data**2)+(var_dict['u'].data**2))
            UTC_3['ff'] = np.concatenate((np.mean(diur_ff[5:, 0, 199:201, 199:201].data, axis=(1, 2)),
                                       np.mean(diur_ff[:5, 0, 199:201, 199:201].data, axis=(1, 2))), axis=0)
            for x in var_dict:
                UTC_3[x] = np.concatenate((np.mean(var_dict[x][5:, 0, 199:201, 199:201], axis=(1, 2)),
                                           np.mean(var_dict[x][:5, 0, 199:201, 199:201], axis=(1, 2))), axis=0)
        elif domain == 'AWS 15' or domain == 'AWS15':
            UTC_3 = pd.DataFrame()
            for x in var_dict:
                UTC_3[x] = np.concatenate((np.mean(var_dict[x][5:, 0, 161:163, 182:184].data, axis=(1, 2)),
                                           np.mean(var_dict[x][:5, 0, 161:163, 182:184].data, axis=(1, 2))), axis=0)
            diur_ff = iris.cube.Cube(data=np.sqrt(var_dict['v'].data ** 2) + (var_dict['u'].data ** 2))
            UTC_3['ff'] = np.concatenate((np.mean(diur_ff[5:, 0, 161:163, 182:184].data, axis=(1, 2)),
                                       np.mean(diur_ff[:5, 0,  161:163, 182:184].data, axis=(1, 2))), axis=0)
    ## Set up plotting options
    if var == 'SEB':
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.spines['top'].set_visible(False)
        plt.setp(ax.spines.values(), linewidth=3, color='dimgrey')
        ax.spines['right'].set_visible(False)
        [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
        ax.set_ylabel('Mean energy \nflux (W m$^{-2}$', fontname='SegoeUI semibold', color='dimgrey', rotation=0,fontsize=28, labelpad=75)
        for x in var_dict:
            ax.plot(UTC_3[x].data, color=colour_dict[x], lw=2)
        plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_diurnal_SEB.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_diurnal_SEB.eps', transparent=True)
    elif var == 'met':
        ## Set up plotting options
        fig, ax = plt.subplots(2, 2, sharex = True, figsize=(10, 10))
        ax = ax.flatten()
        plot = 0
        for axs in ax:
            axs.spines['top'].set_visible(False)
            plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
            axs.spines['right'].set_visible(False)
            [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        for x in var_dict:
            ax[plot].plot(UTC_3[x], color = colour_dict[x], lw = 2)
            ax[plot].set_ylabel(x, fontname='SegoeUI semibold', color='dimgrey', rotation=0, fontsize=28, labelpad=75)
            ax[plot].tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
            ax[plot].set_xlabel('Time', fontname='SegoeUI semibold', color='dimgrey', fontsize=28, labelpad=35)
            plot = plot+1
    plt.show()

#plot_diurnal(var = 'SEB', domain = 'AWS14')

def plot_mean_met(contour_var, colour_var, wind_vectors, vector_level = 0):
    orog = iris.load_cube('OFCAP_orog.nc', 'surface_altitude')
    lsm = iris.load_cube('OFCAP_lsm.nc', 'land_binary_mask')
    orog = orog[0, 0, :, :]
    lsm = lsm[0, 0, :, :]
    for i in [orog, lsm]:
        real_lat, real_lon = rotate_data(i, 0, 1)
    # Load daily mean files in
    Tair = iris.load_cube('../netcdfs/OFCAP_Tair.nc', 'air_temperature')
    Ts = iris.load_cube('../netcdfs/OFCAP_Ts.nc', 'surface_temperature')
    q = iris.load_cube('../netcdfs/OFCAP_q.nc', 'specific_humidity')  # ?
    MSLP = iris.load_cube('../netcdfs/OFCAP_MSLP.nc', 'air_pressure_at_sea_level')
    MSLP.convert_units('hPa')
    if wind_vectors == 'True' or wind_vectors == 'yes':
        u = iris.load_cube('../netcdfs/OFCAP_u_wind.nc', 'eastward_wind')
        v = iris.load_cube('../netcdfs/OFCAP_v_wind.nc', 'northward_wind')
        v = v[:, :, 1:, :]
        var_list = [Tair, Ts, q, MSLP, u, v]
    else:
        var_list = [Tair, Ts, q, MSLP]
    Model_time = MSLP.coord('t')[:-1]
    Model_time = Model_time.units.num2date(Model_time.points)
    for i in var_list:
        real_lat, real_lon = rotate_data(i, 2, 3)
    # Calculate means
    Tair = np.mean(Tair[:,0,:,:].data, axis=0)
    MSLP = np.mean(MSLP[:,0,:,:].data,axis = 0)
    Ts = np.mean(Ts[:,0,:,:].data, axis=0)
    q = np.mean(q[:, vector_level, :, :].data,axis=0)
    if wind_vectors == 'True' or wind_vectors == 'yes':
        u = np.mean(u[:, vector_level, :, :].data,axis=0)
        v = np.mean(v[:, vector_level, :, :].data,axis=0)
    var_dict = {'MSLP': MSLP, 'Tair': Tair, 'Ts': Ts, 'q': q}
    label_dict = {'Tair': '1.5 m air temperature ($^{\circ}$C)', 'MSLP': 'Mean sea level pressure (hPa)', 'q': 'Specific humidity (g kg$^{-1}$)', 'Ts': 'Surface temperature ($^{\circ}$C)'}
    ## Start plotting
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_axes([0.18, 0.25, 0.75, 0.63], frameon=False)  # , projection=ccrs.PlateCarree())#
    ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
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
    plt.xticks(XTicks, XTickLabels)
    ax.set_xlim(PlotLonMin, PlotLonMax)
    ax.tick_params(which='both', pad=10, labelsize=34, color='dimgrey')
    YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
    YTickLabels = [None] * len(YTicks)
    for i, YTick in enumerate(YTicks):
        if YTick < 0:
            YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
        else:
            YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
    plt.yticks(YTicks, YTickLabels)
    ax.set_ylim(PlotLatMin, PlotLatMax)
    plt.contour(lsm.coord('longitude').points, lsm.coord('latitude').points, lsm.data, colors='#757676', linewidths=2.5,zorder=2)  # , transform=RH.coord('grid_longitude').coord_system.as_cartopy_projection())
    plt.contour(orog.coord('longitude').points, orog.coord('latitude').points, orog.data, colors='#757676',levels=[15], linewidths=2.5, zorder=3)
    P_lev = plt.contour(lsm.coord('longitude').points, lsm.coord('latitude').points, var_dict[contour_var], colors='#222222', linewidths=3, levels=range(960, 1020, 4), zorder=4)
    ax.clabel(P_lev, v=[960, 968, 976, 982, 990], inline=True, inline_spacing=3, fontsize=28, fmt='%1.0f')
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-30, max_val=10, name='bwr_zero', var=var_dict[colour_var], start = 0., stop = 1.)
    c = ax.pcolormesh(real_lon, real_lat, var_dict[colour_var], cmap=bwr_zero, vmin=-30, vmax=10, zorder=1)  #
    CBarXTicks = [-30, -10, 10]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.25, 0.15, 0.6, 0.03])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks)  #
    CBar.set_label(label_dict[colour_var], fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0,tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    if wind_vectors == 'True' or wind_vectors == 'yes':
        x, y = np.meshgrid(real_lon, real_lat)
        quiver = ax.quiver(x[::25, ::25], y[::25, ::25], u.data[::25, ::25], v.data[::25, ::25], color='#414345', pivot='middle', scale=100, zorder=5)
        plt.quiverkey(quiver, 0.25, 0.9, 10, r'$10$ $m$ $s^{-1}$', labelpos='N', color='#414345', labelcolor='#414345',fontproperties={'size': '32', 'weight': 'bold'},coordinates='figure', )
    plt.draw()
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_met.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_met.eps', transparent=True)
    plt.show()


#plot_mean_met(contour_var='MSLP', colour_var='Ts', wind_vectors=True, vector_level = 0 )


def correlations():
    # Load data
    corr_cl_melt = iris.load_cube('timcor_cl_melt.nc', '')
    corr_LWd_LWP = iris.load_cube('timcor_LWd_v_LWP.nc')
    corr_SWd_IWP = iris.load_cube('timcor_SWd_v_IWP.nc')


def load_AWS(station):
    ## --------------------------------------------- SET UP VARIABLES ------------------------------------------------##
    ## Load data from AWS 14 and AWS 15 for January 2011
    print('\nDayum grrrl, you got a sweet AWS...')
    os.chdir('/data/clivarm/wip/ellgil82/AWS/')
    for file in os.listdir('/data/clivarm/wip/ellgil82/AWS/'):
        if fnmatch.fnmatch(file, '%(station)s*' % locals()):
            AWS_srs = pd.read_csv(str(file), header = 0)
            print(AWS_srs.shape)
    # Calculate date, given list of years and day of year
    date_list = compose_date(AWS_srs['year'], days=AWS_srs['day'])
    AWS_srs['Date'] = date_list
    # Set date as index
    AWS_srs.index = AWS_srs['Date']
    # Calculate actual time from decimal DOY (seriously, what even IS that format?)
    AWS_srs['time'] = 24*(AWS_srs['Time'] - AWS_srs['day'])
    case = AWS_srs.loc['2011-01-01':'2011-02-07'] #'2015-01-01':'2015-12-31'
    print '\nconverting times...'
    # Convert times so that they can be plotted
    time_list = []
    for i in case['time']:
        hrs = int(i)                 # will now be 1 (hour)
        mins = int((i-hrs)*60)       # will now be 4 minutes
        secs = int(0 - hrs*60*60 + mins*60) # will now be 30
        j = datetime.time(hour = hrs, minute=mins)
        time_list.append(j)
    case['Time'] = time_list
    case['datetime'] = case.apply(lambda r : pd.datetime.combine(r['Date'],r['Time']),1)
    case['E'] = case['LWnet_corr'] + case['SWnet_corr'] + case['Hlat'] + case['Hsen'] - case['Gs']
    if host == 'jasmin':
        os.chdir('/group_workspaces/jasmin4/bas_climate/users/ellgil82/OFCAP/proc_data/')
    elif host == 'bsl':
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/proc_data/')
    return case

AWS14_SEB = load_AWS('AWS14_SEB_2009-2017_norp')
# Trim down to match model times (t+12 forecasts)
AWS14_SEB = AWS14_SEB[12:-1]

def calc_BL_bias():
    AWS_var = load_AWS('AWS14_SEB_2009-2017_norp')
    AWS_var = AWS_var[12:-1]
    obs = [AWS_var['SWin_corr'], AWS_var['LWin'], AWS_var['SWnet_corr'], AWS_var['LWnet_corr'], AWS_var['Hsen'], AWS_var['Hlat'], AWS_var['E'], AWS_var['melt_energy']]
    BL_mod = [BL_SEB_dict['SWdown_srs'], BL_SEB_dict['LWdown_srs'], BL_SEB_dict['SWnet_srs'], BL_SEB_dict['LWnet_srs'],  BL_SEB_dict['HS_srs'],  BL_SEB_dict['HL_srs'], BL_SEB_dict['Etot_srs'], BL_SEB_dict['melt_srs']]
    ctrl_mod = [SEB_dict['SWdown_srs'], SEB_dict['LWdown_srs'], SEB_dict['SWnet_srs'], SEB_dict['LWnet_srs'],  SEB_dict['HS_srs'],  SEB_dict['HL_srs'], SEB_dict['Etot_srs'], SEB_dict['melt_srs']]
    mean_obs = []
    mean_mod = []
    bias = []
    errors = []
    r2s = []
    rmses = []
    for i in np.arange(len(BL_mod)):
        b = BL_mod[i] - obs[i]
        errors.append(b)
        mean_obs.append(np.mean(obs[i]))
        mean_mod.append(np.mean(BL_mod[i]))
        bias.append(mean_mod[i] - mean_obs[i])
        slope, intercept, r2, p, sterr = scipy.stats.linregress(obs[i], BL_mod[i])
        r2s.append(r2)
        rmse = mean_squared_error(y_true = obs[i], y_pred = BL_mod[i])
        rmses.append(rmse)
        idx = ['SWd', 'LWd', 'SWn', 'LWn', 'SH', 'LH', 'total', 'melt']
    df = pd.DataFrame(index = idx)
    df['obs mean'] = pd.Series(mean_obs, index = idx)
    df['mod mean'] = pd.Series(mean_mod, index = idx)
    df['bias'] =pd.Series(bias, index=idx)
    df['rmse'] = pd.Series(rmses, index = idx)
    df['% RMSE'] = ( df['rmse']/df['obs mean'] ) * 100
    df['correl'] = pd.Series(r2s, index = idx)
    df.to_csv('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/BL_run/OFCAP_BL_errors.csv')
    print(df)

calc_BL_bias()

def calc_bias():
    # Calculate bias of time series
    # Forecast error
    AWS_var = load_AWS('AWS14_SEB_2009-2017_norp')
    AWS_var = AWS_var[12:-1]
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/netcdfs/')
    SEB_dict = load_t_srs(var='SEB', domain='AWS14')
    met_dict = load_t_srs(var = 'met', domain = 'AWS14')
    surf_met_obs = [AWS_var['Tsobs'], AWS_var['Tair_2m'], AWS_var['qair_2m'], AWS_var['FF_10m'], AWS_var['SWin_corr'], AWS_var['LWin'], AWS_var['SWnet_corr'], AWS_var['LWnet_corr'], AWS_var['Hsen'], AWS_var['Hlat'], AWS_var['E'], AWS_var['melt_energy']]#, AWS_var['melt_energy']]
    surf_mod = [met_dict['Ts_srs'], met_dict['Tair_srs'], met_dict['q_srs'], met_dict['ff_srs'], SEB_dict['SWdown_srs'], SEB_dict['LWdown_srs'], SEB_dict['SWnet_srs'], SEB_dict['LWnet_srs'],  SEB_dict['HS_srs'],  SEB_dict['HL_srs'], SEB_dict['Etot_srs'], SEB_dict['melt_srs']]#, SEB_1p5['melt_forced']]
    mean_obs = []
    mean_mod = []
    bias = []
    errors = []
    r2s = []
    rmses = []
    for i in np.arange(len(surf_met_obs)):
        b = surf_mod[i] - surf_met_obs[i]
        errors.append(b)
        mean_obs.append(np.mean(surf_met_obs[i]))
        mean_mod.append(np.mean(surf_mod[i]))
        bias.append(mean_mod[i] - mean_obs[i])
        slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_met_obs[i], surf_mod[i])
        r2s.append(r2)
        rmse = mean_squared_error(y_true = surf_met_obs[i], y_pred = surf_mod[i])
        rmses.append(rmse)
        idx = ['Ts', 'Tair', 'RH', 'wind', 'SWd', 'LWd', 'SWn', 'LWn', 'SH', 'LH', 'total', 'melt']#, 'melt forced']
    df = pd.DataFrame(index = idx)
    df['obs mean'] = pd.Series(mean_obs, index = idx)
    df['mod mean'] = pd.Series(mean_mod, index = idx)
    df['bias'] =pd.Series(bias, index=idx)
    df['rmse'] = pd.Series(rmses, index = idx)
    df['% RMSE'] = ( df['rmse']/df['obs mean'] ) * 100
    df['correl'] = pd.Series(r2s, index = idx)
    for i in range(len(surf_mod)):
        slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_met_obs[i], surf_mod[i])
        print(idx[i])
        print('\nr2 = %s\n' % r2)
    print('RMSE/bias = \n\n\n')
    df.to_csv('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/OFCAP_bias_and_RMSE.csv')
    print(df)

calc_bias()

AWS_var = load_AWS('AWS14_SEB_2009-2017_norp')
AWS_var = AWS_var[12:-1]
obs = [AWS_var['SWin_corr'], AWS_var['LWin'], AWS_var['SWnet_corr'], AWS_var['LWnet_corr'], AWS_var['Hsen'], AWS_var['Hlat'], AWS_var['E'], AWS_var['melt_energy']]
os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/BL_run/')
BL_SEB_dict = load_t_srs(var = 'SEB', domain = 'AWS14')
BL_mod = [BL_SEB_dict['SWdown_srs'], BL_SEB_dict['LWdown_srs'], BL_SEB_dict['SWnet_srs'], BL_SEB_dict['LWnet_srs'], BL_SEB_dict['HS_srs'], BL_SEB_dict['HL_srs'], BL_SEB_dict['Etot_srs'], BL_SEB_dict['melt_srs']]
ctrl_mod = [SEB_dict['SWdown_srs'], SEB_dict['LWdown_srs'], SEB_dict['SWnet_srs'], SEB_dict['LWnet_srs'], SEB_dict['HS_srs'], SEB_dict['HL_srs'], SEB_dict['Etot_srs'], SEB_dict['melt_srs']]

fig, ax = plt.subplots(4,2)
ax = ax.flatten()
for i in np.arange(len(BL_mod)):
    #ax[i].plot(AWS_var['datetime'], obs[i], color = 'k', lw = 2, label = 'Observed')
    ax[i].plot(BL_SEB_dict['Model_time'], BL_mod[i], color = '#33a02c',  linestyle = '--', lw = 2, label = 'BL run')
    ax[i].plot(SEB_dict['Model_time'], ctrl_mod[i], color = '#1f78b4',  linestyle = '--', lw = 2, label = 'ctrl run')

plt.legend()
plt.show()


def melt_plot():
    fig, ax = plt.subplots(figsize = (18,8))
    ax2 = ax.twiny()
    ax.plot(SEB_dict['Model_time'], SEB_dict['melt_srs'][:-1], lw = 2, color = '#f68080', label = 'modelled melt flux', zorder = 1)
    ax2.plot(AWS14_SEB['datetime'], AWS14_SEB['melt_energy'], lw=2, color='k', label='observed melt flux', zorder = 2)
    ax2.set_xlim(AWS14_SEB['datetime'][0], AWS14_SEB['datetime'][-1])
    ax.set_xlim(SEB_dict['Model_time'][0], SEB_dict['Model_time'][-1])
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%d %b')
    ax.set_ylim(0, 200)
    ax.set_ylabel('Melt flux\n(W m$^{-2}$)', rotation = 0, fontsize = 36, labelpad = 100, color = 'dimgrey')
    ax2.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
    ax.tick_params(axis='both', which='both', labelsize=36, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    ax.xaxis.set_major_formatter(dayfmt)
    #Legend
    lns = [Line2D([0],[0], color='k', linewidth = 2.5),
           Line2D([0],[0], color =  '#f68080', linewidth = 2.5)]
    labs = ['Observed melt flux', 'Modelled melt flux']
    lgd = ax2.legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=28)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left = 0.22, right = 0.95)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_melt.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_melt.eps', transparent=True)
    plt.show()

#melt_plot()

def flux_plot():
    fig, ax = plt.subplots(figsize = (18,8), sharex = True)
    ax2 = ax.twiny()
    ax.plot(SEB_dict['Model_time'], SEB_dict['HL_srs'][:-1], lw = 2, color = '#33a02c',  zorder = 1)
    ax2.plot(AWS14_SEB['datetime'], AWS14_SEB['Hlat'], lw=2, color='k', zorder = 2)
    ax2.set_xlim(AWS14_SEB['datetime'][0], AWS14_SEB['datetime'][-1])
    ax.set_xlim(SEB_dict['Model_time'][0], SEB_dict['Model_time'][-1])
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%d %b')
    ax.set_ylim(-50,50)
    ax.set_ylabel('Energy flux\n(W m$^{-2}$)', rotation = 0, fontsize = 36, labelpad = 100, color = 'dimgrey')
    ax2.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
    ax.tick_params(axis='both', which='both', labelsize=36, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    ax.xaxis.set_major_formatter(dayfmt)
    #Legend
    lns = [Line2D([0],[0], color='k', linewidth = 2.5),
           Line2D([0],[0], color =  '#33a02c', linewidth = 2.5)]
    labs = ['Observed H$_{L}$', 'Modelled H$_{L}$']# ['Observed LW$_{\downarrow}$', 'Modelled LW$_{\downarrow}$']
    lgd = ax2.legend(lns, labs, bbox_to_anchor=(0.65, 1.1), loc=2, fontsize=28)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left = 0.22, right = 0.95)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_HL.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_HL.eps', transparent=True)
    plt.show()

#flux_plot()


def full_SEB():
    fig, ax = plt.subplots(figsize = (18,8), sharex = True)
    ax2 = ax.twiny()
    ax.plot(SEB_dict['Model_time'], SEB_dict['HL_srs'][:-1], lw = 2, color = '#33a02c',  linestyle = '--', zorder = 1)
    ax2.plot(AWS14_SEB['datetime'], AWS14_SEB['Hlat'], lw=2, color= '#33a02c', zorder = 2)
    ax.plot(SEB_dict['Model_time'], SEB_dict['HS_srs'][:-1], lw = 2, color = '#1f78b4',  linestyle = '--', zorder = 1)
    ax2.plot(AWS14_SEB['datetime'], AWS14_SEB['Hsen'], lw=2, color= '#1f78b4', zorder = 2)
    ax.plot(SEB_dict['Model_time'], SEB_dict['LWnet_srs'][:-1], lw = 2, color = '#86ad63',  linestyle = '--', zorder = 1)
    ax2.plot(AWS14_SEB['datetime'], AWS14_SEB['LWnet_corr'], lw=2, color= '#86ad63', zorder = 2)
    ax.plot(SEB_dict['Model_time'], SEB_dict['SWnet_srs'][:-1], lw = 2, color = '#6fb0d2',  linestyle = '--', zorder = 1)
    ax2.plot(AWS14_SEB['datetime'], AWS14_SEB['SWnet_corr'], lw=2, color= '#6fb0d2', zorder = 2)
    ax.plot(SEB_dict['Model_time'], SEB_dict['melt_srs'][:-1], lw = 2, color = '#f68080',  linestyle = '--', zorder = 1)
    ax2.plot(AWS14_SEB['datetime'], AWS14_SEB['melt_energy'], lw=2, color= '#f68080', zorder = 2)
    ax2.set_xlim(AWS14_SEB['datetime'][0], AWS14_SEB['datetime'][-1])
    ax.set_xlim(SEB_dict['Model_time'][0], SEB_dict['Model_time'][-1])
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%d %b')
    ax.set_ylim(-100,200)
    ax.set_ylabel('Energy flux\n(W m$^{-2}$)', rotation = 0, fontsize = 36, labelpad = 100, color = 'dimgrey')
    ax2.axis('off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
    ax.tick_params(axis='both', which='both', labelsize=36, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
    ax.xaxis.set_major_formatter(dayfmt)
    #Legend
    lns = [Line2D([0],[0], color='#6fb0d2', linewidth = 2.5),
           Line2D([0], [0], color='#86ad63', linewidth=2.5),
           Line2D([0], [0], color='#33a02c', linewidth=2.5),
           Line2D([0], [0], color='#1f78b4', linewidth=2.5),
           Line2D([0],[0], color =  '#f68080', linewidth = 2.5)]
    labs = ['$SW_{net}$', '$LW_{net}$', '$H_{L}$', '$H_{L}$', '$E_{melt}$']# ['Observed LW$_{\downarrow}$', 'Modelled LW$_{\downarrow}$']
    lgd = ax2.legend(lns, labs, bbox_to_anchor=(0.65, 1.1), loc=2, fontsize=22)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left = 0.22, right = 0.95)
    if host == 'bsl':
        plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_SEB.png', transparent = True)
        plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_SEB.eps', transparent=True)
    plt.show()

#full_SEB()


def plot_synop(time_idx):
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/proc_data/')
    orog = iris.load_cube('../netcdfs/OFCAP_orog.nc', 'surface_altitude')
    lsm = iris.load_cube('../netcdfs/OFCAP_lsm.nc', 'land_binary_mask')
    orog = orog[0, 0, :, :]
    lsm = lsm[0, 0, :, :]
    for i in [orog, lsm]:
        real_lat, real_lon = rotate_data(i, 0, 1)
    # Load daily mean files in
    Tair = iris.load_cube('daymn_Tair.nc', 'air_temperature')
    Tair.convert_units('celsius')
    Ts = iris.load_cube('daymn_Ts.nc', 'surface_temperature')
    Ts.convert_units('celsius')
    q = iris.load_cube('daymn_q.nc', 'specific_humidity')  # ?
    q.convert_units('g kg-1')
    MSLP = iris.load_cube('daymn_MSLP.nc', 'air_pressure_at_sea_level')
    MSLP.convert_units('hPa')
    u = iris.load_cube('daymn_u.nc', 'eastward_wind')
    v = iris.load_cube('daymn_v.nc', 'northward_wind')
    v = v[:, :, 1:, :]
    for i in [Tair, Ts, q, MSLP, u, v]:
        real_lat, real_lon = rotate_data(i, 2,3)
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_axes([0.18, 0.25, 0.75, 0.63], frameon=False)#, projection=ccrs.PlateCarree())#
    ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
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
    plt.xticks(XTicks, XTickLabels)
    ax.set_xlim(PlotLonMin, PlotLonMax)
    ax.tick_params(which='both', pad=10, labelsize = 34, color = 'dimgrey')
    YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
    YTickLabels = [None] * len(YTicks)
    for i, YTick in enumerate(YTicks):
        if YTick < 0:
            YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
        else:
            YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
    plt.yticks(YTicks, YTickLabels)
    ax.set_ylim(PlotLatMin, PlotLatMax)
    plt.contour(real_lon, real_lat,lsm.data, colors='#757676', linewidths=2.5, zorder=2)#, transform=RH.coord('grid_longitude').coord_system.as_cartopy_projection())
    plt.contour(real_lon, real_lat, orog.data, colors='#757676', levels=[15], linewidths=2.5, zorder=3)
    P_lev = plt.contour(real_lon, real_lat,  MSLP[time_idx, 0,:,:].data, colors='#222222', linewidths=3, levels = range(960,1020,2), zorder=4)
    ax.clabel(P_lev, v = [960,968,976,982,990, 994,998,1002,1006,1010, 1014,1018], inline=True, inline_spacing  = 3, fontsize = 28, fmt = '%1.0f')
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-20., max_val=10., name='bwr_zero', var=Tair[time_idx, 0,:,:].data, start = 0.15, stop = .85)
    c = ax.pcolormesh(real_lon, real_lat, Tair[time_idx, 0,:,:].data, cmap=bwr_zero, vmin=-20., vmax=10., zorder=1)  #
    CBarXTicks = [-20,  -10, 0, 10]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.25, 0.15, 0.6, 0.03])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks)  #
    CBar.set_label('1.5 m air temperature ($^{\circ}$C)', fontsize=34, labelpad=10, color = 'dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    CBar.outline.set_linewidth(2)
    x, y = np.meshgrid(real_lon, real_lat)
    quiver = ax.quiver(x[::50, ::50], y[::50, ::50], u.data[time_idx,24, ::50, ::50], v.data[time_idx,24, ::50, ::50], color = '#414345', pivot='middle', scale=100, zorder = 5)
    plt.quiverkey(quiver, 0.25, 0.9, 10, r'$10$ $m$ $s^{-1}$', labelpos='N', color = '#414345', labelcolor = '#414345',
                  fontproperties={'size': '32',  'weight': 'bold'},
                  coordinates='figure', )
    plt.draw()
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/synop_cond_tstep'+str(time_idx)+ '_2100m.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/synop_cond_tstep'+ str(time_idx) + '_2100m.eps', transparent=True)
    #plt.show()


for j in range(39):
    plot_synop(j)

# Plot some kind of time series of daily mean met vars

# Calculate some averages

# Plot mean diurnal cycle of temps, plus fluxes

# Plot some maps of synoptic conditions; MSLP (contours) + (daily max? mean?) temperature (colours) + u/v winds (vectors)

# Plot time series of downwelling fluxes

# Calculate autocorrelation between (hourly?) timeseries of:
#       a) cloud cover and melt
#       b) SWdown and IWP
#       c) LWdown and LWP
#
#               --> N.B. do the timcorr in cdo first, and then plot the time series in python

# Calculate differences between AWS/model for:
#       a) meteorological variables
#       b) fluxes
#       c) melt




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
    print('\nimporting data from %(config)s...' % locals())
    if vars == 'water paths':
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/proc_data/')
        print('\nice water path')
        try:
            IWP = iris.load_cube(OFCAP_IWP.nc, iris.AttributeConstraint(STASH='m01s02i392') & iris.Constraint(grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9, forecast_period=lambda cell: cell >= 12.5))# stash code s02i392
        except iris.exceptions.ConstraintMismatchError:
            print('\n IWP not in this file')
        print('\nliquid water path')
        try:
            LWP = iris.load_cube(pb, iris.AttributeConstraint(STASH='m01s02i391') & iris.Constraint(grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9, forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n LWP not in this file')
        for j in [LWP, IWP,]:
            j.convert_units('g m-2')
        mean_IWP = IWP.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # take mean of lazy data so cube data not loaded into memory
        mean_LWP = LWP.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        AWS14_mean_IWP = IWP[:, :,165:167, 98:100].collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        AWS14_mean_LWP = LWP[:, :,165:167, 98:100].collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        AWS15_mean_IWP = IWP[:, :,127:129, 81:83].collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        AWS15_mean_LWP = LWP[:, :,127:129, 81:83].collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        config_dict = {'AWS14_mean_IWP': AWS14_mean_IWP,'AWS15_mean_IWP': AWS15_mean_IWP, 'AWS14_mean_LWP': AWS14_mean_LWP,
                       'AWS15_mean_LWP': AWS15_mean_LWP, 'mean_IWP': mean_IWP, 'mean_LWP': mean_LWP}
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
        mean_QCF = ice_mass_frac.collapsed('altitude', iris.analysis.MEAN) #np.mean(ice_mass_frac.data, axis=(0, 1, 3, 4))
        mean_QCL = liq_mass_frac.collapsed('altitude', iris.analysis.MEAN) #np.mean(liq_mass_frac.data, axis=(0, 1, 3, 4))  # 0,2,3
        AWS14_mean_QCF = ice_mass_frac[:, :, :40, 165:167, 98:100].collapsed('altitude', iris.analysis.MEAN)
        AWS14_mean_QCL = liq_mass_frac[:, :, :40, 165:167, 98:100].collapsed('altitude', iris.analysis.MEAN)
        AWS15_mean_QCF = ice_mass_frac[:, :, :40, 127:129, 81:83].collapsed('altitude', iris.analysis.MEAN)
        AWS15_mean_QCL = liq_mass_frac[:, :, :40, 127:129, 81:83].collapsed('altitude', iris.analysis.MEAN)
        altitude = ice_mass_frac.coord('level_height').points / 1000
        config_dict = {'altitude': altitude,'mean_QCF': mean_QCF,'mean_QCL': mean_QCL,
                       'AWS14_mean_QCF': AWS14_mean_QCF, 'AWS14_mean_QCL': AWS14_mean_QCL,
                       'AWS15_mean_QCF': AWS15_mean_QCF, 'AWS15_mean_QCL': AWS15_mean_QCL}
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
        mean_IWP = IWP.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # take mean of lazy data so cube data not loaded into memory
        mean_LWP = LWP.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        AWS14_mean_IWP = IWP[:, :,165:167, 98:100].collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        AWS14_mean_LWP = LWP[:, :,165:167, 98:100].collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        AWS15_mean_IWP = IWP[:, :,127:129, 81:83].collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        AWS15_mean_LWP = LWP[:, :,127:129, 81:83].collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
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
        mean_QCF = ice_mass_frac.collapsed('altitude', iris.analysis.MEAN) #np.mean(ice_mass_frac.data, axis=(0, 1, 3, 4))
        mean_QCL = liq_mass_frac.collapsed('altitude', iris.analysis.MEAN) #np.mean(liq_mass_frac.data, axis=(0, 1, 3, 4))  # 0,2,3
        AWS14_mean_QCF = ice_mass_frac[:, :, :40, 165:167, 98:100].collapsed('altitude', iris.analysis.MEAN)
        AWS14_mean_QCL = liq_mass_frac[:, :, :40, 165:167, 98:100].collapsed('altitude', iris.analysis.MEAN)
        AWS15_mean_QCF = ice_mass_frac[:, :, :40, 127:129, 81:83].collapsed('altitude', iris.analysis.MEAN)
        AWS15_mean_QCL = liq_mass_frac[:, :, :40, 127:129, 81:83].collapsed('altitude', iris.analysis.MEAN)
        altitude = ice_mass_frac.coord('level_height').points / 1000
        config_dict = {'altitude': altitude,'mean_QCF': mean_QCF,'mean_QCL': mean_QCL,
                       'AWS14_mean_QCF': AWS14_mean_QCF, 'AWS14_mean_QCL': AWS14_mean_QCL,
                       'AWS15_mean_QCF': AWS15_mean_QCF, 'AWS15_mean_QCL': AWS15_mean_QCL,
                       'AWS14_mean_IWP': AWS14_mean_IWP,'AWS15_mean_IWP': AWS15_mean_IWP,
                       'AWS15_mean_LWP': AWS15_mean_LWP, 'mean_IWP': mean_IWP, 'mean_LWP': mean_LWP}
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
    return config_dict, constr_lsm, constr_orog

def load_SEB(config, vars):
    ''' Import surface energy balance quantities at AWS 14 from an OFCAP model run.

    Inputs:
        - config = model configuration used
        - vars = string describing which variables you want to output. Should be either 'downwelling' or 'SEB'.

    Author: Ella Gilbert, 2018.

    '''
    pa = []
    pf = []
    print('\nimporting data from %(config)s...' % locals())
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test/'):
        if fnmatch.fnmatch(file,  '*%(config)s_pf*' % locals()):
            pf.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/')
    for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/'):
        if fnmatch.fnmatch(file,  '*%(config)s_pa*' % locals()):
            pa.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test/')
    if vars == 'downwelling':
        print('\n Downwelling longwave')
        try:
            LW_down = iris.load_cube(pf, iris.Constraint(name='surface_downwelling_longwave_flux',
                                                         grid_longitude = lambda cell: 178.5 < cell < 180.6,
                                                         grid_latitude = lambda cell: -2.5 < cell < 0.9,
                                                         forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n Downwelling LW not in this file')
        print('\nDownwelling shortwave')
        try:
            SW_down = iris.load_cube(pf, iris.Constraint(name='surface_downwelling_shortwave_flux_in_air',
                                                         grid_longitude=lambda cell: 178.5 < cell < 180.6,
                                                         grid_latitude=lambda cell: -2.5 < cell < 0.9,
                                                         forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n Downwelling SW not in this file')
        var_dict = {'LW_down': LW_down, 'SW_down': SW_down}
    elif vars == 'SEB':
        print('\n Downwelling longwave')
        try:
            LW_down = iris.load_cube(pf, iris.Constraint(name='surface_downwelling_longwave_flux', grid_longitude=180,
                                                         grid_latitude=0,
                                                         forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n Downwelling LW not in this file')
        print('\n Net longwave')
        try:
            LW_net = iris.load_cube(pf, iris.Constraint(name='surface_net_downward_longwave_flux', grid_longitude=180,
                                                         grid_latitude=0,
                                                         forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n Net LW not in this file')
        print('\n Net shortwave')
        try:
            SW_net = iris.load_cube(pf, iris.Constraint(name='surface_net_downward_shortwave_flux', grid_longitude=180,
                                                         grid_latitude=0,
                                                         forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n Net SW not in this file')
        print('\nDownwelling shortwave')
        try:
            SW_down = iris.load_cube(pf, iris.Constraint(name='surface_downwelling_shortwave_flux_in_air',
                                                         grid_longitude=180,
                                                         grid_latitude=0, forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n Downwelling SW not in this file')
        print('\nUpwelling shortwave')
        try:
            SW_up = iris.load_cube(pf, iris.Constraint(name='upwelling_shortwave_flux_in_air',
                                                         grid_longitude=180,
                                                         grid_latitude=0,
                                                       model_level_number = 1, forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n Upwelling SW not in this file')
            SW_up = SW_net - SW_down
        print('\nUpwelling longwave')
        try:
            LW_up = iris.load_cube(pf, iris.Constraint(name='upwelling_longwave_flux_in_air',
                                                       grid_longitude=180,
                                                       grid_latitude=0,
                                                       model_level_number=1,
                                                       forecast_period=lambda cell: cell >= 12.5))
        except iris.exceptions.ConstraintMismatchError:
            print('\n Upwelling LW not in this file')
            LW_up = LW_net - LW_down
        print('\nSensible heat')
        try:
            SH = iris.load_cube(pf, iris.Constraint(name='surface_upward_sensible_heat_flux',
                                                       grid_longitude=180,
                                                       grid_latitude=0,
                                                       forecast_period=lambda cell: cell >= 12.5))
            SH = 0 - SH.data
        except iris.exceptions.ConstraintMismatchError:
            print('\n SH not in this file')
        print('\nLatent heat')
        try:
            LH = iris.load_cube(pf, iris.Constraint(name='surface_upward_latent_heat_flux',
                                                       grid_longitude=180,
                                                       grid_latitude=0,
                                                       forecast_period=lambda cell: cell >= 12.5))
            LH = 0 - LH.data
        except iris.exceptions.ConstraintMismatchError:
            print('\n LH not in this file')
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/')
        print('\nSurface temperature')
        try:
            Ts = iris.load_cube(pa, iris.Constraint(name='surface_temperature',
                                                    grid_longitude=180,
                                                    grid_latitude=0,
                                                    forecast_period=lambda cell: cell >= 12.5))
            # Ts.convert_units('celsius')
        except iris.exceptions.ConstraintMismatchError:
            print('\n Ts not in this file')
        var_dict = {'SW_up': SW_up, 'SW_down': SW_down, 'LH': LH, 'SH': SH, 'LW_up': LW_up, 'LW_down': LW_down,  'Ts': Ts}
    return var_dict



def load_met(config):
    ''' Import meteorological quantities at AWS 14 from an OFCAP model run.

    Inputs:
        - config = model configuration used

    Author: Ella Gilbert, 2018.

    '''
    start = time.time()
    pa = ['20110101T0000Z_Peninsula_1p5km_RA1M_mods_lg_t_pa000.pp','20110101T1200Z_Peninsula_1p5km_RA1M_mods_lg_t_pa000.pp' ]
    #print('\nimporting data from %(config)s...' % locals())
    #for file in os.listdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/'):
    #    if fnmatch.fnmatch(file, '*%(config)s*_pa*' % locals()):
    #        pa.append(file)
    os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/large 00Z files/')
    print('\nAir temperature')
    # Load only last 12 hours of forecast (i.e. t+12 to t+24, discarding preceding 12 hours as spin-up) for bottom 40 levels, and perform unit conversion from K to *C
    try:
        T_air = iris.load_cube(pa, iris.Constraint(name='air_temperature'))
                                               #grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9))
        T_air.convert_units('celsius')
    except iris.exceptions.ConstraintMismatchError:
        print('\n T_air not in this file')
    print('\nAir potential temperature')
    try:
        theta = iris.load_cube(pa, iris.Constraint(name='air_potential_temperature', model_level_number=lambda cell: cell <= 40))
                                               #grid_longitude=lambda cell: 178.5 < cell < 180.6,
                                               #grid_latitude=lambda cell: -2.5 < cell < 0.9))
        theta.convert_units('celsius')
    except:
        print('\n theta not in this file')
    print('\nSurface temperature')
    try:
        Ts = iris.load_cube(pa, iris.Constraint(name='surface_temperature'))
                                            #grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9))
        Ts.convert_units('celsius')
    except:
        print('\n Ts not in this file')
    print('\nSpecific humidity')
    try:
        q = iris.load_cube(pa, iris.Constraint(name='specific_humidity', model_level_number=lambda cell: cell <= 40))
                                           #grid_longitude = lambda cell: 178.5 < cell < 180.6, grid_latitude = lambda cell: -2.5 < cell < 0.9,))
        q.convert_units('g kg-1') # Convert to g kg-1
    except:
        print('\n q not in this file')
    print('\nMean sea level pressure')
    try:
        MSLP = iris.load_cube(pa, iris.Constraint(name = 'air_pressure_at_sea_level')  & iris.Constraint(forecast_period=lambda cell: cell >= 12.5))
        MSLP.convert_units('hPa')
    except:
        print('\n MSLP not in this file')
    print('\nZonal component of wind')
    #try:
    #    u = iris.load_cube(pa, iris.Constraint(name = 'x_wind', forecast_period=lambda cell: cell >= 12.5))
    #print('\nMeridional component of wind')
    #v = iris.load(pa, iris.Constraint(name = 'y_wind', forecast_period=lambda cell: cell >= 12.5))
    print('\nLSM')
    lsm = iris.load_cube(pa, 'land_binary_mask')
    lsm = lsm[0,:,:]
    print('\nOrography')
    orog = iris.load_cube(pa, 'surface_altitude')
    orog = orog[0,:,:]
    for i in [theta, T_air,  q]: # 4-D variables     u, v,
        real_lon, real_lat = rotate_data(i, 3, 4)
    for j in [Ts, MSLP]:  # 3-D variables
        real_lon, real_lat = rotate_data(j, 2, 3)  # time vars don't load in properly = forecast time + real time
    for k in [lsm, orog]: # 2-D variables
        real_lon, real_lat = rotate_data(k, 0, 1)
    # Convert times to useful ones
    print('\nConverting times...')
    for i in [theta, T_air, Ts, q, MSLP]:# , u, v,
        i.coord('time').convert_units('hours since 2011-01-01 00:00')
    # Create spatial means for maps
    print('\nCalculating means...')
    mean_MSLP = np.mean(MSLP.data, axis = (0,1))
    mean_Ts = np.mean(Ts.data, axis = (0,1))
    # Sort out time series loading
    def construct_srs(cube):
        i = range(len(cube.coord('forecast_reference_time').points))
        k = cube.data
        series = []
        for j in i:
            a = k[:, j]
            a = np.array(a)
            series = np.append(series, a)
        return series
    # Produce time series
    print('\nCreating time series...')
    AWS14lon, AWS14lat = find_gridbox(-67.01, -61.03, real_lat, real_lon)
    AWS15lon, AWS15lat = find_gridbox(-67.34, -62.09, real_lat, real_lon)
    AWS14_Ts = Ts[:,:,AWS14lat,AWS14lon]
    AWS14_Ts_srs = construct_srs(AWS14_Ts)
    AWS14_Tair = T_air[:,:,0, AWS14lat,AWS14lon]
    AWS14_Tair_srs = construct_srs(AWS14_Tair)
    AWS15_Ts = Ts[:,:,AWS15lat,AWS15lon]
    AWS15_Ts_srs = construct_srs(AWS15_Ts)
    AWS15_Tair = T_air[:,:,0, AWS15lat,AWS15lon]
    AWS15_Tair_srs = construct_srs(AWS15_Tair)
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
    AWS14_mean_T = np.mean(T_air[:, :, :, 199:201, 199:201].data, axis=(0, 1, 3, 4))
    AWS14_mean_theta = np.mean(theta[:, :, :, 199:201, 199:201].data, axis=(0, 1, 3, 4))
    AWS14_mean_q= np.mean(q[:, :, :, 199:201, 199:201].data, axis=(0, 1, 3, 4))
    print('\nLast bit! Repeating for AWS 15...')
    AWS15_mean_T = np.mean(T_air[:, :, :, 161:163, 182:184].data, axis=(0, 1, 3, 4))
    AWS15_mean_theta = np.mean(theta[:, :, :, 161:163, 182:184].data, axis=(0, 1, 3, 4))
    AWS15_mean_q= np.mean(q[:, :, :, 161:163, 182:184].data, axis=(0, 1, 3, 4))
    altitude = T_air.coord('level_height').points[:40] / 1000
    var_dict = {'real_lon': real_lon, 'real_lat': real_lat,'lsm': lsm, 'orog': orog, 'altitude': altitude, 'box_T': box_T,
                'box_theta': box_q, 'AWS14_mean_T': AWS14_mean_T, 'AWS14_mean_theta': AWS14_mean_theta, 'AWS14_mean_q': AWS14_mean_q,
                'AWS15_mean_T': AWS15_mean_T, 'AWS15_mean_theta': AWS15_mean_theta, 'AWS15_mean_q': AWS15_mean_q,
                'AWS14_Ts_srs': AWS14_Ts_srs, 'AWS14_Tair_srs': AWS14_Tair_srs, 'AWS15_Ts_srs': AWS15_Ts_srs, 'AWS15_Tair_srs': AWS15_Tair_srs,
                'T_air': T_air, 'Ts':  Ts, 'q': q, 'theta': theta, 'MSLP': MSLP}#
    end = time.time()
    print
    '\nDone, in {:01d} secs'.format(int(end - start))
    return var_dict

#Jan_SEB = load_SEB(config = 'lg_t', vars = 'SEB')
#Jan_mp, constr_lsm, constr_orog = load_mp(config = 'lg_t', vars = 'water paths')
Jan_met = load_met('lg_t')

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

AWS15_Jan = load_AWS('AWS15', period = 'OFCAP')
AWS14_SEB_Jan  = load_AWS('AWS14_SEB', period = 'OFCAP')


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

## Hacky fix to deal with iris loading forecast period and forecast time separately

def construct_srs(var_name):
    i = np.arange(var_name.shape[1])
    k = var_name
    series = []
    for j in i:
        a = k[:, j]
        a = np.array(a)
        series = np.append(series, a)
    return series

os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test')

IWP14_srs = construct_srs(Jan_mp['AWS14_mean_IWP'])
LWP14_srs = construct_srs(Jan_mp['AWS14_mean_LWP'])
IWP15_srs = construct_srs(Jan_mp['AWS15_mean_IWP'])
LWP15_srs = construct_srs(Jan_mp['AWS15_mean_LWP'])
box_IWP_srs = construct_srs(Jan_mp['mean_IWP'])
box_LWP_srs = construct_srs(Jan_mp['mean_LWP'])
#AWS14_SW_srs = construct_srs(np.mean(Jan_SEB['SW_down'][:,:,165:167, 98:100].data, axis = (2,3)))
#AWS14_LW_srs = construct_srs(np.mean(Jan_SEB['LW_down'][:,:,165:167, 98:100].data, axis = (2,3)))
#AWS15_SW_srs = construct_srs(np.mean(Jan_SEB['SW_down'][:,:,127:129, 81:83].data, axis = (2,3)))
#AWS15_LW_srs = construct_srs(np.mean(Jan_SEB['LW_down'][:,:,127:129, 81:83].data, axis = (2,3)))
#box_LW_srs = construct_srs(np.mean(Jan_SEB['LW_down'].data, axis = (2,3)))
#box_SW_srs = construct_srs(np.mean(Jan_SEB['SW_down'].data, axis = (2,3)))
Jan_SEB['SW_down'].coord('time').convert_units('seconds since 1970-01-01 00:00:00')
#Jan_mp['mean_LWP'].coord('time').convert_units('seconds since 1970-01-01 00:00:00')
Time_srs = construct_srs(np.swapaxes(Jan_SEB['SW_down'].coord('time').points,0,1))
Time_srs = matplotlib.dates.num2date(matplotlib.dates.epoch2num(Time_srs))

AWS14_SEB_Jan[AWS14_SEB_Jan['LWin'] < -200] = np.nan
AWS15_Jan[AWS15_Jan['Lin'] < -200] = np.nan

#AWS15_dif_SW = AWS15_SW_srs - AWS15_Jan['Sin'][12:]
#AWS15_dif_LW = AWS15_LW_srs - AWS15_Jan['Lin'][12:]
#AWS14_dif_SW = AWS14_SW_srs - AWS14_SEB_Jan['SWin_corr'][24::2]
#AWS14_dif_LW = AWS14_LW_srs - AWS14_SEB_Jan['LWin'][24::2]

# Create time series of SEB parameters
SW_down_srs = construct_srs(Jan_SEB['SW_down'].data)
SW_up_srs = construct_srs(Jan_SEB['SW_up'].data)
LW_down_srs = construct_srs(Jan_SEB['LW_down'].data)
LW_up_srs = construct_srs(Jan_SEB['LW_up'].data)
SH_srs = construct_srs(Jan_SEB['SH'])
LH_srs = construct_srs(Jan_SEB['LH'])
Ts_srs = construct_srs(Jan_SEB['Ts'].data)
E_srs = (SW_down_srs - SW_up_srs) + (LW_down_srs - LW_up_srs) + LH_srs[:732] + SH_srs[:732]
melt_srs = np.ma.masked_where(Ts_srs < -0.025, E_srs)
melt_srs[Ts_srs < -0.025 & E_srs > 0] = 0
print('melt mean = ' + np.mean(melt_srs))

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

#mod_profile()


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
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test/')
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
    #fig.text(0.5, 0.04, 'Time (hours)', fontsize=24, fontweight = 'bold', ha = 'center', va = 'center', color = 'dimgrey')
    fig.text(0.03, 0.8, 'IWP \n(g m$^{-2}$)', fontsize=30, ha= 'center', va='center', rotation = 0, color = 'dimgrey')
    fig.text(0.03, 0.4, 'LWP \n(g m$^{-2}$)', fontsize=30, ha='center', va='center', color = 'dimgrey', rotation=0)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vn11_water_path_time_srs_OFCAP.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vn11_water_path_time_srs_OFCAP.eps')
    plt.show()

#IWP_time_srs()

def rad_time_srs():
    model_runs = [Jan_SEB]
    fig, ax = plt.subplots(2,1, sharex = True, figsize = (30,14))
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=3, color='dimgrey')
        axs.spines['right'].set_visible(False)
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    plot = 0
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    for run in model_runs:
        os.chdir('/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/test/')
        print('\nPLOTTING DIS BIATCH...')
        ax[plot].spines['right'].set_visible(False)
        ax[plot].plot(Time_srs, AWS14_SW_srs, label = 'AWS14 SW$_{\downarrow}$: modelled', linewidth = 2,  linestyle = '--', color = 'darkred')
        ax[plot].plot(Time_srs,AWS15_SW_srs, label='AWS15 SW$_{\downarrow}$', linewidth=2, linestyle = '--', color='darkblue')
        #ax[plot].plot(Time_srs,box_SW_srs, label='Cloud box SW$_{\downarrow}$', linewidth=2, linestyle='--', color='k')
        ax2 = ax[plot].twiny()
        ax2.set_xlim(1.5,max(AWS14_SEB_Jan['Time']))
        ax2.plot(AWS14_SEB_Jan['Time'], AWS14_SEB_Jan['SWin_corr'], label = 'AWS14 SW$_{\downarrow}$: observed', linewidth = 2,  color = 'darkred')
        ax2.plot(AWS15_Jan['Jday'], AWS15_Jan['Sin'], label = 'AWS15 SW$_{\downarrow}$: observed', linewidth = 2,  color = 'darkblue')
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
        ax[plot+1].plot(Time_srs,AWS15_LW_srs, label = 'AWS15_LW$_{\downarrow}$', linewidth=2, linestyle = '--', color='darkblue')
        #ax[plot+1].plot(Time_srs,box_LW_srs, label = 'Cloud box LW$_{\downarrow}$', linewidth=2, linestyle='--', color='k')
        ax[plot+1].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
        ax2 = ax[plot+1].twiny()
        ax2.plot(AWS14_SEB_Jan['Time'],AWS14_SEB_Jan['LWin'], label = 'AWS14 LW$_{\downarrow}$: observed', linewidth = 2,  color = 'darkred')
        ax2.set_xlim(1.5,max(AWS14_SEB_Jan['Time']))
        ax2.plot(AWS15_Jan['Jday'], AWS15_Jan['Lin'], label='AWS15 LW$_{\downarrow}$: observed', linewidth=2, color='darkblue')
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
           Line2D([0], [0], color='darkred', linestyle = '--', linewidth=3),
           Line2D([0], [0], color='darkblue', linewidth=3),
           Line2D([0], [0], color='darkblue', linestyle = '--', linewidth=3)]
    labs = ['AWS 14, observed', 'AWS 14, modelled','AWS 15, observed', 'AWS 15, modelled']
    lgd = plt.legend(lns, labs, ncol=2, bbox_to_anchor=(1., 2.12), borderaxespad=0., loc='best', prop={'size': 24})
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.97, wspace = 0.05, hspace = 0.1)
    ax[1].tick_params(axis='both', labelsize=28, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    fig.text(0.5, 0.04, 'Time (hours)', fontsize=24, fontweight = 'bold', ha = 'center', va = 'center', color = 'dimgrey')
    fig.text(0.03, 0.8, 'SW$_{\downarrow}$\n(W m$^{-2}$)', fontsize=30, ha= 'center', va='center', rotation = 0, color = 'dimgrey')
    fig.text(0.03, 0.4, 'LW$_{\downarrow}$\n(W m$^{-2}$)', fontsize=30, ha='center', va='center', color = 'dimgrey', rotation=0)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vn11_rad_time_srs_AWS14_Jan_2011.png')
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/vn11_rad_time_srs_AWS14_Jan_2011.eps')
    plt.show()

#rad_time_srs()


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
        #ax.set_xlim(0,800)
        #ax.set_ylim(0,300)
        ax.scatter(box_SW_srs, box_LWP_srs*1000, color='#f68080',s=50)
#        ax.set_ylim(np.min(np.mean(runMP['LWP'][:,:, 133:207, 188:213].data, axis=0)),
#                          np.max(np.mean(runMP['LWP'][:,:, 133:207, 188:213].data, axis=(0))))
#        ax.set_xlim(np.min(np.mean(runSEB['LW_down'][:,:, 133:207, 188:213].data, axis=(0))),
#                          np.max(np.mean(runSEB['LW_down'][:,:, 133:207, 188:213].data, axis=(0))))
        slope, intercept, r2, p, sterr = scipy.stats.linregress(box_SW_srs, box_LWP_srs)
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
        slope, intercept, r2, p, sterr = scipy.stats.linregress(box_SW_srs, box_IWP_srs)
        if p <= 0.01:
            ax.text(0.75, 0.9, horizontalalignment='right', verticalalignment='top',
                              s='r$^{2}$ = %s' % np.round(r2, decimals=2), fontweight='bold',
                              transform=ax.transAxes, size=24, color='dimgrey')
        else:
            ax.text(0.75, 0.9, horizontalalignment='right', verticalalignment='top', s='r$^{2}$ = %s' % np.round(r2, decimals=2),
                              transform=ax.transAxes, size=24,color='dimgrey')
        ax.scatter(box_SW_srs, box_IWP_srs*1000, color='#f68080', s=50)
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
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_SEB_v_'+phase+'.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_SEB_v_'+phase+'.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_SEB_v_'+phase+'.pdf', transparent=True)
    plt.show()

#correl_SEB_sgl(Jan_SEB, Jan_mp, phase = 'liquid')

## Caption: Box plots showing the modelled variation in ice and liquid water paths over the OFCAP period across the entire
## Larsen ice shelf, and at AWSs 14 and 15. Median values are indicated by the pink line in the centre of each box, while
## the green diamonds show the model mean. The whiskers extend to the 5th and 95th percentiles of the data, and outlying
## points are shown with grey crosses.

def boxplot(data):
    fig, ax = plt.subplots(1,1, figsize = (18,8))
    #ax.set_ylim(-10,1000)
    import matplotlib.cbook as cbook
    if data == 'SEB':
        model_data = [SW_down_srs, SW_up_srs, LW_down_srs, LW_up_srs, SH_srs, LH_srs, E_srs, melt_srs]
        obs_data = [AWS14_SEB_Jan['SWin_corr'], (0 - AWS14_SEB_Jan['SWout']), AWS14_SEB_Jan['LWin'],
                    (0 - AWS14_SEB_Jan['LWout_corr']), AWS14_SEB_Jan['Hsen'], AWS14_SEB_Jan['Hlat'], E_tot,
                    AWS14_SEB_Jan['melt_energy']]
        labels = ['SW$_{down}$', 'SW$_{up}$', 'LW$_{down}$', 'LW$_{up}$', 'H$_S$', 'H$_L$', 'E$_{tot}$', 'E$_{melt}$']
        stats = cbook.boxplot_stats(model_data, labels=labels)
        # change means in model_data table to be means of observations
        for n in range(len(stats)):
            stats[n]['mean'] = np.mean(obs_data[n])
        text_str = 'SEB'
        #ax.set_xticklabels(['SW$_{down}$', 'SW$_{up}$','LW$_{down}$','LW$_{up}$','H$_S$', 'H$_L$', 'E$_{tot}$', 'E$_{melt}$'])
    elif data == 'mp':
        model_data = [box_IWP_srs*1000, box_LWP_srs*1000, IWP14_srs*1000, LWP14_srs*1000, IWP15_srs*1000, LWP15_srs*1000]
        labels = ['Ice shelf\n mean IWP', 'Ice shelf \nmean LWP', 'AWS14\n IWP', 'AWS14\n LWP', 'AWS15\n IWP', 'AWS14\n LWP']
        stats = cbook.boxplot_stats(model_data, labels=labels)
        text_str = 'mp'
        #['ice shelf\n mean IWP', 'ice shelf \nmean LWP', 'AWS14\n IWP', 'AWS14\n LWP', 'AWS15\n IWP', 'AWS14\n LWP'])
    ax.boxplot(model_data,
               whis = [5,95], showmeans = True,
               whiskerprops= dict(linestyle='--', color = '#222222', linewidth = 1.5),
               capprops = dict(color = '#222222', linewidth = 1.5, zorder = 11),
               medianprops = dict(color = '#f68080', linewidth = 2.5, zorder = 6),
               meanprops = dict(marker = 'D', markeredgecolor = '#222222', markerfacecolor = '#33a02c', markersize = 10, zorder = 10),
               flierprops = dict(marker = 'x', markeredgecolor = 'dimgrey', zorder = 2, markersize = 10),
               boxprops = dict(linewidth = 1.5, color = '#222222', zorder = 8))# insert LWP at AWS 14 once I have got it!
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    #ax.set_yticks([ 0,250,500,750,1000])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Water path \n(g m$^{-2}$)', color = 'dimgrey', fontsize = 24, rotation = 0, labelpad =  100)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
    plt.subplots_adjust(bottom = 0.2, top = 0.95, right = 0.99, left = 0.2)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_'+text_str+'_boxplot.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_'+text_str+'_boxplot.pdf', transparent=True)
    plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_'+text_str+'_boxplot.eps', transparent=True)
    plt.show()

boxplot(data = 'SEB')



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




'''
E_tot = (AWS14_SEB_Jan['SWin_corr'] - AWS14_SEB_Jan['SWout']) + (AWS14_SEB_Jan['LWin'] - AWS14_SEB_Jan['LWout_corr']) + AWS14_SEB_Jan['Hlat']+ AWS14_SEB_Jan['Hsen'] - AWS14_SEB_Jan['Gs']


# https://matplotlib.org/gallery/statistics/bxp.html

fig, ax = plt.subplots(1, 1, figsize=(18, 8))
ax.set_ylim(-800,950)
ax.boxplot([AWS14_SEB_Jan['SWin_corr'], (0-AWS14_SEB_Jan['SWout']),AWS14_SEB_Jan['LWin'],  (0-AWS14_SEB_Jan['LWout_corr']),  AWS14_SEB_Jan['Hsen'],  AWS14_SEB_Jan['Hlat'], E_tot,  AWS14_SEB_Jan['melt_energy']],
           whis=[5, 95], showmeans=True,
           whiskerprops=dict(linestyle='--', color='#222222', linewidth=1.5),
           capprops=dict(color='#222222', linewidth=1.5, zorder=10),
           medianprops=dict(color='#f68080', linewidth=2.5, zorder=6),
           meanprops=dict(marker='D', markeredgecolor='#222222', markerfacecolor='#33a02c', markersize=10, zorder=11),
           flierprops=dict(marker='x', markeredgecolor='dimgrey', zorder=2, markersize=10),
           boxprops=dict(linewidth=1.5, color='#222222', zorder=8))  # insert LWP at AWS 14 once I have got it!
ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
ax.set_yticks([ -800, -400, 0, 400, 800])
ax.axhline(y=0, linewidth = 1.5, linestyle = '--', color = 'dimgrey')
ax.set_xticklabels(
    ['SW$_{down}$', 'SW$_{up}$','LW$_{down}$', 'LW$_{up}$', 'H$_S$', 'H$_L$', 'E$_{tot}$', 'E$_{melt}$']) #
ax.set_ylabel('Energy flux \n(W m$^{-2}$)', color='dimgrey', fontsize=24, rotation=0, labelpad=60)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.setp(ax.spines.values(), linewidth=2, color='dimgrey', )
plt.subplots_adjust(bottom=0.2, top=0.95, right=0.99, left=0.2)
plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_SEB_boxplot_AWS.png', transparent=True)
plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_SEB_boxplot_AWS.pdf', transparent=True)
plt.savefig('/users/ellgil82/figures/Cloud data/OFCAP_period/OFCAP_SEB_boxplot_AWS.eps', transparent=True)
plt.show()

#IWP_time_srs(),
#QCF_plot(), QCL_plot()

#T_plot()

'''

