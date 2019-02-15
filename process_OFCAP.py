## N.B. for monsoon type module load scitools/experimental-current before starting python

''' Script to process month-long OFCAP run in to individual variable files, which are (hopefully) smaller and easier to read and load.

Author: Ella Gilbert, 2019. Adapted from code supplied by James Pope. '''

# Import packages
import iris
import os
import fnmatch

# Define dictionary of standard strings that iris will look for when loading variables
long_name_dict = {'sfc_P': 'surface_air_pressure',
'Ts': 'surface_temperature',
'MSLP': 'air_pressure_at_sea_level',
'u_wind': 'x_wind',
'v_wind': 'y_wind',
'T_air': 'air_temperature', 
'q': 'specific_humidity',
'theta': 'air_potential_temperature', 
'QCF': 'mass_fraction_of_cloud_ice_in_air',
'QCL': 'mass_fraction_of_cloud_liquid_water_in_air',
}

# Define function to load individual variables, amend the time dimension, and return a single cube.
def load_var(var):
	pa = []
	pb = []
	pf = []
	for file in os.listdir('/projects/polar/elgil/OFCAP/'):
	        if fnmatch.fnmatch(file, '*pa*.pp'):
	            pa.append(file)
	        elif fnmatch.fnmatch(file, '*pb*.pp'):
	        	pb.append(file)
	        elif fnmatch.fnmatch(file, '*pf*.pp'):
	        	pf.append(file)
	try:
		raw = iris.load_raw(pa, long_name_dict[var])
	except iris.exceptions.ConstraintMismatchError:
		raw = iris.load_raw(pb, long_name_dict[var])
	except iris.exceptions.ConstraintMismatchError:
		raw = iris.load_raw(pf, long_name_dict[var])
	# copy the cubes to a new cubelist, with each cube having a 1-element time dimension plus auxiliary coordinates for forecast_reference_time and forecast_period
	cl = iris.cube.CubeList()
	for cube in raw:
	    new_cube = iris.util.new_axis(cube, 'time')
	    for coord_name in ['forecast_period', 'forecast_reference_time']:
	        coord = new_cube.coord(coord_name)
	        new_cube.remove_coord(coord_name)
	        new_cube.add_aux_coord(coord, new_cube.coord_dims('time')[0])
	    if new_cube.coord('forecast_period').points[0] != 0:
	        cl.append(new_cube)
	combo_cube = cl.concatenate_cube()
	return combo_cube

# Load and save variables into files
sfc_P = load_var('sfc_P')
Ts = load_var('Ts')
T_air = load_var('T_air')
theta = load_var('theta')
u_wind = load_var('u_wind')
v_wind = load_var('v_wind')

# Save as individual time series
iris.save((sfc_P), '/projects/polar/elgil/OFCAP/sfc_P.nc', netcdf_format = 'NETCDF3_CLASSIC')
iris.save((Ts), '/projects/polar/elgil/OFCAP/Ts.nc', netcdf_format = 'NETCDF3_CLASSIC')
iris.save((T_air), '/projects/polar/elgil/OFCAP/T_air.nc', netcdf_format = 'NETCDF3_CLASSIC')
iris.save((theta), '/projects/polar/elgil/OFCAP/theta.nc', netcdf_format = 'NETCDF3_CLASSIC')
iris.save((u_wind), '/projects/polar/elgil/OFCAP/u_wind.nc', netcdf_format = 'NETCDF3_CLASSIC')
iris.save((v_wind), '/projects/polar/elgil/OFCAP/v_wind.nc', netcdf_format = 'NETCDF3_CLASSIC')
