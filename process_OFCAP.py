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

# Define which HPC the code will run on, and set up file path accordingly
host = 'BAS'

if host == 'BAS' or host == 'bslcenh':
	filepath = '/data/mac/ellgil82/cloud_data/um/vn11_test_runs/Jan_2011/'
elif host == 'JASMIN' or host == 'jasmin':
	filepath = '/group_workspaces/jasmin4/bas_climate/users/ellgil82/OFCAP/'
elif host == 'Monsoon' or host == 'monsoon':
	filepath = '/projects/polar/elgil/OFCAP/'

# Define function to load individual variables, amend the time dimension, and return a single cube.
def load_var(var):
	pa = []
	pb = []
	pf = []
	for file in os.listdir(filepath):
	        if fnmatch.fnmatch(file, '*pa*.pp'):
	            pa.append(file)
	        elif fnmatch.fnmatch(file, '*pb*.pp'):
	        	pb.append(file)
	        elif fnmatch.fnmatch(file, '*pf*.pp'):
	        	pf.append(file)
	os.chdir(filepath)
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
#sfc_P = load_var('sfc_P')
try:
	#print('Ts')
	#Ts = load_var('Ts')
	#iris.save((Ts), filepath+'/Ts.nc', netcdf_format='NETCDF3_CLASSIC')
#except (AttributeError, ValueError):
	T_air = load_var('T_air')
	iris.save((T_air), filepath+'T_air.nc', netcdf_format='NETCDF3_CLASSIC')
except (AttributeError, ValueError):
	theta = load_var('theta')
	iris.save((theta), filepath+'theta.nc', netcdf_format='NETCDF3_CLASSIC')
except (AttributeError, ValueError):
	q = load_var('q')
	iris.save((q), filepath+'q.nc', netcdf_format='NETCDF3_CLASSIC')
except (AttributeError, ValueError):
	MSLP = load_var('MSLP')
	iris.save((MSLP), filepath+'MSLP.nc', netcdf_format='NETCDF3_CLASSIC')
except (AttributeError, ValueError):
	u_wind = load_var('u_wind')
	iris.save((u_wind), filepath+'u_wind.nc', netcdf_format = 'NETCDF3_CLASSIC')
except (AttributeError, ValueError):
	v_wind = load_var('v_wind')
	iris.save((v_wind), filepath+'v_wind.nc', netcdf_format='NETCDF3_CLASSIC')
except (AttributeError, ValueError):
	print('Sozzzzz, nah')


# Save as individual time series




