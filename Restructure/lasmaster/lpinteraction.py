from laspy.file import File
import numpy as np
from lasmaster import geo
from lasmaster import fun

# Governs the interaction with laspy

# the following function simply generates a name for output file
def name_modifier(config):
		
	N = config["timeIntervals"]
	k = config["k"]
	radius = config["radius"]
	v_speed = config["virtualSpeed"]
	u = config["decimation"]
	spacetime = bool(v_speed)
	decimate = bool(u)
	R_int = int(radius)
	R_rat = int(100*(radius-int(radius)))
	C_int = int(v_speed)
	C_rat = int(100*(v_speed-int(v_speed)))
	U_int = int(u)
	U_rat = int(100*(u-int(u)))
	num = "N"+str(N).zfill(3)
	K = "k"+str(k).zfill(3)
	R = "radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)
	C = spacetime*("v_speed"+str(C_int).zfill(2)+"_"+str(C_rat).zfill(2))
	U = decimate*("dec"+str(U_int).zfill(2)+"_"+str(U_rat).zfill(2))
	return "attr"+num+K+R+C+U

# the attr function applies the attribute defintions to the output of geo.geo
def attr(file_name, config, fun_eig, fun_vec, fun_kdist):
	in_file = File(file_name, mode = "r")
	header=in_file.header
	
	coord_dictionary = {"x": in_file.x, "y": in_file.y, "z": in_file.z, "gps_time": in_file.gps_time}

	val1, val2, val3, vec1, vec2, vec3, k, kdist = geo.geo(coord_dictionary, config)
	
	mod = name_modifier(config)
	out_file = File(mod+file_name, mode = "w", header = header)
	
	# extract names of pre-existing attributes
	dimensions = [spec.name for spec in in_file.point_format]

	for fun_set in fun_eig, fun_vec, fun_kdist:
		for dimension in fun_set:
			if not(dimension in dimensions):
				out_file.define_new_dimension(name = dimension, data_type = 9, description = dimension)

	# add pre-existing point records
	for dimension in dimensions:
		dat = in_file.reader.get_dimension(dimension)
		out_file.writer.set_dimension(dimension, dat)

	for dimension in fun_eig:
		value = fun_eig[dimension](val1, val2, val3)
		value[np.logical_or(np.isnan(value),np.isinf(value))]=0
		out_file.writer.set_dimension(dimension, value)

	for dimension in fun_vec:
		value = fun_vec[dimension](vec1, vec2, vec3)
		value[np.logical_or(np.isnan(value),np.isinf(value))]=0
		out_file.writer.set_dimension(dimension, value)

	for dimension in fun_kdist:
		value = fun_kdist[dimension](k, kdist)
		value[np.logical_or(np.isnan(value),np.isinf(value))]=0
		out_file.writer.set_dimension(dimension, value)
		






