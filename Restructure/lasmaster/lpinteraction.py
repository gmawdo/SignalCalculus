import numpy as np
from laspy.file import File
from lasmaster import geo
from lasmaster import fun
from sklearn.neighbors import NearestNeighbors
import time

# Governs the interaction with laspy

# the following function simply generates a name for output file
def name_modifier(config):
	N = config["timeIntervals"]
	k = config["k"]
	radius = config["radius"]
	v_speed = config["virtualSpeed"]
	spacetime = bool(v_speed)
	if np.isinf(radius):
		R = "Infty"
	else:
		R_int = int(radius)
		R_rat = int(100*(radius-int(radius)))
		R = "radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)
	C_int = int(v_speed)
	C_rat = int(100*(v_speed-int(v_speed)))
	num = "N"+str(N).zfill(3)
	K = "k"+str(min(k)).zfill(3)+"_"+str(max(k)).zfill(3)
	C = spacetime*("v_speed"+str(C_int).zfill(2)+"_"+str(C_rat).zfill(2))
	return "attr"+num+K+R+C

# the attr function applies the attribute defintions to the output of geo.eig
def attr(file_name, config, fun_eig, fun_vec, fun_kdist):
	in_file = File(file_name, mode = "r")
	header=in_file.header
	
	coord_dictionary = {"x": in_file.x, "y": in_file.y, "z": in_file.z, "gps_time": in_file.gps_time}

	val1, val2, val3, vec1, vec2, vec3, k, kdist = geo.eig(coord_dictionary, config)
	
	mod = name_modifier(config)
	out_file = File(mod+file_name, mode = "w", header = header)
	
	# extract names of pre-existing attributes
	dimensions = [spec.name for spec in in_file.point_format]

	for fun_set in fun_eig, fun_vec, fun_kdist:
		for dimension in fun_set:
			if not(dimension in dimensions):
				out_file.define_new_dimension(name = dimension, data_type = 9, description = dimension)

	out_file.define_new_dimension(name = "kopt", data_type = 6, description = "koptimal")

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

	out_file.writer.set_dimension("kopt", k)

	out_file.close()

# C L I P   N E A R   F L I G H T   L I N E
def nfl(file_name, clip = 100, fl = 10, change_name = True):
	start = time.time()
	in_file = File(file_name, mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	class10 = in_file.classification==fl
	x_array = in_file.x
	y_array = in_file.y
	coords = np.vstack((x_array,y_array))
	x_flight = x_array[class10]
	y_flight = y_array[class10]
	coords_flight = np.vstack((x_flight,y_flight))
	nhbrs = NearestNeighbors(n_neighbors = 1, algorithm = "kd_tree").fit(np.transpose(coords_flight))
	distances, indices = nhbrs.kneighbors(np.transpose(coords))
	out_file = File(file_name[:-4]+change_name*("NFLClip"+str(int(clip)).zfill(3)+"_"+str(int(100*(clip-int(clip)))).zfill(2))+".las", mode = "w", header = in_file.header)
	out_file.points = in_file.points[np.logical_and(distances[:,0]<clip, np.logical_not(class10))]
	out_file.close()
	end = time.time()
	return file_name[:-4]+change_name*("NFLClip"+str(int(clip)).zfill(3)+"_"+str(int(100*(clip-int(clip)))).zfill(2))+".las"
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")