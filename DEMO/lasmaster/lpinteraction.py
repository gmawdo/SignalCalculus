import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from laspy.file import File
from lasmaster import geo
from lasmaster import fun
from laspy.header import VLR
from sklearn.neighbors import NearestNeighbors
import time

def name_modifier_attr(config):
	N = config["timeIntervals"]
	k = config["k"]
	radius = config["radius"]
	v_speed = config["virtualSpeed"]
	u = config["decimate"]
	spacetime = bool(v_speed)
	if np.isinf(radius):
		R = "Infty"
	else:
		R_int = int(radius)
		R_rat = int(100*(radius-int(radius)))
		R = "radius"+str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)
	u_int = int(u)
	u_rat = int(100*(u-int(u)))
	u = "dec"+str(u_int).zfill(2)+"_"+str(u_rat).zfill(2)
	C_int = int(v_speed)
	C_rat = int(100*(v_speed-int(v_speed)))
	num = "N"+str(N).zfill(3)
	K = "k"+str(min(k)).zfill(3)+"_"+str(max(k)).zfill(3)
	C = spacetime*("v_speed"+str(C_int).zfill(2)+"_"+str(C_rat).zfill(2))
	return "attr"+num+K+R+C+u

def attr(file_name, config, fun_val = fun.std_fun_val, fun_vec = fun.std_fun_vec, fun_kdist = fun.std_fun_kdist):
	print("attr. start", file_name)
	in_file = File(file_name, mode = "r")
	header=in_file.header
	x = in_file.x
	y = in_file.y
	z = in_file.z
	time = in_file.gps_time
	val, vec, k, kdist, inv = geo.attibutes_prelim(x,y,z, time, config)
	mod = name_modifier_attr(config)
	out_file = File(mod+file_name, mode = "w", header = header)
	
	# extract names of pre-existing attributes
	dimensions = [spec.name for spec in in_file.point_format]

	for dimension in fun_val(val):
		if not(dimension in dimensions):
			out_file.define_new_dimension(name = dimension, data_type = 9, description = dimension)

	for dimension in fun_vec(vec):
		if not(dimension in dimensions):
			out_file.define_new_dimension(name = dimension, data_type = 9, description = dimension)

	for modifier in ["max", "one", "opt"]:
		for dimension in fun_kdist():
			if not(modifier+dimension in dimensions):
				out_file.define_new_dimension(name = modifier+dimension, data_type = 9, description = modifier+dimension)

	if not("kopt" in dimensions):
		out_file.define_new_dimension(name = "kopt", data_type = 6, description = "koptimal")
	if not("vox" in dimensions):
		out_file.define_new_dimension(name = "vox", data_type = 6, description = "voxel_number")
	if not("dec" in dimensions):
		out_file.define_new_dimension(name = "dec", data_type = 9, description = "decimation_scale")

	# add pre-existing point records
	for dimension in dimensions:
		dat = in_file.reader.get_dimension(dimension)
		out_file.writer.set_dimension(dimension, dat)

	for dimension in fun_val(val):
		value = fun_val(val)[dimension]
		value[np.logical_or(np.isnan(value),np.isinf(value))]=0
		out_file.writer.set_dimension(dimension, value)

	for dimension in fun_vec(vec):
		value = fun_vec(vec)[dimension]
		value[np.logical_or(np.isnan(value),np.isinf(value))]=0
		out_file.writer.set_dimension(dimension, value)

	for modifier in ["max", "one", "opt"]: 
		for dimension in fun_kdist():
			value = fun_kdist()[dimension](k[modifier], kdist[modifier])
			value[np.logical_or(np.isnan(value),np.isinf(value))]=0
			out_file.writer.set_dimension(modifier+dimension, value)

	out_file.writer.set_dimension("kopt", k["opt"])
	if inv != ...:
		out_file.writer.set_dimension("vox", inv)
	u = config["decimate"]
	out_file.writer.set_dimension("dec", u*np.ones(len(in_file)))

	in_file.close()
	out_file.close()

	print("attr. end", file_name)

def name_modifier_hag(config):
	alpha = config["alpha"] # alpha is small, between 0 and 1
	vox = config["vox"]
	A = "0_"+(str(int(1000*(alpha-int(alpha))))).zfill(3)
	R_int = int(vox)
	R_rat = int(100*(vox-int(vox)))
	R = str(R_int).zfill(2)+"_"+str(R_rat).zfill(2)+"_"
	return "hag"+"alpha"+A+"vox"+R

def add_hag(file_name, config):
	start = time.time()
	in_file = File(file_name, mode = "r")
	x_array = in_file.x
	y_array = in_file.y
	z_array = in_file.z
	coord_dictionary = {
						"x": x_array,
						"y": y_array,
						"z": z_array,
						}
	hag = geo.hag(coord_dictionary, config)
	mod = name_modifier_hag(config)
	out_file = File(mod+file_name, mode = "w", header = in_file.header)
	# add pre-existing point records
	dimensions = [spec.name for spec in in_file.point_format]
	if not("hag" in dimensions):
		out_file.define_new_dimension(name = "hag", data_type = 9, description = "hag")
	for dimension in dimensions:
		dat = in_file.reader.get_dimension(dimension)
		out_file.writer.set_dimension(dimension, dat)
	out_file.writer.set_dimension("hag", hag)
	out_file.close()

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
