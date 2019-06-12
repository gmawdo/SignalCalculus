# DO NOT RE-ORDER THESE IMPORTS

from lasmaster import geo
from lasmaster import infotheory
from lasmaster import fun
from lasmaster import lpinteraction
import time

wpd_config = 	{
			"timeIntervals"	:	6,
			"k"		:	50,
			"radius"	:	0.5,
			"virtualSpeed"	:	2,
			"decimation"	:	0.1,
		}

histo_config =	{}


def wpd_attr(file_name):
	start = time.time()
	lpinteraction.attr(file_name, wpd_config, fun.std_fun_eig(), fun.std_fun_vec(), fun.std_fun_kdist()) 
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def wpd_histo(file_name, attr = ["ent", "iso"]):
	in_file = lpinteraction.File(file_name, mode = "r")
	header = in_file.header
	
	coord_dictionary = {"x": in_file.x, "y": in_file.y, "z": in_file.z, "gps_time": in_file.gps_time}
	attr_dictionary = {name: in_file.reader.get_dimension(name) for name in attr}
	
	return geo.histogram(coord_dictionary, attr_dictionary, wpd_config, num_bins = 50)