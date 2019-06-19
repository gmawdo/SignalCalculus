# DO NOT RE-ORDER THESE IMPORTS

from lasmaster import geo
from lasmaster import infotheory
from lasmaster import fun
from lasmaster import lpinteraction
import time
import numpy as np

wpd_config = 	{
			"timeIntervals"	:	6,
			"k"				:	10,
			"radius"		:	0.5,
			"virtualSpeed"	:	2,
			"decimation"	:	0.0,
			"k-optimise"	:	False,
		}

enel_config = 	{
			"timeIntervals"	:	1,
			"k"				:	10,
			"radius"		:	0.5,
			"virtualSpeed"	:	0,
			"decimation"	:	0,
			"k-optimise"	:	True,
		}

histo_config =	{}


def wpd_attr(file_name):
	start = time.time()
	lpinteraction.attr(file_name, wpd_config, fun.std_fun_eig(), fun.std_fun_vec(), fun.std_fun_kdist()) 
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def enel_attr(file_name):
	start = time.time()
	cf = enel_config
	lpinteraction.attr(file_name, cf, fun.std_fun_eig(), fun.std_fun_vec(), fun.std_fun_kdist())
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def optimal_radius(file_name):
	start = time.time()
	cf = enel_config
	cf["k"]=10
	lpinteraction.radius_intensity(file_name, cf)
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")
