# DO NOT RE-ORDER THESE IMPORTS

from lasmaster import geo
from lasmaster import infotheory
from lasmaster import fun
from lasmaster import lpinteraction
import time
import numpy as np

wpd_config = 	{
			"timeIntervals"	:	6,
			"k"		:	50,
			"radius"	:	0.5,
			"virtualSpeed"	:	2,
			"decimation"	:	0.1,
		}

enel_config = 	{
			"timeIntervals"	:	1,
			"k"		:	50,
			"radius"	:	np.inf,
			"virtualSpeed"	:	2,
			"decimation"	:	0,
		}

histo_config =	{}


def wpd_attr(file_name):
	start = time.time()
	lpinteraction.attr(file_name, wpd_config, fun.std_fun_eig(), fun.std_fun_vec(), fun.std_fun_kdist()) 
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def enel_attr(file_name):
	start = time.time()
	lpinteraction.attr(file_name, enel_config, fun.std_fun_eig(), fun.std_fun_vec(), fun.std_fun_kdist()) 
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

