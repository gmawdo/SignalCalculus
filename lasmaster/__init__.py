from lasmaster import geo
from lasmaster import infotheory
from lasmaster import fun
from lasmaster import lpinteraction
import time
import numpy as np

example_attr_config = 	{
			"timeIntervals"	:	10,
			"k"				:	range(4,50), # must be a generator
			"radius"		:	0.5,
			"virtualSpeed"	:	2.0,
			"decimate"		:	0.1,
						}

example_hag_config = 	{
			"vox"			:	2,
			"alpha"			:	0.01,
		}


def example_attr(file_name):
	print("Starting attributes: ", file_name)
	start = time.time()
	cf = example_attr_config
	lpinteraction.attr(file_name, cf, fun.std_fun_eig(), fun.std_fun_vec(), fun.std_fun_kdist()) 
	end = time.time()
	print("Finished attributes: ",file_name, ", time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")

def example_hag(file_name):
	start = time.time()
	cf = example_hag_config
	lpinteraction.add_hag(file_name, cf)
	end = time.time()
	print(file_name, "Time taken: "+str(int((end - start)/60))+" minutes and "+str(int(end-start-60*int((end - start)/60)))+" seconds")
