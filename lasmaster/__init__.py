# DO NOT RE-ORDER THESE IMPORTS

from lasmaster import geo
from lasmaster import infotheory
from lasmaster import fun
from lasmaster import lpinteraction
import numpy as np

# Below is an example config file. 
# timeIntervals dictates how many subtiles we have - this helps keep down memory usage
# k should be a generator which tells us which values of k we will entertain. In the example we look at 4-49.
# radius tells us the maximum size of a neighbourhood, which could be np.inf
# virtualSpeed is a weighting for how much time between points should affect their closeness
# use virtualSpeed = 0.0 to eliminate spacetime usage
# k-optimise dictates which attribute should be minimised in neighbourhood selection
# if just one value of k is given, optimisation will not occur

example_config = 	{
			"timeIntervals"	:	10,
			"k"				:	range(4,50),
			"radius"		:	0.5,
			"virtualSpeed"	:	2,
			"k-optimise"	:	"diment",
		}


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