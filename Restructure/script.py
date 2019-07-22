import lasmaster as lm
import numpy as np
import os

config = {
"timeIntervals" : 10,
"k": range(4,50), # must be a generator
"radius" : 0.5,
"virtualSpeed" : 0,
}

for file_name in os.listdir():
	if file_name[-4:] == ".las":
		lm.lpinteraction.attr(file_name, config, lm.fun.std_fun_eig(), lm.fun.std_fun_vec(), lm.fun.std_fun_kdist())	
