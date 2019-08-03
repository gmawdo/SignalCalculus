import os
import lasmaster as lm
import time

timingstxt = open("ATTR_TIMINGS.txt", "w+")

def f(file_name):
	cf = 	{
			"timeIntervals"	:	10,
			"k"				:	range(4,50), # must be a generator
			"radius"		:	0.5,
			"virtualSpeed"	:	0,
			}
	try:
		lm.lpinteraction.attr(lm.lpinteraction.nfl(file_name), config = cf)
	except:
		pass

def condition(file_name):
	return (file_name[-4:]==".las") and not("attr" in file_name) and not ("NFL" in file_name)

for tile_name in os.listdir():
	if condition(tile_name):
		start = time.time()
		f(tile_name)
		end = time.time()
		timingstxt.write(tile_name[:-4]+" "+str((end-start)+"\n")

timingstxt.close()
