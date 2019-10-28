import os
import lasmaster as lm
import time
from multiprocessing import Process

timingstxt = open("ATTR_TIMINGS.txt", "w+")

def f(file_name):
	start = time.time()
	cf = 	{
			"timeIntervals"		:	10,
			"k"			:	range(4,50), # must be a generator
			"radius"		:	0.5,
			"virtualSpeed"		:	0,
			"decimate"		:	0.05,
			}
	lm.lpinteraction.attr(file_name, config = cf)
	end = time.time()
	timingstxt.write(file_name[:-4]+" "+str(end-start)+"\n")
	print(file_name)

file_name_type = "notgrd_TILE{}_2017_04_20-05_03_22_2NFLClip100_00.las"

def the_process(L):
	for item in L:
		f(item)

L1 = [file_name_type.format(i) for i in range(0,12)]
L2 = [file_name_type.format(i) for i in range(12,24)]
L3 = [file_name_type.format(i) for i in range(24,36)]
L4 = [file_name_type.format(i) for i in range(36,50)]

if __name__ == '__main__':
	p0 = Process(target=the_process, args=(L1,))
	p1 = Process(target=the_process, args=(L2,))
	p2 = Process(target=the_process, args=(L3,))
	p3 = Process(target=the_process, args=(L4,))
	p0.start()
	p1.start()
	p2.start()
	p3.start()
	p0.join()
	p1.join()
	p2.join()
	p3.join()


os.system("python3 ApplyClassn.py")
timingstxt.close()
