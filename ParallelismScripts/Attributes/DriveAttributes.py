import os
from multiprocessing import Process

def the_process(file_name):
		os.system("cp -r lasmaster "+file_name)
		os.system("cp AttributeApply.py "+file_name)
		os.chdir(file_name)
		os.system("python3 AttributeApply.py")
		os.chdir("..")

A0 = "TILES_2017_04_21-03_25_47_0"
A1 = "TILES_2017_04_21-03_25_47_1"
A2 = "TILES_2017_04_21-03_25_47_2"
A3 = "TILES_2017_04_21-03_25_47_3"

if __name__ == '__main__':
	p0 = Process(target=the_process, args=(A0,))
	p1 = Process(target=the_process, args=(A1,))
	p2 = Process(target=the_process, args=(A2,))
	p3 = Process(target=the_process, args=(A3,))
	p0.start()
	p1.start()
	p2.start()
	p3.start()
	p0.join()
	p1.join()
	p2.join()
	p3.join()
