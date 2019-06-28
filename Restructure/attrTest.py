import os
import lasmaster as lm
os.chdir("testingLAS")
for file_name in os.listdir():	
	if file_name[:4]=="T200":
		lm.wpd_attr(file_name)
