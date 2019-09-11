import numpy as np
from lasmaster.infotheory import entropy

def std_fun_eig():
	output =	{
			"eig0"			:	(lambda x, y, z: x/(x+y+z)),
			"eig1"			:	(lambda x, y, z: y/(x+y+z)),
			"eig2"			:	(lambda x, y, z: z/(x+y+z)),
			"dim3"			:	(lambda x, y, z: 3*x/(x+y+z)),
			"dim2"			:	(lambda x, y, z: 2*(y-x)/(x+y+z)),
			"dim1"			:	(lambda x, y, z: (z-y)/(x+y+z)),
			"diment"		:   (lambda x, y, z: np.clip(entropy(np.stack((3*x/(x+y+z), 2*(y-x)/(x+y+z), (z-y)/(x+y+z)), axis =1)), 0, 1)),
			}
	return output

def std_fun_vec():
	normalised = (lambda v: np.sqrt(v[:,0]**2+v[:,1]**2+v[:,2]**2))
	output =	{					
			"ang0"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v0[:,2])/normalised(v0)))/np.pi,0,1)),
			"ang1"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v1[:,2])/normalised(v1)))/np.pi,0,1)),
			"ang2"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v2[:,2])/normalised(v2)))/np.pi,0,1)),
			"eig20"	:	(lambda v0, v1, v2: v2[:,0]),
			"eig21"	:	(lambda v0, v1, v2: v2[:,1]),
			"eig22"	:	(lambda v0, v1, v2: v2[:,2]),
			"eig10"	:	(lambda v0, v1, v2: v1[:,0]),
			"eig11"	:	(lambda v0, v1, v2: v1[:,1]),
			"eig12"	:	(lambda v0, v1, v2: v1[:,2]),
			"eig00"	:	(lambda v0, v1, v2: v0[:,0]),
			"eig01"	:	(lambda v0, v1, v2: v0[:,1]),
			"eig02"	:	(lambda v0, v1, v2: v0[:,2]),
				}
	return output

def std_fun_kdist():
	sphere_constant = 4*np.pi/3
	output = 	{
			"ptdens":	(lambda k, d: k/(sphere_constant*(d**3))),
			"dist"	:	(lambda k, d: d),
			}
	return output
	
