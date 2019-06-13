import numpy as np
from lasmaster.infotheory import entropy

# F U N C T I O N S   OF   E I G E N V A L U E S
# the point of doing this is that anyone can now define their own attributes, so long as they are functions of eigenvalues
def std_fun_eig():
	output =	{
			"p0"	:	(lambda x, y, z: x/(x+y+z)),
			"p1"	:	(lambda x, y, z: y/(x+y+z)),
			"p2"	:	(lambda x, y, z: z/(x+y+z)),
			"eig0"	:	(lambda x, y, z: x),
			"eig1"	:	(lambda x, y, z: y),
			"eig2"	:	(lambda x, y, z: z),
			"iso"	:	(lambda x, y, z: (x+y+z)/np.sqrt(3*(x**2+y**2+z**2))),
			"eigent"		:	(lambda x, y, z: entropy(np.stack((x, y, z), axis = 1)/((x+y+z)[:,None]))),
			"scattering"	:	(lambda x, y, z: x/z),
			"linearity"		:	(lambda x, y, z: (z-y)/z),
			"planarity"		:	(lambda x, y, z: (y-x)/z),
			"entent"		:	(lambda x, y, z: entropy(np.stack((x/z, (y-x)/z, (z-y)/z), axis = 1))),
			}	
	return output

# F U N C T I O N S   O F   E I G E N V E C T O R S
# the point of doing this is that anyone can now define their own attributes, so long as they are functions of eigenvectors
def std_fun_vec():
	normalised = (lambda v0, v1, v2: np.sqrt(v0[:,0]**2+v0[:,1]**2+v0[:,2]**2))
	output =	{					
			"ang0"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v0[:,2])/normalised(v0, v1, v2))/np.pi),0,1)),
			"ang1"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v1[:,2])/normalised(v0, v1, v2))/np.pi),0,1)),
			"ang2"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v2[:,2])/normalised(v0, v1, v2))/np.pi),0,1)),
			}
	return output

# F U N C T I O N S   O F   D I S T A N C E S
# the point of doing this is that anyone can now define their own attributes, so long as they are functions of k-distances
# when we decimate, these are functions of point counts and decimation parameter u
def std_fun_kdist():
	sphere_constant = 4*np.pi/3
	output = 	{
			"ptdens":	(lambda k, d: k/(sphere_constant*(d**3))),
			}
	return output
	
