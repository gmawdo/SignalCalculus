import numpy as np
from lasmaster.infotheory import entropy
from lasmaster.infotheory import jsd

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
			"ent"	:	(lambda x, y, z: entropy(np.stack((x, y, z))/(x+y+z)).T),
			}
	return output

# F U N C T I O N S   O F   E I G E N V E C T O R S
# the point of doing this is that anyone can now define their own attributes, so long as they are functions of eigenvectors
def std_fun_vec():
	output =	{					
			"ang0"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v0[:,2])/(np.sqrt(v0[:,0]**2+v0[:,1]**2+v0[:,2]**2)))/np.pi),0,1)),
			"ang1"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v1[:,2])/(np.sqrt(v1[:,0]**2+v1[:,1]**2+v1[:,2]**2)))/np.pi),0,1)),
			"ang2"	:	(lambda v0, v1, v2: np.clip(2*(np.arccos(abs(v2[:,2])/(np.sqrt(v2[:,0]**2+v2[:,1]**2+v2[:,2]**2)))/np.pi),0,1)),
			}
	return output

# F U N C T I O N S   O F   D I S T A N C E S
# the point of doing this is that anyone can now define their own attributes, so long as they are functions of k-distances
# when we decimate, these are functions of point counts and decimation parameter u
def std_fun_kdist():
	output = 	{
			"ptdens":	lambda k, d: k/(sphere_constant*(d**3)),
			}
	return output
	
