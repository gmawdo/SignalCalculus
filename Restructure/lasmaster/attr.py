import numpy as np

# E N T R O P Y   O F   P R O B A B I L I T Y   V E C T O R S
# input: (...,N) probability vectors
# that is, we must have np.sum(distribution, axis = -1)=1
# output: entropies of vectors
def entropy(distribution):
	N = distribution.shape[-1]
	logs = np.log(distribution)
	logs[np.logical_or(np.isnan(logs),np.isinf(logs))]=0
	entropies = np.average(-logs, axis = -1, weights = distribution)/np.log(N)
	return entropies

# J E N S E N - S H A N N O N   D I V E R G E N C E
# input: (...,M,N) matrices whose rows are probability vectors
# output: J-S div. of the collection of vectors
def jsd(distribution):
	M = distribution.shape[-2]
	N = distribution.shape[-1]
	return (entropy(np.mean(distribution, axis = -2))-np.mean(entropy(distribution), axis = -1))*np.log(N)/np.log(M)

# F U N C T I O N S   OF   E I G E N V A L U E S
# the point of doing this is that anyone can now define their own attributes, so long as they are functions of eigenvalues
def std_fun_eig(thresh):
	output =	{
			"p0"	:	(lambda x, y, z: x/(x+y+z)),
			"p1"	:	(lambda x, y, z: y/(x+y+z)),
			"p2"	:	(lambda x, y, z: z/(x+y+z)),
			"eig0"	:	(lambda x, y, z: x),
			"eig1"	:	(lambda x, y, z: y),
			"eig2"	:	(lambda x, y, z: z),
			"iso"	:	(lambda x, y, z: (x+y+z)/np.sqrt(3*(x**2+y**2+z**2))),
			"ent"	:	(lambda x, y, z: entropy(np.stack((x, y, z))/(x+y+z))),
			"rank"	:	(lambda x, y, z: 1*(x>thresh)+1*(y>thresh)+1*(z>thresh)),
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
def std_fun_dist(decimate):
	sphere_constant = (4/3)*np.pi
	if decimate:
		output =	{
				"ptdens":	lambda k, d: k/(d**3)
				}
	else:
		output = 	{
				"ptdens":	lambda k, d: k/(sphere_constant*(d**3))
				}
	return output
	

