import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from lasmaster.infotheory import entropy

def std_fun_val(val):
	d = val.shape[-1]
	output = {}
	actions = np.zeros((d,d), dtype = float)
	for i in range(d):
		actions[i, d-i-1] = d-i
		actions[i, d-i-2] = i-d+1
	LPS = np.matmul(val, actions)
	for i in range(d):
		output["eig"+str(i)] = val[:, i]/np.sum(val, axis = - 1) # for purpose of robust classification, our eigenvalues should come out in descending order
		output["dim"+str(i+1)] = LPS[:, i]
	output["diment"] = entropy(LPS)
	return output

def std_fun_vec(vec):
	d = vec.shape[-1]
	output = {}
	for i in range(d):
		output["ang"+str(i)] = np.clip(2*np.arccos(vec[:,i,2]), 0 , 1)
		for j in range(d):
			output["eig"+str(j)+str(i)] = vec[:, i, j] # in eigenvalue-ascending order
	return output

def std_fun_kdist():
	sphere_constant = 4*np.pi/3
	output = 	{
			"ptdens":	(lambda k, d: k/(sphere_constant*(d**3))),
			"dist"	:	(lambda k, d: d),
			}
	return output
	
