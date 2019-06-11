# E N T R O P Y   O F   P R O B A B I L I T Y   V E C T O R S
# input: (..., N) probability vectors
# that is, we must have np.sum(distribution, axis = -1)=1
# output: entropies of vectors

import numpy as np

def entropy(distribution):
	N = distribution.shape[-1]
	logs = np.log(distribution)
	logs[np.logical_or(np.isnan(logs),np.isinf(logs))]=0
	entropies = np.average(-logs, axis = -1, weights = distribution)/np.log(N)
	return entropies

# J E N S E N - S H A N N O N   D I V E R G E N C E
# input: (..., M, N) matrices whose rows are probability vectors
# output: J-S div. of the collection of vectors
def jsd(distribution):
	M = distribution.shape[-2]
	N = distribution.shape[-1]
	return (entropy(np.mean(distribution, axis = -2))-np.mean(entropy(distribution), axis = -1))*np.log(N)/np.log(M)
