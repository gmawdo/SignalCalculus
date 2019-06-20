import numpy as np
from sklearn.neighbors import NearestNeighbors
from laspy.file import File
import matplotlib.pyplot as plt

def entropy(distribution):
	N = distribution.shape[-1]
	logs = np.log(distribution)
	logs[np.logical_or(np.isnan(logs),np.isinf(logs))]=0
	entropies = np.average(-logs, axis = -1, weights = distribution)/np.log(N)
	return entropies

def ladder_tensor(m, n):
	I = np.empty((n,n,n))
	for i in range(n):
		I[i,:,:]=np.identity(n)
		I[i,(i+1):,:]=0
	return I[m-1:,:,:] # returns an (n-m+1) x n x n, rank 3 tensor

inFile = File("pointselection.las", mode = "rw")

x = inFile.x
y = inFile.y
z = inFile.z

x0 = 385774.910
y0 = 207381.690
z0 = 127.320

min_num = 2
N = 1
k = 10
d=3

coords = np.vstack((x, y, z))

aux = min_num

nhbrs = NearestNeighbors(n_neighbors = k, algorithm = "kd_tree").fit(np.transpose(coords))
distances, indices = nhbrs.kneighbors(np.transpose(coords)) # (num_pts,k)
J = ladder_tensor(aux, k) # we use this tensor to avoid a loop # (k-aux+1, k, k)
raw_deviations_prestack = (coords[:,indices] - coords[:,:,None]) # (d,num_pts,k)

# matrix multiplications
raw_deviations = np.matmul(raw_deviations_prestack[:,None,:,:], J) # (d,k-aux+1,num_pts,k)
cov_matrices = np.matmul(raw_deviations.transpose(1,2,0,3), raw_deviations.transpose(1,2,3,0)) #(k-aux+1,num_pts,d,d)
cov_matrices = np.maximum(cov_matrices, cov_matrices.transpose(0,1,3,2)) #(k-aux+1,num_pts,d,d)
evals, evects = np.linalg.eigh(cov_matrices) #(k-aux+1,num_pts, d), (k-aux+1,num_pts,d,d)
linearity = (evals[:,:,-1]-evals[:,:,-2])/evals[:,:,-1] #(k-aux+1,num_pts)
planarity = (evals[:,:,-2]-evals[:,:,-3])/evals[:,:,-1] #(k-aux+1,num_pts)
scattering = evals[:,:,-3]/evals[:,:,-1] #(k-aux+1,num_pts)
stack = np.stack((linearity, planarity, scattering), axis = -1) #(k-aux+1,num_pts)
dim_ent = entropy(stack) #(k-aux+1,num_pts)
k_opt = aux + np.argmin(dim_ent, axis = 0)

inFile.classification = np.arange(10)
inFile.close()

for i in range(aux,10):
	for j in range(10):
		fig = plt.figure()
		y_pos = range(3)
		plt.bar(y_pos, stack[i-aux, j, :], align='center', alpha=0.5)
		plt.xticks(y_pos, ["linearity","planarity","scaling"])
		plt.xlabel('States')
		plt.ylabel('Probability')
		plt.title('Pt '+str(j)+' k = '+str(i)+' entropy = '+str(dim_ent[i-aux,j]))
		fig.savefig('Pt'+str(j)+'k'+str(i))
		plt.close()

for j in range(10):
	fig = plt.figure()
	plt.bar(range(k-aux+1), dim_ent[:,j], align='center', alpha=0.5)
	plt.xticks(range(k-aux+2), range(aux,k+1))
	plt.xlabel('k')
	plt.ylabel('Dim. ent.')
	plt.title('Pt '+str(j))
	fig.savefig('EntropyPt'+str(j)+'.png')
	plt.close()