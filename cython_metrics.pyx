import numpy as np

def euc_p2(double[:] x, double[:] y):
	cdef double res = 0
	for i in range(3,6):
		res += (x[i] - y[i])**2
	j in range(3):
		res += 5*abs(x[i] * y[i])
	return res