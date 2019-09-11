# SignalCalculus
Investigatory repo looking for generic interpretation mechanism across ad hoc and derived signals.

# Signals for Point Clouds
This contains English descriptions of the structure of RedHawk, and explanations and plots of signals. We can't store the pictures on github.

# Lasmaster
This name used to be given to a module, the code is restructured into a package. It contains several modules:
* **lpinteraction.py** which governs the way that files are read and written. 
* **infotheory.py** which defines entropy and Jensen-Shannon divergence of probability arrays.
* **geo.py** which takes coordinates and produces geometric attributes - eigenvalues, eigenvectors, and k-distance.
* **fun.py** which defines all of the functions which act on geometric attributes. 

## lasmaster.\_\_init\_\_
Imports modules. Initialises package. Contains examples of running attribute generation and HAG (height above ground).

### Geo
Pretty important module. This is contains the mathematical heart of neighbourhood definition and the step thereafter: Structure tensor (covariance matrices) and the eigeninformation.

## optimise_k
Selects a $k$ value, $k>3$

## Info theory module lasmaster.infotheory

### entropy
Computes entropy.
* **input**(..., N) probability vectors
* that is, we must have np.sum(distribution, axis = -1)==1 and distribution>=0
* **output** entropies of vectors

### jsd (potentially in scikit so may be able to discard -- need to check)
Computes Jensen-Shannon Divergence (no weights).
* **input** (..., M, N) matrices whose rows are probability vectors
* **output** J-S div. of the collection of vectors