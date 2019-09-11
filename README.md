# SignalCalculus
Investigatory repo looking for generic interpretation mechanism across ad hoc and derived signals.


## Restructure
Contains a python package called **lasmaster** (this name used to be given to a module, but now we should restructure the code). It contains several modules:
* **lpinteraction.py** which governs the way that files are read and written. 
* **infotheory.py** which defines entropy and Jensen-Shannon divergence of probability arrays.
* **geo.py** which takes coordinates and produces geometric attributes - eigenvalues, eigenvectors, and k-distance.
* **fun.py** which defines all of the functions which act on geometric attributes. 

## Signals for Point Clouds
This contains English descriptions of the structure of Red Hawk, and explanations and plots of signals. We can't store the pictures on github.

# Info theory module lasmaster.infotheory

## Entropy
* **input**(..., N) probability vectors
* that is, we must have np.sum(distribution, axis = -1)=1
* **output** entropies of vectors

## Jensen-Shannon divergence (potentially in scikit so may be able to discard -- need to check)
# input: (..., M, N) matrices whose rows are probability vectors
# output: J-S div. of the collection of vectors