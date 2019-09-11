<pre>
S I G N A L   C A L C U L U S
</pre>
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
***Do not re-order any of the imports***
Imports modules. Initialises package. Contains examples of running attribute generation and HAG (height above ground).

* timeIntervals dictates how many subtiles we have - this helps keep down memory usage
* k should be a generator which tells us which values of k we will entertain. In the example we look at 4-49.
* radius tells us the maximum size of a neighbourhood, which could be np.inf (no bound)
* virtualSpeed is a weighting for how much time between points should affect their closeness
* use virtualSpeed = 0 to eliminate spacetime usage
* if just one value of k is given, optimisation will not occur

## Geo
Pretty important module. This is contains the mathematical heart of neighbourhood definition and the step thereafter: Structure tensor (covariance matrices) and the eigeninformation.

### optimise_k
Selects a k value, k>3, for each point, from a given iterable. That is, this step computes the *neighbourhood definition*. 

### eig
Splits the tile into many time slices, without edge effects, in order to decrease memory usage. It applies the neighbourhood definition learnt from optimise_k in order to deduce structure tensor, and hence eigeninformation.

### hag
Finds height above ground using 2d voxels. Takes 100(1+alpha)-th percentile of z. Perhaps could be improved by including a dual rank operator instead of bottom percentile.

## infotheory

### entropy
Computes entropy.
* **input**(..., N) probability vectors
* that is, we must have np.sum(distribution, axis = -1)==1 and distribution>=0
* **output** entropies of vectors

### jsd (potentially in scikit so may be able to discard -- need to check)
Computes Jensen-Shannon Divergence (no weights).
* **input** (..., M, N) matrices whose rows are probability vectors
* **output** J-S div. of the collection of vectors

## fun
This contains lambda-terms which defines the functions of eigeninformation. We can actually change those function defintions, but these are the WPD69-2017 standard choices.

## lpinteraction
Governs the interaction with laspy (other modules are agnostic of file interaction).

### attr
Adds attributes as defined by function definitions (usually those in the **fun** module).

### add_hag
Self-explanatory.

### nfl
Discards points a chosen distance from a chosen classification. We use this to find the points *near the flight line*, hence the name.