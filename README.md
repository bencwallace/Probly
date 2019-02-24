# ProblY
## Probabilistic computations with Python

ProblY is a Python package for the symbolic computation of random variables.

In its most general sense, this means treating random variables as objects that can be composed with functions. Specifically, this makes random variables whose realizations are numerical types (scalars and arrays) subject to arithmetical manipulations. However, ProblY random variables can have more general types, e.g. random variables whose realizations are objects of a given class.

An immediate requirement and hence consequence of this approach of this treatment of random objects is the implementation of a *dependence structure* that tracks the relationships between different instances of a random variable. This will be discussed in more detail but a simple example is that if `X` is a random variable and we set `Y = X + 1` and `Z = Y - X` (thereby instantiating two new random variable objects), we expect that `Z` is the constant random variable `1`.

## Comparison with other approaches

### Easier sampling from complex distributions

Other Python packages, such as NumPy and SciPy include methods for sampling from certain common distributions. It is then up to the user of these packages to define their own methods for sampling from more complex distributions. For instance, to sample from `X + \log(X)` where `X` is normally distributed, a user can define a method that set `x = np.random.normal()` and returns `x + np.log(x)`. However, this can become tedious if many different functions of possibly many different random variables are desired. With ProblY, all the user has to define is set `X` to be a ProblY normal random variable and sample from
`Y = X + log(X)` (itself a ProblY random variable).

### Dependence structure

A second import feature of ProblY not found NumPy is SciPy is the automatic tracking of dependence structure so that one can expect results consistent with this structure when sampling from a collection of random variables or a random variable defined as a function of such a collection.
For instance, when sampling `Z = [X, X + log(X)]` (regardless of the distribution of `X`) one expects a sample with `Z[0] = 1` to also have `Z[1] = 1 + log(1) = 1`. A user using NumPy would require a certain level of caution not to get inconsistent results, whereas ProblY takes care of this in the background.

## Examples

Examples can be found in the Jupyter [notebook](tour.ipynb) (under construction).

### Dependencies

ProblY makes use of [NumPy](http://www.numpy.org/) and [NetworkX](https://networkx.github.io/).
