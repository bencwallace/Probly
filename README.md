# ProblY
## Probabilistic computations with Python

This Python module was written to allow the easy and intuitive manipulation of random variables according to the following principle:
> Individual random variables should be treated as **numeric types**.

This means treating them as objects that can be composed with functions of a number, array, or matrix and, in particular, allowing arithmetical operations to be performed upon them. Moreover, an immediate requirement for any methodology allowing such a treatment of random variables is the implementation of a *dependence structure* that tracks the relationships between different instances of a random variable (more on this below).

## Comparison with other approaches

Other Python implementations of random variables, such as those in `scipy.stats`, are typically geared towards statistical applications and therefore tend to treat random variables as collections of properties (mean, variance, etc.) and methods (pdf, cdf, etc.). It is therefore difficult to perform operations on them as these properties and methods can be difficult to compute for transformed random variables. ProblY trades the convenience of rapid access to such properties in exchange for flexibility in the kinds of random variables that can be constructed by implementing a very general class whose objects' main function is to produce random numbers from some distribution.

## Examples

Examples can be found in the Jupyter [notebook](tour.ipynb).

### Dependencies

ProblY makes use of [NetworkX](https://networkx.github.io/).
