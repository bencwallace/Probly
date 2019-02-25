# ProblY
by Benjamin Wallace

## Description

ProblY is a Python package for the symbolic computation of random variables.

In its most general sense, this means treating random variables as objects that can be composed with functions. Specifically, this makes random variables whose realizations are numerical types (scalars and arrays) subject to arithmetical manipulations. However, ProblY random variables can have more general types, e.g. random variables whose realizations are objects of a given class.

An immediate requirement and hence consequence of this approach of this treatment of random objects is the implementation of a *dependence structure* that tracks the relationships between different instances of a random variable. This will be discussed in more detail but a simple example is that if `X` is a random variable and we set `Y = X + 1` and `Z = Y - X` (thereby instantiating two new random variable objects), we expect that `Z` is the constant random variable `1`.

### Comparison with other approaches

**Easier sampling from complex distributions**

Other Python packages, such as NumPy and SciPy include methods for sampling from certain common distributions. It is then up to the user of these packages to define their own methods for sampling from more complex distributions. For instance, to sample from `X + \log(X)` where `X` is normally distributed, a user can define a method that set `x = np.random.normal()` and returns `x + np.log(x)`. However, this can become tedious if many different functions of possibly many different random variables are desired. With ProblY, all the user has to define is set `X` to be a ProblY normal random variable and sample from
`Y = X + log(X)` (itself a ProblY random variable).

**Dependence structure**

A second import feature of ProblY not found NumPy is SciPy is the automatic tracking of dependence structure so that one can expect results consistent with this structure when sampling from a collection of random variables or a random variable defined as a function of such a collection.
For instance, when sampling `Z = [X, X + log(X)]` (regardless of the distribution of `X`) one expects a sample with `Z[0] = 1` to also have `Z[1] = 1 + log(1) = 1`. A user using NumPy would require a certain level of caution not to get inconsistent results, whereas ProblY takes care of this in the background.

## Getting started

### Requirements

ProblY requires the following Python packages:
* [NumPy](http://www.numpy.org/)
* [NetworkX](https://networkx.github.io/)

### Installation

ProblY can be installed with `pip` as follows:
```bash
pip install git+https://github.com/bencwallace/probly#egg=probly
```
### Simple example

**Declaring simple random variables**
```python
import probly as pr

# A Bernoulli random variable with p=0.5
X = pr.Ber(0.5)

# A Bernoulli random variable independent of X
Y = pr.Ber(0.9)

# A uniform random variable on the interval [-10, 10]
Z = pr.Unif(-10, 10)
```

**Sampling from a distribution**
```python
# Output a random sample from X
print(X())

# Output a random sample from Y seeded with seed 0
print(Y(0))
```

**Performing arithmetic on random variables:**
```python
# Define a new random variable using arithmetic
W = (1 + X) * Z / (1 + Y)

# Outputs "True"
print(W(9) == (1 + X(9)) * Z(9) / (1 + Y(9)))
```

**Defining a random matrix**
```python
M = pr.rv.array([[X, Y], [Z, W]])
```

**Applying a function to a random variable**
```python
@pr.Lift
def f(x):
	return x[0, 0] - x[1, 1]

f_of_M = f(M)
print(f_of_M())
```

**Creating a custom random variable**
```python
import numpy as np
from probly.distr import Distr

class Human(object):
    def __init__(self, gender, height, weight):
        self.gender = gender
        self.height = height
        self.weight = weight

    def BMI(self):
        return self.weight / self.height ** 2


class randomHuman(Distr):
    def sampler(self, female_stats, male_stats):
        female_stats, male_stats = self.params

        gender = {0: 'F', 1: 'M'}[pr.Ber(0.5)()]
        if gender == 0:
            height_mean, weight_mean, cov = female_stats
        else:
            height_mean, weight_mean, cov = male_stats

        means = [height_mean, weight_mean]
        height, weight = np.random.multivariate_normal(means, cov)

        return Human(gender, height, weight)

f_cov = np.array([[80, 5], [5, 99]])
f_stats = [160, 65, f_cov]

m_cov = np.array([[70, 4], [4, 110]])
m_stats = [180, 75, m_cov]

H = randomHuman(f_stats, m_stats)

seed = 11
h = H(seed)
gender = {'F': 'female', 'M': 'male'}[h.gender]
height = h.height
weight = h.weight
print('H({}) is a {} human of height {:.2f}cm'
      ' and weight {:.2f}kg.'.format(seed, gender, height, weight))
```
