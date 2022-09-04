# Probly

by Benjamin Wallace

## Description

Probly is a Python framework for working with random variables.

Probly shifts the emphasis in computational statistics from **probability distributions**—typically
implemented as *functions* that can be called on to produce concrete samples—to **random variables**—*objects* that
represent the potential outcome of a sampling procedure and that can be manipulated in the same ways as these outcomes.
Typically, this means that random variables are subject to the usual laws of arithmetic. This is accounted for by Probly,
which approaches computational statistics from the point of view of symbolic computation.

Here are some examples::

```python
import numpy as np
import probly as pr

// initialize a uniform random variable on [0, 1]
A = pr.Unif()

// sample a random variable
A()

// initialize a normal random variable with mean 2
B = pr.Normal(2)

// sample a random variable given a seed
B(11)

// perform arithmetic on random variables
C = (X + Y) / 2

// condition random variables
D = Z.given(Y > 0)

// produce a random sequence of iid copies of a random variable
E = pr.iid(D, 1000)

// apply functions to random variables
F = np.sum(E)

// construct random variables with random parameters
G = pr.Ber(pr.Unif())

// define custom models
@pr.model('a', 'b')
def SquareOfUniform(a, b):
    def sampler(seed):
        np.random.seed(seed)
        return np.random.uniform() ** 2
    return sampler
    H = SquareOfUniform(2, 3)
```

## [Learn more](https://probly.readthedocs.io/en/latest/quick.html)
