Quick start
===========

First, make sure that ProblY is `installed <https://bencwallace.github.io/installation.html>`_.

A note on reproducible results
------------------------------

The following examples consist of code segments (prefaced by >>>) followed by
expected outputs (no >>>). In order to reproduce the same results, the code
segments must be executed in the prescribed order. Other code segments may
be added between successive segments as long as they do not involve the
instantiation of random variables.

Simple distributions
--------------------

We begin by importing ``probly`` and creating some pre-packaged random variables.
A complete list of available distributions is available at the :ref:`api`.

>>> import probly as pr
>>> # A Bernoulli random variable with p=0.5
>>> X = pr.Ber(0.5)
>>> # A Bernoulli random variable independent of X
>>> Y = pr.Ber(0.9)
>>> # A uniform random variable on the interval [-10, 10]
>>> Z = pr.Unif(-10, 10)

Calling a random variable produces a random sample from its distribution.
In order to obtain reproducible results, we pass a seed as an argument to
the random variable. Calling the same random variable with the same seed
will produce the same result.

>>> seed = 99	# An arbitrary but fixed seed
>>> Z(seed)
0.3279725540489231
>>> # Repeat the last step to obtain the same output

When called with no argument, a seed produced by a random number generator
is used and the output is not reproducible.

>>> # We obtained the following output. You'll almost surely get something different
>>> Z()
4.364698077658325

Nevertheless, the outputs of ``Z()`` are uniformly distributed on the
interval ``[-10, 10]``. Similarly, we can check ``X`` is equally likely
to take on the values ``0`` and ``1``::

	trials = 1000
	total = 0
	for _ in range(trials):
	    total += X()
	average = total / trials

The following output is non-reproducible because we didn't fix a seed,
but it should still be close to ``0.5`` (by the
`law of large numbers <https://en.wikipedia.org/wiki/Law_of_large_numbers>`_).

>>> average
0.503

Combining random variables
--------------------------
Random variables can be combined via arithmetical operations. The
result of such operations is itself a random variable whose
distribution may not be know explicitly.

>>> W = (1 + X) * Z / (1 + Y)
>>> # W is a new random object
>>> type(W)
probly.randomvar.RandomVar

We can nevertheless sample from this unknown distribution!

>>> W(seed)
0.3279725540489231

Dependent random variables
--------------------------

So far, we have define four random variables. While ``X``,
``Y``, and ``Z`` are independent, ``W`` is clearly dependent
on all three.

The following demonstrates the fact that samples drawn from
dependent random variables are consistent with one another
for a given fixed seed.

>>> x = X(seed)
>>> y = Y(seed)
>>> z = Z(seed)
>>> w = W(seed)
>>> w == (1 + x) * z / (1 + y)
True

This is a very important feature of ProblY. For instance, this
ensures that a random variable minus itself is alsways ``0``!
The following example is reproducible despite the fact that
we don't fix a seed because ``Z - Z`` is constant.

>>> (Z - Z)()
0.0

This is clearly different from sampling from ``Z`` with two
different seeds and then subtracting.

>>> Z(seed) - Z(seed + 1)
-1.6256591446723405

Other arithmetical functions
----------------------------
Any function that acts on one of its arguments using only arithmetical
operations can be applied to a random variable to produce a new random
variable (the *composition* of the first random variable and the function)

>>> def f(x, y, z):
...     return (1 + x) * z / (1 + y)
>>> UU = f(X, Y, Z)
>>> UU(seed) == W(seed)
True
>>> UU is W
False

.. UU._id == 17

Notice that ``UU`` produces the same values as ``W`` for a given seed
although they are different objects. This is because, although they
are distinct from the perspective of the Python interpreter, they are
the same random variables from the perspective of probability.

Independent copies
------------------
Independent copies of a random variable can be produced using the ``copy``
method. This can be useful when the distribution of a random variable
isn't explicitly known.

>>> C = UU.copy()
>>> C(seed)
6.956875051242349

Random matrices
---------------
Random NumPy arrays (in particular, random matrices) can be formed from
other random variables.

>>> M = pr.array([[X, Z], [W, Y]])
>>> type(M)
probly.randomvar.RandomVar

Lifting functions
-----------------
Functions can be lifted to maps between random variables using the
``@pr.Lift`` decorator::

	import numpy as np
	@pr.Lift
	def Det(m):
		return np.linalg.det(m)

The function ``Det`` can now be applied to ``M``.

>>> D = Det(M)
>>> D(seed)
0.8924340037906262
