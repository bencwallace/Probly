###########
Quick start
###########

First, make sure that ProblY is `installed <https://bencwallace.github.io/installation.html>`_.

******************************
A note on reproducible results
******************************

The following examples consist of code segments (prefaced by >>>) followed by
expected outputs (no >>>). In order to reproduce the same results, the code
segments must be executed in the prescribed order. Other code segments may
be added between successive segments as long as they do not involve the
instantiation of random variables.

For more information, see :ref:`independence`.

***********************
Simple random variables
***********************

We begin by importing ``probly``.

>>> import probly as pr

.. testsetup::

   from probly.core import RandomVar
   RandomVar.reset()

Next, we initialize some pre-packaged random variables.
A complete list of available distributions is available at the :ref:`api`. See
also also the :class:`~ProblY.RandomVar` documentation if you want to create your own
random variables from scratch.

>>> # A Bernoulli random variable with p=0.5 (the default)
>>> X = pr.Ber()
>>> # A Bernoulli random variable independent of X
>>> Y = pr.Ber(0.9)
>>> # A uniform random variable on the interval [*10, 10]
>>> Z = pr.Unif(-10, 10)

Calling a random variable produces a random sample from its distribution.
In order to obtain reproducible results, we pass a seed as an argument to
the random variable. Calling the same random variable with the same seed
will produce the same result.

>>> seed = 99	# An arbitrary but fixed seed
>>> Z(seed)
-4.340731821079555
>>> # Repeat the last step to obtain the same output

When called with no argument, a seed produced by a random number generator
is used and the output is not reproducible.

>>> # We obtained the following output. You'll probably get something different.
>>> Z() # doctest: +SKIP
-7.722714026707818

Nevertheless, the outputs of ``Z()`` are uniformly distributed on the
interval ``[-10, 10]``. Similarly, we can check ``X`` is equally likely
to take on the values ``0`` and ``1`` by computing its empirical mean.

>>> trials = 1000
>>> total = 0
>>> for i in range(trials):
...     total += X(i)
>>> average = total / trials

The following output is close to ``0.5`` as expected by the
`law of large numbers <https://en.wikipedia.org/wiki/Law_of_large_numbers>`_.

>>> average
0.476

The output should be close to ``0.5`` for most seed choices. Try running the
code above with a few different seeds to see this (this will not affect
reproducibility).	

**************************
Random variable arithmetic
**************************
Random variables can be combined via arithmetical operations. The
result of such operations is itself a random variable whose
distribution may not be know explicitly.

>>> W = (1 + X) * Z / (1 + Y)
>>> # W is a new random object
>>> type(W)
<class 'probly.core.RandomVar'>

We can nevertheless sample from this unknown distribution!

>>> W(seed)
-4.340731821079555

Note that ``W`` is *dependent* on ``X``, ``Y``, and ``Z``.
This essentially means that the following outputs ``True``.

>>> x = X(seed)
>>> y = Y(seed)
>>> z = Z(seed)
>>> w = W(seed)
>>> w == (1 + x) * z / (1 + y)
True

For more information, see :ref:`dependence`.

.. todo::

   Link to LLN and CLT examples.

Other arithmetical functions
============================
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

.. UU._id == 15
.. UU._offset == 1416695020

Notice that ``UU`` produces the same values as ``W`` for a given seed
although they are different objects. This is because, although they
are distinct from the perspective of the Python interpreter, they are
the same random variables from the perspective of probability.

.. todo::

   Discuss making independent copies.

***************
Random matrices
***************
Random NumPy arrays (in particular, random matrices) can be formed from
other random variables.

>>> M = pr.array([[X, Z], [W, Y]])
>>> type(M)
<class 'probly.core.RandomVar'>

Random arrays can be manipulated like ordinary NumPy arrays.

>>> M[0, 0](seed) == X(seed)
True
>>> import numpy as np
>>> S = np.sum(M)
>>> S(seed) == X(seed) + Z(seed) + W(seed) + Y(seed)
True

*****************
Lifting functions
*****************
Certain functions don't work automatically with random variables.
However, any functions can be lifted to maps between random variables
using the
``@pr.Lift`` decorator.

>>> Det = pr.Lift(np.linalg.det)
>>> D = Det(M)

An equivalent way of doing this is as follows::

	import numpy as np
	@pr.Lift
	def Det(m):
		return np.linalg.det(m)

The function ``Det`` can now be applied to ``M``.

>>> D = Det(M)
>>> D(seed)
-17.841952742532634
