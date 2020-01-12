###########
Quick start
###########

************
Installation
************

Probly can be installed using `pip <https://pypi.org/project/pip/>`_ from GitHub as follows::

   pip install git+https://github.com/bencwallace/probly#egg=probly

.. note::

   Probly makes use of `NumPy <http://www.numpy.org/>`_, `SciPy <https://www.scipy.org/>`_, and `Matplotlib <https://matplotlib.org/>`_.

***************
Getting started
***************

We begin by importing ``probly``.

>>> import probly as pr

Next, we initialize some pre-packaged random variables.
A complete list of available distributions is available at the :ref:`api`.

>>> # A Bernoulli random variable with parameter 0.5
>>> X = pr.Ber()
>>> # A Bernoulli random variable independent of X with parameter 0.9
>>> Y = pr.Ber(0.9)
>>> # A uniform random variable on the interval [-10, 10]
>>> Z = pr.Unif(-10, 10)

Calling a random variable produces a random sample from its distribution.
In order to obtain reproducible results, we pass a seed as an argument to
the random variable. Calling the same random variable with the same seed
will produce the same result.

>>> seed = 99	# An arbitrary but fixed seed
>>> Z(seed)
-4.340731821079555
>>> Z(seed)
-4.340731821079555

.. note::

   An entire Probly session can be seeded by using ``pr.seed``. This will determine the sequence of outputs produced
   by sampling a sequence of random variables initialized in a given order with a given sequence of seeds; it is
   distinct from seeding the random variables themselves.

Random variables can be combined via arithmetical operations.

>>> W = (1 + X) * Z / (5 + Y)
>>> # W is a new random object
>>> type(W)
<class 'probly.core.RandomVariable'>

The result of such operations is itself a random variable whose
distribution may not be know explicitly.
We can nevertheless sample from this unknown distribution!

>>> W(seed)
-1.4469106070265185

We can also compute properties of a random variable, such as its mean.

>>> W.mean()
0.023611159797914952

**********
Dependence
**********
Note that ``W`` is *dependent* on ``X``, ``Y``, and ``Z``.
This essentially means that the following must output ``True``.

>>> x = X(seed)
>>> y = Y(seed)
>>> z = Z(seed)
>>> w = W(seed)
>>> w == (1 + x) * z / (5 + y)
True

******************
Independent copies
******************
Separate instantiations of a random variable will produce independent copies: for instance, samples from two
instantiations of a normal random variable will be independent of one another, even with the same seed.

>>> pr.Normal()(seed)
-0.8113001427396095
>>> pr.Normal()(seed)
0.09346601550504334

Independent copies of a random variable can also be produced as follows.

>>> Wcopy = W.copy()
>>> Wcopy(seed)
2.430468450181704

***************
Random matrices
***************
Random NumPy arrays (in particular, random matrices) can be formed from
other random variables.

>>> M = pr.array([[X, Z], [W, Y]])
>>> type(M)
<class 'probly.core.RandomVariable'>

Random arrays can be manipulated like ordinary NumPy arrays.

>>> M[0, 0](seed) == X(seed)
True
>>> import numpy as np
>>> S = np.sum(M)
>>> S(seed) == X(seed) + Z(seed) + W(seed) + Y(seed)
True

********************
Function application
********************
Any functions can be lifted to a map between random variables
using the ``@pr.lift`` decorator.

>>> from numpy.linalg import det
>>> det = pr.lift(det)

An equivalent way of doing this is as follows::

	@pr.lift
	def det(m):
		return np.linalg.det(m)

The function ``det`` can now be applied to ``M``.

>>> D = det(M)
>>> D(seed)
-5.280650914177544

************
Conditioning
************
Random variables can be conditioned as in the following example:

>>> C = W.given(Y == 1, Z > 0)
>>> C(seed)
1.97965814796514

Any boolean-valued random variable can be used as a condition.

*****************
Random parameters
*****************
Random variables can themselves be used to parameterize other random variables, as in the following example:

>>> U = pr.Unif()
>>> B = pr.Ber(U)
>>> B(seed)
0
