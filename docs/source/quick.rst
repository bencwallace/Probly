###########
Quick start
###########

************
Installation
************

Probly can be installed using `pip <https://pypi.org/project/pip/>`_ from GitHub as follows::

   pip install git+https://github.com/bencwallace/probly#egg=probly

.. note::

   Probly makes use of `NumPy <http://www.numpy.org/>`_.

***********************
Simple random variables
***********************

We begin by importing ``probly``.

>>> import probly as pr

.. testsetup::

   from probly.core import RandomVar
   RandomVar.reset()

Next, we initialize some pre-packaged random variables.
A complete list of available distributions is available at the :ref:`api`.
See also also the :class:`~Probly.RandomVar` documentation if you want to
create your own random variables from scratch.

.. note::

   The following examples consist of code segments (prefaced by >>>) followed by expected outputs (no >>>). In order to reproduce the same results, the code segments must be executed in the prescribed order. Other code segments may be added between successive segments as long as they do not involve the instantiation of random variables.

   For more information, see :ref:`independence`.

>>> # A Bernoulli random variable with p=0.5 (the default)
>>> X = pr.Ber()
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

*****************************
Manipulating random variables
*****************************
Random variables can be combined via arithmetical operations. The
result of such operations is itself a random variable whose
distribution may not be know explicitly.

>>> W = (1 + X) * Z / (5 + Y)
>>> # W is a new random object
>>> type(W)
<class 'probly.core.RandomVar'>

We can nevertheless sample from this unknown distribution!

>>> W(seed)
-1.4469106070265185

Note that ``W`` is *dependent* on ``X``, ``Y``, and ``Z``.
This essentially means that the following outputs ``True``.

>>> x = X(seed)
>>> y = Y(seed)
>>> z = Z(seed)
>>> w = W(seed)
>>> w == (1 + x) * z / (5 + y)
True

For more information, see :ref:`dependence`.

.. todo::

   Link to LLN and CLT examples.

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

We could also sum the elements of ``M`` as follows, but read the note below.

>>> T = np.sum([[X, Z], [W, Y]])
>>> T(seed) == S(seed)
True

.. note::

   Due to the way in which NumPy sums arrays and the recursive nature of a
   random variable's call method, summing a large collection
   of random variables has the potential to result in a ``RecursionError``.
   So, for example, instead of applying ``np.linalg.sum`` directly to an
   array or list ``array`` of random variables, it is preferable to convert
   this collection to a random variable by running
   ``np.linalg.sum(pr.array(collection))``.


********************
Function composition
********************
Certain functions don't work automatically with random variables.
However, any functions can be lifted to maps between random variables
using the
``@pr.Lift`` decorator.

>>> Det = pr.Lift(np.linalg.det)

An equivalent way of doing this is as follows::

	import numpy as np
	@pr.Lift
	def Det(m):
		return np.linalg.det(m)

The function ``Det`` can now be applied to ``M``.

>>> D = Det(M)
>>> D(seed)
-5.280650914177544
