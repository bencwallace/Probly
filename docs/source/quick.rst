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

   Since different instances of a random variable are independent, your samples from a distribution (even with the
   same seed) may produce different results from those in this text. Nevertheless, a single instance sampled multiple
   times with the same seed will always produce the same result.

********************
Symbolic computation
********************
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

For composite random variables like ``W``, the ``mean`` method returns an approximate
value.

>>> W.mean()
0.023611159797914952

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

>>> M = pr.RandomArray([[X, Z], [W, Y]])
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

>>> Det = pr.lift(np.linalg.det)

An equivalent way of doing this is as follows::

	import numpy as np
	@pr.lift
	def Det(m):
		return np.linalg.det(m)

The function ``Det`` can now be applied to ``M``.

>>> D = Det(M)
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

********************
Custom distributions
********************

The following example shows how to create a custom distribution. We'll start by constructing a simple non-random
class.

>>> class Human:
>>>     def __init__(self, height, weight):
>>>         self.height = height
>>>         self.weight = weight

We'd like to create a kind of normal distribution over possible humans. We can do this as follows.

>>> import numpy as np
>>> from probly.distr.distributions import Distribution
>>> class NormalHuman(Distribution):
>>>     def __init__(self, female_stats, male_stats):
>>>         self.female_stats = female_stats
>>>         self.male_stats = male_stats
>>>         super().__init__()
>>>     def _sampler(self, seed):
>>>         np.random.seed(seed)
>>>         gender = np.random.choice(2, p=[0.5, 0.5])
>>>         if gender == 0:
>>>             height_mean, weight_mean, cov = self.female_stats
>>>         else:
>>>             height_mean, weight_mean, cov = self.male_stats
>>>         means = [height_mean, weight_mean]
>>>         np.random.seed(seed)
>>>         height, weight = np.random.multivariate_normal(means, cov)
>>>         return Human(gender, height, weight)

Let's initialize an instance of this random variable.

>>> f_cov = np.array([[80, 5], [5, 99]])
>>> f_stats = [160, 65, f_cov]
>>> m_cov = np.array([[70, 4], [4, 11]])
>>> m_stats = [180, 75, m_cov]
>>> H = NormalHuman(f_stats, m_stats)

We can sample from and manipulate such a random variable as usual.

>>> @pr.lift
>>> def bmi(human):
>>>     return human.weight / (human.height / 100) ** 2
>>> BMI = bmi(H)
>>> BMI(seed)
23.57076738620301
