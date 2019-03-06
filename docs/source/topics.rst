###############
Assorted topics
###############

.. _independence:

************
Independence
************

In order to allow for independent but identically distributed random variables,
each instance of a random variable is initialized in such a way as to behave
differently from other instances, *even when all instances are seeded the same
way*. Take the following for example.

.. testsetup::

   from probly.core import RandomVar
   RandomVar.reset()

>>> import probly as pr
>>> X = pr.Unif(0, 1)
>>> Y = pr.Unif(0, 1)
>>> X(10) == Y(10)	# True with probability almost 0
False

Nevertheless, it is desirable in many instances to produce outputs that are
reproducible. For that reason, each random variable is equipped with an `_id`
attribute that increases in steps of 1 every time an independent random variable
(i.e. a random variable not defined in terms of other random variables) is
initialized. This makes outputs reproducible as long as they are generated in
the same order and without interruption by the initialization of new random variables.

>>> x = X(0)
>>> x 	# Produces the following value if no other random variables have been initialized
0.44334357301565486
>>> X = pr.Unif(0, 1)	# Initializes a new random variable
>>> X(0) == X() 		# True with probability almost 0
False

.. _dependence:

**********
Dependence
**********

Probly tracks the *dependence structure* of random variables.
This is a very important feature of Probly. For instance, this
ensures that a random variable minus itself is alsways ``0``!

>>> import probly as pr
>>> X = pr.Unif(0, 1)
>>> Y = X - X
>>> Y() == 0.0		# Outputs True for any seed
True

This is clearly different from sampling from the following.

>>> seed = 11
>>> X(seed) - X(seed + 1) == 0		# Typically False
False

Here's another example where we want dependence structure to be maintained.

>>> import probly as pr
>>> X = pr.Unif(0, 1)
>>> Y = pr.array([X, X + 1])
>>> Z = Y[1] - Y[0]
>>> Z() == 1
True
