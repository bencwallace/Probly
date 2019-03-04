"""
Random variables
================

This modules defines `RandomVar` objects as nodes in the dependency graph
(`Node` objects) with extra structure allowing them to be transformed and
operated on in whatever way is compatible with their realizations.
"""

# def array(arr):
#     """
#     Turns a collection of random variables and constants into a random array.

#     Parameters
#     ----------
#     arr (array_like)
#         An `array_like` object of `RandomVar` objects, constants, and other
#         `array_like` objects.

#     Returns
#     -------
#     RandomVar
#         A random variable whose samples are arrays of samples of the objects
#         in `arr`.
#     """

#     arr = [RandomVar._cast(var) for var in arr]

#     @Lift
#     def make_array(*args):
#         return np.array(args)

#     return make_array(*arr)
