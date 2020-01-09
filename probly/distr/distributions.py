"""
Random variables following common distributions.
"""

from ..core.random_variables import RandomVariableWithIndependence


class Distribution(RandomVariableWithIndependence):
    def __init__(self):
        op = self._sampler
        super().__init__(op)
