"""Utility function
"""
import numpy as np
from foolbox.distances import Distance

class Ltwo(Distance):
    """Calculates the mean squared error between two images.
    """

    def _calculate(self):
        min_, max_ = self._bounds
        n = self.reference.size
        f = n * (max_ - min_)**2

        diff = self.other - self.reference
        value = np.linalg.norm(diff) / np.linalg.norm(self.reference)

        # calculate the gradient only when needed
        self._g_diff = diff
        self._g_f = f
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        if self._gradient is None:
            self._gradient = self._g_diff / (self._g_f / 2)
        return self._gradient

    def __str__(self):
        return 'normalized MSE = {:.2e}'.format(self._value)
