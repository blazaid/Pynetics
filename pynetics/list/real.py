# ======================================================================
# pynetics: a simple yet powerful evolutionary computation library
# Copyright (C) 2020 Alberto Díaz-Álvarez
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# “Software”), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
# THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ======================================================================
"""TODO TBD...
"""
import copy
import random
from typing import Tuple

from .genotype import ListGenotype
from .initializer import IntervalInitializer
# ~~~~~~~~~~~~
# Initializers
# ~~~~~~~~~~~~
from ..exception import BoundsCannotBeTheSame


class RealIntervalInitializer(IntervalInitializer):
    """Initializer for int based ListGenotype instances."""

    def get_value_from_interval(self) -> float:
        """Generates a new value that belongs to the interval.

        This value will be an integer value.

        :return: A value for the specified interval at initialization.
        """
        return random.uniform(self.lower, self.upper)


# ~~~~~~~~~~~~~~~~~~~~~
# Recombination schemas
# ~~~~~~~~~~~~~~~~~~~~~
class FlexibleRecombination:
    """TODO TBD..."""

    def __init__(self, *, lower: float, upper: float, alpha: float):
        """TODO TBD...

        :param alpha: TODO TBD...
        :param lower: The lower bound for the genes in the genotype. If
            its value is greater than the upper bound, the values are
            switched.
        :param upper: The upper bound for the genes in the genotype.
        :raise BoundsCannotBeTheSame: If the two bounds have the same
        value.
        """
        if lower == upper:
            raise BoundsCannotBeTheSame(lower)
        self.lower, self.upper = min(lower, upper), max(lower, upper)
        self.alpha = alpha

    def __call__(
            self,
            child1: ListGenotype,
            child2: ListGenotype,
    ) -> Tuple[ListGenotype, ListGenotype]:
        """Implements the specific crossover logic.

        :param child1: One of the genotypes.
        :param child2: The other.
        :return: A tuple with the progeny.
        """
        for i, (lower, upper) in enumerate(zip(child1, child2)):
            if lower > upper:
                lower, upper = upper, lower

            lower = max(self.lower, lower - self.alpha)
            upper = min(self.upper, upper + self.alpha)

            child1[i] = random.uniform(lower, upper)
            child2[i] = upper - (child1[i] - lower)
        return child1, child2


def plain_recombination(
        parent1: ListGenotype,
        parent2: ListGenotype,
) -> Tuple[ListGenotype, ListGenotype]:
    """TODO TBD...

    :param parent1: One of the genotypes.
    :param parent2: The other.
    :return: A tuple with the progeny.
    """
    # Clone both parents
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

    for i, (lower, upper) in enumerate(zip(child1, child2)):
        child1[i] = random.uniform(lower, upper)
        child2[i] = upper - (child1[i] - lower)
    return child1, child2
