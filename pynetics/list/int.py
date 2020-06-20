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


class IntegerIntervalInitializer(IntervalInitializer):
    """Initializer for int based ListGenotype instances."""

    def get_value_from_interval(self) -> int:
        """Generates a new value that belongs to the interval.

        This value will be an integer value.

        :return: A value for the specified interval at initialization.
        """
        return random.randint(self.lower, self.upper)


# ~~~~~~~~~~~~~~~~~~~~~
# Recombination schemas
# ~~~~~~~~~~~~~~~~~~~~~
class RangeCrossover:
    """Offspring is obtained by crossing individuals gene by gene.

    For each gene, the interval of their values is calculated. Then, the
    difference of the interval is used for calculating the new interval
    from where to pick the values of the new genes. First, a value is
    taken from the new interval. Second, the other value is calculated
    by taking the symmetrical by the center of the range.

    It is expected for the genotypes to have the same length. If not,
    the operation works over the first common genes, leaving the rest
    untouched.
    """

    def __init__(self, *, lower: int, upper: int):
        """Initializes this object.

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

    def __call__(
            self,
            parent1: ListGenotype,
            parent2: ListGenotype,
    ) -> Tuple[ListGenotype, ListGenotype]:
        """Implements the specific crossover logic.

        :param parent1: One of the genotypes.
        :param parent2: The other.
        :return: A tuple with the progeny.
        """
        # Clone both parents
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        for i, (a, b) in enumerate(zip(child1, child2)):
            # For each gene, we calculate the the crossover interval. If
            # the genes are equal, we take the whole possible interval
            if a != b:
                diff = abs(a - b)
            else:
                diff = self.upper - self.lower
            # Now the genes values
            gene_1 = random.randint(a - diff, b + diff)
            gene_2 = a + b - gene_1
            # Ensure the gene values belong to the interval
            child1[i] = max(min(gene_1, self.upper), self.lower)
            child2[i] = max(min(gene_2, self.upper), self.lower)
        return child1, child2
