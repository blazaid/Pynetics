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
"""Operators for specific int representation of list-based genotypes.
"""
import copy
import random
from typing import Tuple, Optional

from . import ListGenotype
from .initializer import IntervalInitializer
from .mutation import PerGeneMutation
from ..exception import BoundsCannotBeTheSame
from ..util import take_chances


# ~~~~~~~~~~~~
# Initializers
# ~~~~~~~~~~~~
class IntegerIntervalInitializer(IntervalInitializer):
    """Initializer for int based ListGenotype instances."""

    def get_value_from_interval(self) -> int:
        """Generates a new value that belongs to the interval.

        This value will be an integer value.

        :return: A value for the specified interval at initialization.
        """
        return random.randint(self.lower, self.upper)


# ~~~~~~~~~~~~~~~~
# Mutation schemas
# ~~~~~~~~~~~~~~~~
class Creep(PerGeneMutation):
    """Mutates the genotype by adding a small value to some genes.

    This value may be positive or negative, and the resulting value may
    belong to a closer or upper interval.
    """

    def __init__(
            self, *,
            amount: int,
            fixed: Optional[bool] = True,
            lower: Optional[int] = None,
            upper: Optional[int] = None,
    ):
        """Initializes this object.

        :param amount: How much to add (or subtract) from the mutated
            gene. Must be greater or lower than zero.  As it will be
            added or subtracted, it'll be stored as a positive value.
        :param fixed: If the amount should be added or subtracted (True)
            or if the value should belong to the interval [1, amount)
            (False). If false, the value will be selected uniformly
            from the interval.
        :param lower: The lower bound for the genes in the genotype. If
            its not specified, the genes will not have a lower bound. If
            both the lower and upper limits are set
        :param upper: The upper bound for the genes in the genotype.
        :raise ValueError: If the amount to add to the mutated genes is
            zero.
        :raise BoundsCannotBeTheSame: If the two bounds have the same
            value.
        """
        if amount == 0:
            raise ValueError('The amount to add (or subtract) cannot be zero')
        if lower == upper and lower is not None:
            raise BoundsCannotBeTheSame(lower)

        self.amount = abs(amount)
        self.fixed = fixed
        if lower is None or upper is None:
            self.lower, self.upper = lower, upper
        else:
            self.lower, self.upper = min(lower, upper), max(lower, upper)

    def do(self, genotype: ListGenotype, index: int):
        """Performs the specific mutation

        :param genotype: the genotype to mutate.
        :param index: which gene is affected.
        """
        new_gene = genotype[index] + self.compute_amount()
        if self.lower is not None:
            new_gene = max(new_gene, self.lower)
        if self.upper is not None:
            new_gene = min(new_gene, self.upper)
        # Update the gene value
        genotype[index] = new_gene

    def compute_amount(self) -> int:
        """Compute the amount to add or subtract base on the arguments.

        :return: An integer value to add to the gene.
        """
        # Compute the amount depending on if it's variable or fixed.
        if self.fixed:
            amount = self.amount
        else:
            amount = random.randint(1, self.amount)
        # Then, randomize if we have to add or substract it
        if take_chances(probability=0.5):
            return amount
        else:
            return -amount


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
