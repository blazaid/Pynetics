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
"""Recombination algorithms for list-based genotypes.
"""
import copy
import random
from typing import Tuple

from .genotype import ListGenotype
from ..util import take_chances


class NPivot:
    """Progeny obtained by mixing parents with N random pivot points.

    The process is as follows. N random points belonging to the [1, L-1]
    interval (being L the size of the genotypes) is selected. Then, one
    by one, the contents of both individuals are interchanged each time
    a new pivot point is reached. For example:

    pivots    : 3, 5
    parents  : XXXXXXXX, OOOOOOOO
    -----------
    children : XXXOOXXX, OOOXXOOO

    The method implementation has been slightly modified to allow the
    recombination of different length genotypes. In this case, L will be
    the length of the shortest genotype.

    Also, if the specified pivots number is greater than L-1, then L-1
    will be used as the new N.
    """

    def __init__(self, *, num_pivots: int):
        """Initializes this object.

        :param num_pivots: The number of pivot points for this operator.
            It is expected to be a positive number.
        """
        self.num_pivots = num_pivots

    def __call__(
            self, parent1: ListGenotype, parent2: ListGenotype
    ) -> Tuple[ListGenotype, ListGenotype]:
        """The specific crossover logic.

        :param parent1: One of the genotypes.
        :param parent2: The other.
        :return: A tuple with the progeny.
        """
        # Clone both parents
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Get the length of the smallest parent
        length = min(len(child1), len(child2))

        # Bound the pivot points number, in case is too high
        num_pivots = min(length - 1, self.num_pivots)

        # Select uniformly the pivots
        pivots = sorted(random.sample(range(1, length), num_pivots))

        # Swap contents
        next_pivot = pivots.pop(0)
        swap = False
        for i in range(length):
            # If we have reached the next pivot, switch the swap
            # behavior and set the next pivot position
            if i == next_pivot:
                swap = not swap
                next_pivot = pivots.pop(0) if pivots else length
            # If the swap behavior is enabled, do the crossover
            if swap:
                child1[i], child2[i] = child2[i], child1[i]

        return child1, child2


one_point_crossover = NPivot(num_pivots=1)
two_point_crossover = NPivot(num_pivots=2)


def pmx(
        parent1: ListGenotype, parent2: ListGenotype,
) -> Tuple[ListGenotype, ListGenotype]:
    """TODO TBD...

    :param parent1: One of the genotypes.
    :param parent2: The other.
    :return: A tuple with the progeny.
    """
    # Clone both parents and empty
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    # Select two random pivot points (beware with the genotype lengths)
    size = min(len(child1), len(child2))
    p1, p2 = random.sample(range(size + 1), 2)
    if p1 > p2:
        p1, p2 = p2, p1
    # Swap the loci between children and empty the rest
    mapping_child1, mapping_child2 = {}, {}
    for i in range(size):
        if i < p1 or i >= p2:
            child1[i] = child2[i] = None
        else:
            child1[i], child2[i] = child2[i], child1[i]
            mapping_child1[child1[i]] = child2[i]
            mapping_child2[child2[i]] = child1[i]
    # Fill all those spaces with genes if it's possible
    for parent, child in ((parent1, child1), (parent2, child2)):
        for i, (p, g) in enumerate(zip(parent, child)):
            if g is None and p not in child:
                child[i] = p
    # Fill those conflicting genes with the first value available
    for parent, child, mapping in (
            (parent1, child1, mapping_child1),
            (parent2, child2, mapping_child2),
    ):
        for i, (p, g) in enumerate(zip(parent, child)):
            if g is None:
                g = mapping[p]
                while g in child:
                    g = mapping[g]
                child[i] = g

    return child1, child2


def random_mask(
        parent1: ListGenotype, parent2: ListGenotype
) -> Tuple[ListGenotype, ListGenotype]:
    """Progeny is created by using a random mask.

    The random mask is generated to determine which genes are switched
    between genotypes. For example:

    random mask : 00100110
    parents     : XXXXXXXX, OOOOOOOO
    -----------
    children    : XXOXXOOX, OOXOOXXO

    The method implementation has been slightly modified to allow the
    recombination of different length genotypes. In this case, the mask
    will only affect to the minimum length, leaving the rest of the
    genes untouched.

    :param parent1: One of the genotypes.
    :param parent2: The other.
    :return: A tuple with the progeny.
    """
    # Clone both parents
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

    # Get the length of the smallest parent
    length = min(len(child1), len(child2))

    # A random mask is equivalent to a 50% chance to switch genes
    for i in range(length):
        if not take_chances(.5):
            child1[i], child2[i] = child2[i], child1[i]

    return child1, child2
