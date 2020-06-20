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
"""Specific implementations for binary alphabet based algorithms.
"""
import copy
import random
from typing import Tuple

from pynetics.list.genotype import ListGenotype


# ~~~~~~~~~~~~~~~~~~~~~
# Recombination schemas
# ~~~~~~~~~~~~~~~~~~~~~
def generalised_crossover(
        parent1: ListGenotype,
        parent2: ListGenotype,
) -> Tuple[ListGenotype, ListGenotype]:
    """Progeny obtained by crossing genotypes as if they where integers.

    This recombination algorithm is expected to be used with binary
    genotypes of the same length. It may work with other kind of
    individuals, but the outcome is unpredictable.

    :param parent1: One of the genotypes.
    :param parent2: The other.
    :return: A tuple with the progeny.
    """
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

    # Obtain the crossover range (as integer values)
    a = int(''.join([str(b1 & b2) for (b1, b2) in zip(child1, child2)]), 2)
    b = int(''.join([str(b1 | b2) for (b1, b2) in zip(child1, child2)]), 2)

    # Get the children (as integer values)
    c = random.randint(a, b)
    d = b - (c - a)

    # Convert to binary lists again (NOTE: we remove the starting 0b)
    bin_formatter = '{:#0' + str(len(child1) + 2) + 'b}'
    bin_c = [int(x) for x in bin_formatter.format(c)[2:]]
    bin_d = [int(x) for x in bin_formatter.format(d)[2:]]

    # Convert to chromosomes and we're finish
    for i in range(len(bin_c)):
        child1[i], child2[i] = bin_c[i], bin_d[i]
    return child1, child2
