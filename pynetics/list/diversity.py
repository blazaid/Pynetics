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
"""Implementations of different diversity algorithms for genotypes
represented by lists.
"""
import itertools
from typing import Sequence

from .alphabet import Alphabet
from ..list.genotype import ListGenotype


# ~~~~~~~~~~~~~~~~~~~
# Diversity operators
# ~~~~~~~~~~~~~~~~~~~
def average_hamming(genotypes: Sequence[ListGenotype]) -> float:
    """Average of each hamming loci distances.

    The diversity will be computed as follows:

    1. Calculate all the hamming distances between each pair of
        genotypes.
    2. Divide between the length of the genotypes.

    It is assumed that all the genotypes have the same length; if not,
    the diversity will be computed as if all the genotypes had the same
    length (the minimum among them).

    :param genotypes: A sequence of individuals from which obtain
        the diversity.
    :return: A float value representing the diversity.
    """

    def hamming_dist(str1: ListGenotype, str2: ListGenotype) -> int:
        """Computes the hamming distance between two strings.

        :param str1: One string.
        :param str2: The other.
        :return: The distance as int.
        """
        return len([1 for c1, c2 in zip(str1, str2) if c1 != c2])

    # Distances between each pair of genotypes
    distances = [
        hamming_dist(g1, g2)
        for g1, g2 in itertools.combinations(genotypes, 2)
    ]
    # Average
    return sum(distances) / (len(distances) * len(genotypes[0]))


# ~~~~~~~~~~~~~~~~~~~~~~~~
# Diversity meta-operators
# ~~~~~~~~~~~~~~~~~~~~~~~~
class DifferentGenesInPopulation:
    """Diversity based on the appearance of different genes regardless
    their position.

    The value is computed as follows. For each gene position, a value of
    M (the different appearing genes) is computed. Then, all the values
    are sum added and the divided by Y = N * L where L is the length of
    the individual and N the number of possible alleles. The value then
    is expected to belong to the interval [0, 1], where 0 is no
    diversity at all and 1 a completely diverse population.

    It is expected for the genotypes to have the same length.
    """

    def __init__(self, alphabet: Alphabet):
        """Initializes this object.

        :param alphabet: The alphabet with the different genes.
        """
        self.alphabet = alphabet

    def __call__(self, genotypes: Sequence[ListGenotype]) -> float:
        """Returns the diversity.

        :param genotypes: A sequence of individuals from which obtain
            the diversity.
        :return: A float value representing the diversity.
        """
        # Compute the total available genes
        total_diversity = len(self.alphabet)
        # Compute the different number of genes per genotype position
        genes_per_position = tuple(
            (len(set(x)) - 1) / (total_diversity - 1)
            for x in zip(*genotypes)
        )
        # Return the proportion
        return sum(genes_per_position) / len(genes_per_position)
