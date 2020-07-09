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
"""Generic mutation schemas for list-based genotypes.
"""
import abc
import copy
import random

from . import ListGenotype
from .alphabet import Alphabet
from ..util import take_chances


class PerGeneMutation:
    """General behaviour for a mutation applied per gene."""

    def __call__(
            self, p: float, genotype: ListGenotype
    ) -> ListGenotype:
        """Performs the mutation schema.

        :param p: The probability of mutation.
        :param genotype: The genotype to be mutated.
        :return: A new mutated genotype. If no mutation was performed,
            the same genotype instance is returned.
        """
        clone = None
        for i, gene in enumerate(genotype):
            # Check if a mutation occurs over this gene
            if take_chances(probability=p):
                # Check if this is the first mutation and clone the
                # original genotype if it is
                if clone is None:
                    clone = copy.deepcopy(genotype)
                # Apply the specific mutation over the gene
                self.do(clone, i)

        # Return either the mutated genotype or the original one if no
        # mutation was performed
        return clone or genotype

    @abc.abstractmethod
    def do(self, genotype: ListGenotype, index: int):
        """Performs the specific mutation

        :param genotype: the genotype to mutate.
        :param index: which gene is affected.
        """


class RandomGene(PerGeneMutation):
    """Mutates the genotype by changing some genes values.

    For each gene the mutation probability will be check and, if a
    mutation is requested, that gene will be replaced by a random one
    extracted from the alphabet.

    genotype  : aacgaata
    alphabet  : (a, c, t, g)
    mutate in : 2, 7
    -----------
    mutated    : abcdaaba
    """

    def __init__(self, alphabet: Alphabet, same: bool = False):
        """ Initializes this object.

        :param alphabet: The set of values to choose from.
        :param same: If it is allowed to select the same gene for
            replacement from the alphabet. Defaults to False (i.e. the
            new gene is always different).
        """
        self.alphabet = alphabet
        self.same = same

    def do(self, genotype: ListGenotype, index: int):
        """Performs the specific mutation

        :param genotype: the genotype to mutate.
        :param index: which gene is affected.
        """
        # Select a different index to swap with
        new_gene = self.alphabet.get()
        while not self.same and genotype[index] == new_gene:
            new_gene = self.alphabet.get()
        genotype[index] = new_gene


class SwapGenes(PerGeneMutation):
    """Mutates the genotype by swapping two random genes.

    For each gene, a random value will be compared against the mutation
    probability and, if the value is lower, the mutation will take place
    and its value will be swapped with the value of another random gene
    in the genotype.

    For example, if the mutation occurs in the gene located in position
    3 and it has to be swapped with the one in position 5, then:

    genotype  : 12345678
    positions : 3, 5
    -----------
    mutated   : 12365478
    """

    def do(self, genotype: ListGenotype, index: int):
        """Performs the specific mutation

        :param genotype: the genotype to mutate.
        :param index: which gene is affected.
        """
        # Select a different index to swap with
        j = random.randint(0, len(genotype) - 1)
        while j == index:
            j = random.randint(0, len(genotype) - 1)
        # Swap the genes in the cloned genotype
        genotype[index], genotype[j] = genotype[j], genotype[index]
