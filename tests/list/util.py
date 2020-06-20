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
"""Some utility functions common to the list genotype implementation
tests."""

from pynetics.list.genotype import ListGenotype


def equal_genes_in_genotypes(genotype_1, genotype_2):
    """Compares the genes of two genotypes.

    :param genotype_1: One genotype.
    :param genotype_2: The other.
    """
    for gene1, gene2 in zip(genotype_1, genotype_2):
        assert gene1 == gene2


def execute_mutation_test(p, mutation, genes, expected):
    """Base function to execute recombination tests.

    :param p: The mutation probability.
    :param mutation: Instance used to perform the mutation.
    :param genes: The genes of the genotype to mutate.
    :param expected: The expected result after the mutation.
    """
    mutated = mutation(p, ListGenotype(genes=genes))
    expected = ListGenotype(genes=expected)

    equal_genes_in_genotypes(mutated, expected)
