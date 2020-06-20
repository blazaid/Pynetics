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
"""Tests for the recombination operators."""
from unittest import mock

import pytest

from pynetics.list.alphabet import GENETIC_CODE
from pynetics.list.mutation import RandomGene, swap_genes
from tests.list.util import execute_mutation_test
from tests.util import random_sequence


class TestRandomGene:
    @pytest.mark.parametrize('alphabet, same, p, genes, expected, choice', [
        (GENETIC_CODE, False, 1.0, 'AAAAAAAAAA', 'CCCCCCCCCC', 'AC'),
        (GENETIC_CODE, True, 1.0, 'AAAAAAAAAA', 'ACACACACAC', 'AC'),
    ])
    def test_mutate_genotype(self, alphabet, same, p, genes, expected, choice):
        seq = random_sequence(choice)
        mutation = RandomGene(alphabet=alphabet, same=same)
        with mock.patch('random.choice', side_effect=lambda _: next(seq)):
            execute_mutation_test(p, mutation, genes, expected)


class TestSwapGenes:
    @pytest.mark.parametrize('probability, genes, expected, rand, choice', [
        (0.5, 'AAAAAAAAAA', 'AAAAAAAAAA', [0.4, 0.6], [5, 5, 4, 3]),
        (0.5, 'ACACACACAC', 'CCAACAACAC', [0.4, 0.6], [5, 5, 4, 3]),
        (0.5, 'ACTGACTGAC', 'CCAAGATGTC', [0.4, 0.6], [5, 5, 4, 3]),
    ])
    def test_mutate_genotype(self, probability, genes, expected, rand, choice):
        choice = random_sequence(choice)
        rand = random_sequence(rand)

        with mock.patch('random.choice', side_effect=lambda arg: next(choice)):
            with mock.patch('random.random', side_effect=lambda: next(rand)):
                execute_mutation_test(probability, swap_genes, genes, expected)
