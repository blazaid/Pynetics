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
"""Tests for the different diversity implementations."""
import pytest

from pynetics.list import ListGenotype
from pynetics.list.alphabet import (
    BINARY,
    GENETIC_CODE,
)
from pynetics.list.diversity import average_hamming, DifferentGenesInPopulation
from tests.test_api import DiversityTests


class TestAverageHamming(DiversityTests):
    def get_instance(self, **kwargs):
        return average_hamming

    @pytest.mark.parametrize('genes, expected', [
        ([(0,), (0,)], 0),
        ([(0,), (1,)], 1),
        ([(1,), (0,)], 1),
        ([(0, 0), (0, 1)], 0.5),
        ([(1, 0), (0, 0)], 0.5),
        ([(1, 1), (0, 0)], 1),
        ([(1, 1), (0, 1)], 0.5),
        ([(1, 1), (1, 0)], 0.5),
        ([(1, 1), (1, 1)], 0),
        ([(0, 0, 0, 0), (1, 1, 0, 0), (0, 0, 0, 1), (1, 1, 1, 1)], 0.62),
        ([('A', 'C', 'G'), ('T', 'C', 'A'), ('T', 'A', 'G')], 0.66),
    ])
    def test_calculate_diversity(self, genes, expected):
        genotypes = [ListGenotype(genes=g) for g in genes]
        diversity = self.get_instance()

        assert diversity(genotypes) == pytest.approx(expected, 10e-2)


class TestDifferentGenesInPopulation(DiversityTests):
    def get_instance(self, alphabet=None, **kwargs):
        return DifferentGenesInPopulation(alphabet=alphabet or GENETIC_CODE)

    @pytest.mark.parametrize('alphabet, genotypes, expected', [
        (BINARY, [
            ListGenotype(genes=(0, 0, 0, 0)),
            ListGenotype(genes=(0, 0, 0, 0)),
            ListGenotype(genes=(0, 0, 0, 0)),
            ListGenotype(genes=(0, 0, 0, 0)),
        ], 0),
        (BINARY, [
            ListGenotype(genes=(0, 0, 0, 0)),
            ListGenotype(genes=(0, 0, 0, 0)),
            ListGenotype(genes=(0, 0, 0, 0)),
            ListGenotype(genes=(0, 0, 0, 1)),
        ], 1 / 4),
        (BINARY, [
            ListGenotype(genes=(0, 0, 1, 0)),
            ListGenotype(genes=(0, 0, 0, 0)),
            ListGenotype(genes=(0, 0, 1, 0)),
            ListGenotype(genes=(0, 0, 0, 1)),
        ], 1 / 2),
        (BINARY, [
            ListGenotype(genes=(1, 0, 0, 0)),
            ListGenotype(genes=(0, 0, 0, 0)),
            ListGenotype(genes=(0, 0, 1, 0)),
            ListGenotype(genes=(0, 0, 0, 1)),
        ], 3 / 4),
        (GENETIC_CODE, [
            ListGenotype(genes=('A', 'C', 'G')),
            ListGenotype(genes=('T', 'C', 'A')),
            ListGenotype(genes=('T', 'A', 'G')),
        ], 1 / 3),
    ])
    def test_calculate_diversity(self, alphabet, genotypes, expected):
        diversity = self.get_instance(alphabet=alphabet)

        assert diversity(genotypes) == expected
