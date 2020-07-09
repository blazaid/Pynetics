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
from unittest.mock import patch

import pytest

from pynetics.list import ListGenotype
from pynetics.list.recombination import NPivot, pmx, random_mask
from tests.util import random_sequence


class TestNPivot:
    @pytest.mark.parametrize('genes, random_pivots, expected', [
        (('111111', '000000'), ([1],), ('100000', '011111')),
        (('111111', '000000'), ([2],), ('110000', '001111')),
        (('111111', '000000'), ([3],), ('111000', '000111')),
        (('111111', '000000'), ([4],), ('111100', '000011')),
        (('111111', '000000'), ([5],), ('111110', '000001')),
        (('111111', '000000'), ([1, 2],), ('101111', '010000')),
        (('111111', '000000'), ([2, 3],), ('110111', '001000')),
        (('111111', '000000'), ([3, 4],), ('111011', '000100')),
        (('111111', '000000'), ([4, 5],), ('111101', '000010')),
        (('111111', '000000'), ([1, 3],), ('100111', '011000')),
        (('111111', '000000'), ([2, 4],), ('110011', '001100')),
        (('111111', '000000'), ([3, 5],), ('111001', '000110')),
        (('111111', '000000'), ([1, 4],), ('100011', '011100')),
        (('111111', '000000'), ([2, 5],), ('110001', '001110')),
        (('111111', '000000'), ([1, 5],), ('100001', '011110')),
        (('111111', '000000'), ([1, 2, 3],), ('101000', '010111')),
        (('111111', '000000'), ([2, 3, 4],), ('110100', '001011')),
        (('111111', '000000'), ([3, 4, 5],), ('111010', '000101')),
        (('111111', '000000'), ([1, 3, 5],), ('100110', '011001')),
        (('111111', '000000'), ([1, 2, 3, 4],), ('101011', '010100')),
        (('111111', '000000'), ([2, 3, 4, 5],), ('110101', '001010')),
        (('111111', '000000'), ([1, 2, 3, 4, 5],), ('101010', '010101')),
        (('11111111', '000000'), ([1, 2, 3],), ('10100011', '010111')),
        (('111111', '00000000'), ([1, 2, 3],), ('101000', '01011100')),
    ])
    def test_generate_progeny(self, genes, random_pivots, expected):
        seq = random_sequence(random_pivots)
        with patch('random.sample', side_effect=lambda *args: next(seq)):
            recombination = NPivot(num_pivots=len(random_pivots))
            child1, child2 = recombination(
                ListGenotype(genes=genes[0]),
                ListGenotype(genes=genes[1]),
            )
            assert child1 == ListGenotype(genes=expected[0])
            assert child2 == ListGenotype(genes=expected[1])

    @pytest.mark.parametrize('pivot_points', [5, 6, 7])
    @pytest.mark.parametrize('genes, expected', [
        (('111111', '000000'), ('101010', '010101')),
    ])
    def test_too_many_pivot_points(self, pivot_points, genes, expected):
        recombination = NPivot(num_pivots=pivot_points)
        child1, child2 = recombination(
            ListGenotype(genes=genes[0]),
            ListGenotype(genes=genes[1]),
        )
        assert child1 == ListGenotype(genes=expected[0])
        assert child2 == ListGenotype(genes=expected[1])


class TestRandomMask:
    @pytest.mark.parametrize('genes, rand, expected', [
        (('1111', '0000'), (.4, .6, .4, .6), ('1010', '0101')),
    ])
    def test_generate_progeny(self, genes, rand, expected):
        rand = random_sequence(rand)
        with patch('random.random', side_effect=lambda: next(rand)):
            parent1 = ListGenotype(genes=genes[0])
            parent2 = ListGenotype(genes=genes[1])
            child1, child2 = random_mask(parent1, parent2)

            assert len(child1) == len(parent1)
            assert len(child2) == len(parent2)
            assert child1 == ListGenotype(genes=expected[0])
            assert child2 == ListGenotype(genes=expected[1])


class TestPmx:
    @pytest.mark.parametrize('genes, random_pivots, expected', [
        (('ACTG', 'GTAC'), ([0, 4],), ('GTAC', 'ACTG')),
        (('ACTG', 'GTAC'), ([4, 0],), ('GTAC', 'ACTG')),
        (('ACTG', 'GTAC'), ([1, 3],), ('CTAG', 'GCTA')),
        (('ACTG', 'GTAC'), ([3, 1],), ('CTAG', 'GCTA')),
        (('34827165', '42516837'), ([3, 6],), ('34216875', '48527136')),
    ])
    def test_generate_progeny(self, genes, random_pivots, expected):
        seq = random_sequence(random_pivots)
        with patch('random.sample', side_effect=lambda *args: next(seq)):
            parent1 = ListGenotype(genes=genes[0])
            parent2 = ListGenotype(genes=genes[1])
            child1, child2 = pmx(parent1, parent2)

            assert len(parent1) == len(child1)
            assert len(parent2) == len(child2)
            assert child1 == ListGenotype(genes=expected[0])
            assert child2 == ListGenotype(genes=expected[1])
