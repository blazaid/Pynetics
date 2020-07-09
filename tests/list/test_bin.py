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
"""Tests for the binary list implementations."""
from unittest.mock import patch

import pytest

from pynetics.list import ListGenotype
from pynetics.list.bin import generalised_crossover
from tests.util import random_sequence


class TestGeneralizedCrossover:
    @pytest.mark.parametrize('genes, rand_int, expected', [
        (('1111', '0000'), (7,), ('0111', '1000')),
        (('0000', '1111'), (7,), ('0111', '1000')),
        (('0000', '0000'), (0,), ('0000', '0000')),
        (('1111', '1111'), (15,), ('1111', '1111')),
        (('1111', '0000'), (0,), ('0000', '1111')),
        (('0111', '1000'), (3,), ('0011', '1100')),
    ])
    def test_generate_progeny(self, genes, rand_int, expected):
        rand_int = random_sequence(rand_int)
        with patch('random.randint', side_effect=lambda *args: next(rand_int)):
            child1, child2 = generalised_crossover(
                ListGenotype(genes=(int(g) for g in genes[0])),
                ListGenotype(genes=(int(g) for g in genes[1])),
            )

            assert child1 == ListGenotype(genes=(int(g) for g in expected[0]))
            assert child2 == ListGenotype(genes=(int(g) for g in expected[1]))
