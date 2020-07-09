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
"""Tests for the real list implementations."""
from unittest.mock import patch

import pytest

from pynetics.exception import BoundsCannotBeTheSame
from pynetics.list import ListGenotype
from pynetics.list.real import (
    RealIntervalInitializer,
    FlexibleRecombination, plain_recombination,
)
from tests.list.test_initializer import IntervalInitializerTests
from tests.util import random_sequence


# ~~~~~~~~~~~~~~~~~
# Initializer tests
# ~~~~~~~~~~~~~~~~~
class TestRealIntervalInitializer(IntervalInitializerTests):
    def get_instance(self, **kwargs):
        size = kwargs.get('size', 10)
        lower = kwargs.get('lower', 0)
        upper = kwargs.get('upper', 1)
        cls = kwargs.get('cls', None)
        return RealIntervalInitializer(
            size=size, lower=lower, upper=upper, cls=cls
        )

    @pytest.mark.parametrize('lower, upper, size', [
        (0, 1, 100),
        (-1, 1, 100),
        (-100, 100, 100),
    ])
    def test_check_returned_types_are_real_values(self, lower, upper, size):
        initializer = self.get_instance(size=size, lower=lower, upper=upper)
        genotype = initializer.create()

        for gene in genotype:
            assert isinstance(gene, float)


# ~~~~~~~~~~~~~~~~~~~
# Recombination tests
# ~~~~~~~~~~~~~~~~~~~
def assert_recombination(exp, genes, recombination):
    child1, child2 = recombination(
        ListGenotype(genes=(float(g) for g in genes[0])),
        ListGenotype(genes=(float(g) for g in genes[1])),
    )
    exp1 = ListGenotype(genes=(float(g) for g in exp[0]))
    exp2 = ListGenotype(genes=(float(g) for g in exp[1]))
    for real, expected in (child1, exp1), (child2, exp2):
        assert len(real) == len(exp)
        for g1, g2 in zip(real, expected):
            assert g1 == pytest.approx(g2, 0.01)


class TestFlexibleRecombination:
    @pytest.mark.parametrize('lower, upper', [(0, 1), (0, 10), (1, 10)])
    def test_correct_lower_and_upper_bounds(self, lower, upper):
        crossover = FlexibleRecombination(lower=lower, upper=upper, alpha=1)
        assert crossover.lower == lower
        assert crossover.upper == upper

    @pytest.mark.parametrize('lower, upper', [(1, 0), (10, 0), (10, 1)])
    def test_switched_lower_and_upper_bounds(self, lower, upper):
        crossover = FlexibleRecombination(lower=lower, upper=upper, alpha=1)
        assert crossover.lower == upper
        assert crossover.upper == lower

    @pytest.mark.parametrize('lower, upper', [(0, 0), (1, 1), (10, 10)])
    def test_bounds_cannot_be_the_same(self, lower, upper):
        with pytest.raises(BoundsCannotBeTheSame):
            FlexibleRecombination(lower=lower, upper=upper, alpha=1)

    @pytest.mark.parametrize('lower, upper, alpha, genes, rand, exp', [
        (0, 1, 0.1,
         ([0, 0], [1, 1]),
         (0.25,),
         ([0.25, 0.25], [0.75, 0.75])
         ),
        (0, 1, 0.1,
         ([0, 0], [1, 1]),
         (0.5,),
         ([0.50, 0.50], [0.50, 0.50])
         ),
        (0, 1, 0.1,
         ([0, 0], [1, 1]),
         (0.75,),
         ([0.75, 0.75], [0.25, 0.25])
         ),
        (0, 1, 0.1,
         ([1, 1], [0, 0]),
         (0.25,),
         ([0.25, 0.25], [0.75, 0.75])
         ),
        (0, 1, 0.1,
         ([1, 1], [0, 0]),
         (0.75,),
         ([0.75, 0.75], [0.25, 0.25])
         ),
    ])
    def test_generate_progeny(self, lower, upper, alpha, genes, rand, exp):
        rand = random_sequence(rand)
        with patch('random.uniform', side_effect=lambda *args: next(rand)):
            recombination = FlexibleRecombination(
                lower=lower,
                upper=upper,
                alpha=alpha,
            )
            assert_recombination(exp, genes, recombination)


class TestPlainRecombination:
    @pytest.mark.parametrize('genes, rand, exp', [
        (([0, 0], [1, 1]), (0.25,), ([0.25, 0.25], [0.75, 0.75])),
        (([0, 0], [1, 1]), (0.5,), ([0.50, 0.50], [0.50, 0.50])),
        (([0, 0], [1, 1]), (0.75,), ([0.75, 0.75], [0.25, 0.25])),
    ])
    def test_generate_progeny(self, genes, rand, exp):
        rand = random_sequence(rand)
        with patch('random.uniform', side_effect=lambda *args: next(rand)):
            assert_recombination(exp, genes, plain_recombination)
