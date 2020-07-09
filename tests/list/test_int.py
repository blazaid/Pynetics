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
"""Tests for the integer list genotype representation operators."""
from unittest.mock import patch

import pytest

from pynetics.exception import BoundsCannotBeTheSame
from pynetics.list import ListGenotype
from pynetics.list.int import IntegerIntervalInitializer, RangeCrossover, Creep
from tests.list.test_initializer import IntervalInitializerTests
from tests.util import random_sequence


class TestIntegerIntervalInitializer(IntervalInitializerTests):
    def get_instance(self, **kwargs):
        size = kwargs.get('size', 10)
        lower = kwargs.get('lower', 0)
        upper = kwargs.get('upper', 1)
        cls = kwargs.get('cls', None)
        return IntegerIntervalInitializer(
            size=size, lower=lower, upper=upper, cls=cls
        )

    @pytest.mark.parametrize('lower, upper, size', [
        (0, 1, 100),
        (-1, 1, 100),
        (-100, 100, 100),
    ])
    def test_check_returned_types_are_integer_values(self, lower, upper, size):
        initializer = self.get_instance(size=size, lower=lower, upper=upper)
        genotype = initializer.create()

        for gene in genotype:
            assert isinstance(gene, int)


class TestCreep:
    """Tests for the creep mutation schema."""

    def test_amount_of_zero_raises_error(self):
        with pytest.raises(ValueError):
            Creep(amount=0)

    @pytest.mark.parametrize('amount, lower, upper', [(1, 0, 0), (1, 1, 1)])
    def test_bounds_cannot_be_the_same(self, amount, lower, upper):
        with pytest.raises(BoundsCannotBeTheSame):
            Creep(amount=amount, lower=lower, upper=upper)

    @pytest.mark.parametrize('amount, lower, upper, exp_lower, exp_upper', [
        (1, None, None, None, None),
        (1, None, 10, None, 10), (1, -10, None, -10, None),
        (1, -10, 10, -10, 10), (1, 10, -10, -10, 10),
        (1, -10, -5, -10, -5), (1, -5, -10, -10, -5),
        (1, 5, 10, 5, 10), (1, 10, 5, 5, 10),
    ])
    def test_boundaries_are_managed_correctly(
            self, amount, lower, upper, exp_lower, exp_upper
    ):
        mutation = Creep(amount=amount, lower=lower, upper=upper)
        assert mutation.lower == exp_lower
        assert mutation.upper == exp_upper

    @pytest.mark.parametrize('amount, exp_amount', [(1, 1), (-1, 1)])
    def test_amount_is_stored_as_a_positive_number(self, amount, exp_amount):
        mutation = Creep(amount=amount)
        assert mutation.amount == exp_amount

    @pytest.mark.parametrize(
        'amount, fixed, lo, hi, chances, ints, p, genes, expected', [
            (1, True, 0, 10, [0, 0], [1], 1, [5], [6]),
            (1, True, 0, 10, [0, 1], [1], 1, [5], [4]),
            (1, True, 0, 10, [0, 0], [2], 1, [5], [6]),
            (1, True, 0, 10, [0, 1], [2], 1, [5], [4]),
            (2, True, 0, 10, [0, 0], [1], 1, [5], [7]),
            (2, True, 0, 10, [0, 1], [1], 1, [5], [3]),
            (10, True, 0, 10, [0, 0], [1], 1, [5], [10]),
            (10, True, 0, 10, [0, 1], [1], 1, [5], [0]),
            (1, False, 0, 10, [0, 0], [1], 1, [5], [6]),
            (1, False, 0, 10, [0, 1], [1], 1, [5], [4]),
            (1, False, 0, 10, [0, 0], [2], 1, [5], [7]),
            (1, False, 0, 10, [0, 1], [2], 1, [5], [3]),
            (1, False, 0, 10, [0, 0], [10], 1, [5], [10]),
            (1, False, 0, 10, [0, 1], [10], 1, [5], [0]),
            (10, True, None, 10, [0, 0], [1], 1, [5], [10]),
            (10, True, 10, None, [0, 0], [1], 1, [5], [15]),
            (10, True, None, 10, [0, 1], [1], 1, [5], [-5]),
            (10, True, 0, None, [0, 1], [1], 1, [5], [0]),
            (10, False, None, 10, [0, 0], [10], 1, [5], [10]),
            (10, False, 10, None, [0, 0], [10], 1, [5], [15]),
            (10, False, None, 10, [0, 1], [10], 1, [5], [-5]),
            (10, False, 0, None, [0, 1], [10], 1, [5], [0]),
        ]
    )
    def test_mutate_genotypes(
            self, amount, fixed, lo, hi, chances, ints, p, genes, expected
    ):
        mutation = Creep(amount=amount, fixed=fixed, lower=lo, upper=hi)

        take_chances = random_sequence(chances)
        randint = random_sequence(ints)

        with patch('random.random', side_effect=lambda: next(take_chances)):
            with patch('random.randint', side_effect=lambda *_: next(randint)):
                mutated = mutation(p, ListGenotype(genes=genes))

        assert mutated == ListGenotype(genes=expected)


class TestRangeCrossover:
    @pytest.mark.parametrize('lower, upper', [(0, 1), (0, 10), (1, 10)])
    def test_correct_lower_and_upper_bounds(self, lower, upper):
        crossover = RangeCrossover(lower=lower, upper=upper)
        assert crossover.lower == lower
        assert crossover.upper == upper

    @pytest.mark.parametrize('lower, upper', [(1, 0), (10, 0), (10, 1)])
    def test_switched_lower_and_upper_bounds(self, lower, upper):
        crossover = RangeCrossover(lower=lower, upper=upper)
        assert crossover.lower == upper
        assert crossover.upper == lower

    @pytest.mark.parametrize('lower, upper', [(0, 0), (1, 1), (10, 10)])
    def test_bounds_cannot_be_the_same(self, lower, upper):
        with pytest.raises(BoundsCannotBeTheSame):
            RangeCrossover(lower=lower, upper=upper)

    @pytest.mark.parametrize('lower, upper, genes, rand_int, expected', [
        (0, 9, ('0000', '0000'), (7,), ('7777', '0000')),
        (0, 9, ('0000', '9999'), (7,), ('7777', '2222')),
        (0, 9, ('9999', '0000'), (0,), ('0000', '9999')),
        (0, 9, ('9999', '9999'), (15,), ('9999', '3333')),
        (0, 9, ('4534', '6543'), (4, 9, 4, 4,), ('4944', '6133')),
        (0, 9, ('4534', '6543'), (5, 9, 4, 4,), ('5944', '5133')),
        (0, 9, ('4534', '6543'), (6, 9, 4, 4,), ('6944', '4133')),
        (0, 9, ('6543', '4534'), (4, 9, 4, 4,), ('4944', '6133')),
        (0, 9, ('6543', '4534'), (5, 9, 4, 4,), ('5944', '5133')),
        (0, 9, ('6543', '4534'), (6, 9, 4, 4,), ('6944', '4133')),
    ])
    def test_generate_progeny(self, lower, upper, genes, rand_int, expected):
        rand_int = random_sequence(rand_int)
        with patch('random.randint', side_effect=lambda *args: next(rand_int)):
            child1, child2 = RangeCrossover(lower=lower, upper=upper)(
                ListGenotype(genes=(int(g) for g in genes[0])),
                ListGenotype(genes=(int(g) for g in genes[1])),
            )

            assert child1 == ListGenotype(genes=(int(g) for g in expected[0]))
            assert child2 == ListGenotype(genes=(int(g) for g in expected[1]))
