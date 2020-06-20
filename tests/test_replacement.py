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
"""TODO TDB..."""
import abc
from unittest.mock import Mock

import pytest

from pynetics.exception import (
    OffspringSizeBiggerThanPopulationSize,
    PopulationSizesDoNotMatchAfterReplacement,
)
from pynetics.replacement import LowElitism, HighElitism
from tests.test_api import GenericTest
from tests.util import build_population


# ~~~~~~~~~~~~~
# Generic tests
# ~~~~~~~~~~~~~
class ReplacementSchemaTests(GenericTest, metaclass=abc.ABCMeta):
    """General tests for the replacement schemas common behavior."""

    @pytest.mark.parametrize('size', [2, 4, 8, 16])
    def test_offspring_size_smaller_than_population_size(self, size):
        """Offspring size can be smaller than population size."""
        replacement = self.get_instance()

        population = build_population(size=size)
        for size in range(1, size):
            offspring = build_population(size=size)
            replacement(population=population, offspring=offspring)

    @pytest.mark.parametrize('size', [2, 4, 8, 16])
    def test_offspring_size_equal_to_population_size(self, size):
        """Offspring size can be smaller than population size."""
        replacement = self.get_instance()

        population = build_population(size=size)
        offspring = build_population(size=size)
        replacement(population=population, offspring=offspring)

    @pytest.mark.parametrize('size', [2, 4, 8, 16])
    def test_offspring_size_bigger_than_population_size(self, size):
        """Offspring size cannot be bigger than population size."""
        replacement = self.get_instance()

        population = build_population(size=size)
        for size in range(size + 1, size + 10):
            offspring = build_population(size=size)
            with pytest.raises(OffspringSizeBiggerThanPopulationSize):
                replacement(population=population, offspring=offspring)

    @pytest.mark.parametrize('population_size', [2, 4, 8, 16])
    def test_population_size_cannot_vary(self, population_size):
        """Everything works fine if replacement is set."""
        replacement = self.get_instance(maintain=True)

        for offspring_size in range(1, population_size + 1):
            population = build_population(size=population_size)
            offspring = build_population(size=offspring_size)

            replacement(population=population, offspring=offspring)

    @pytest.mark.parametrize('old_size', [2, 4, ])
    @pytest.mark.parametrize('new_size', [3, 5, ])
    def test_error_maintain_population_size(self, old_size, new_size):
        """Error if old and new population sizes do not match."""
        population = build_population(size=old_size)
        offspring = build_population(size=old_size)
        new_population = build_population(size=new_size)

        replacement = self.get_instance(maintain=True)
        replacement.do = Mock(side_effect=lambda **kwargs: new_population)

        with pytest.raises(PopulationSizesDoNotMatchAfterReplacement):
            replacement(population=population, offspring=offspring)


# ~~~~~~~~~~~~~~
# Specific tests
# ~~~~~~~~~~~~~~
class TestHighElitism(ReplacementSchemaTests):
    """ Tests for high elitism replacement method. """

    def get_instance(self, maintain=True):
        return HighElitism()

    @pytest.mark.parametrize('pop_size, off_size, exp_ids', [
        (2, 1, [1, 2]),
        (2, 2, [1, 3]),
        (5, 3, [2, 3, 4, 6, 7]),
        (5, 5, [3, 4, 7, 8, 9]),
    ])
    def test_correct_replacement(self, pop_size, off_size, exp_ids):
        replacement = self.get_instance(maintain=True)
        new_population = replacement(
            population=build_population(size=pop_size),
            offspring=build_population(size=off_size, first_id=pop_size)
        )

        real_ids = [g.id for g in new_population]

        assert sorted(real_ids) == sorted(exp_ids)


class TestLowElitism(ReplacementSchemaTests):
    """ Tests for low elitism replacement method. """

    def get_instance(self, maintain=True):
        return LowElitism()

    @pytest.mark.parametrize('pop_size, off_size, exp_ids', [
        (2, 1, [1, 2]),
        (2, 2, [2, 3]),
        (5, 3, [3, 4, 5, 6, 7]),
        (5, 5, [5, 6, 7, 8, 9]),
    ])
    def test_correct_replacement(self, pop_size, off_size, exp_ids):
        replacement = self.get_instance(maintain=True)
        new_population = replacement(
            population=build_population(size=pop_size),
            offspring=build_population(size=off_size, first_id=pop_size)
        )

        real_ids = [g.id for g in new_population]

        assert sorted(real_ids) == sorted(exp_ids)
