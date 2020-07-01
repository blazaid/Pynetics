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
from unittest.mock import Mock

import pytest

from pynetics.stop import FitnessBound, never, NumSteps


class TestFitnessBound:
    """Test for the `FitnessBound` stop condition."""

    @staticmethod
    def mock_ga_with_best_genotype(fitness):
        """Creates a mock genetic algorithm with a mock best genotype.

         :param fitness: The fitness to assign to the best genotype.
         """
        individual = Mock()
        individual.fitness = Mock(return_value=fitness)
        genetic_algorithm = Mock()
        genetic_algorithm.best = Mock(return_value=individual)

        return genetic_algorithm

    @pytest.mark.parametrize('fitness', [0.0, 0.25, 0.5, 0.75, 0.9, 0.9999999])
    def test_criteria_not_met_when_lower_fitness(self, fitness):
        """Not met when fitness is lower than expected."""
        genetic_algorithm = self.mock_ga_with_best_genotype(fitness)

        stop_condition = FitnessBound(bound=1.0)
        assert not stop_condition(genetic_algorithm)

    @pytest.mark.parametrize('fitness', [1.0, 1.000000001, 1.25, 1.5, 1.75, 2])
    def test_criteria_met_when_higher_fitness(self, fitness):
        """Met when fitness is greater or equal than expected."""
        genetic_algorithm = self.mock_ga_with_best_genotype(fitness)

        stop_condition = FitnessBound(bound=1.0)
        assert stop_condition(genetic_algorithm)


class TestNever:
    """Test for the `never` stop condition."""

    def test_criteria_is_always_false(self):
        """Condition is never met."""
        assert not never(Mock())


class TestNumSteps:
    """Test for the `NumSteps` stop condition."""

    @pytest.mark.parametrize('steps', [10])
    def test_criteria_not_met_when_lower_steps(self, steps):
        """Not met when generations are lower than expected."""
        stop_condition = NumSteps(steps=steps)

        genetic_algorithm = Mock()
        for generation in range(steps):
            genetic_algorithm.generation = generation
            assert not stop_condition(genetic_algorithm)

    @pytest.mark.parametrize('steps', [10])
    def test_criteria_met_when_higher_fitness(self, steps):
        """Met when generations are greater or equal than expected."""
        stop_condition = NumSteps(steps=steps)

        genetic_algorithm = Mock()
        for generation in range(steps + 1, steps + 10):
            genetic_algorithm.generation = generation
            assert stop_condition(genetic_algorithm)
