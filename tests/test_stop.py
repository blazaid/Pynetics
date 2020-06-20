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
"""Tests for all the stop criteria."""
import abc
from unittest.mock import Mock

import pytest

from pynetics.stop import FitnessBound, NumSteps


# ~~~~~~~~~~~~~
# Generic tests
# ~~~~~~~~~~~~~
class StopConditionTests(metaclass=abc.ABCMeta):
    """Generic behavior for the implemented stop conditions."""

    @abc.abstractmethod
    def get_stop_condition(self):
        """Returns a valid stop condition for this particular test.

        :return: An stop condition instance.
        """

    def test_genetic_algorithm_cannot_be_none(self):
        stop_condition = self.get_stop_condition()


# ~~~~~~~~~~~~~~
# Specific tests
# ~~~~~~~~~~~~~~
class TestFitnessBound(StopConditionTests):
    def get_stop_condition(self):
        return FitnessBound(bound=1.0)

    def test_correct_construction(self):
        for bound in range(0, 10):
            FitnessBound(bound=bound)

    def test_criteria_not_met_when_lower_fitness(self):
        stop_condition = FitnessBound(bound=1.0)
        for fitness in (0.0, 0.25, 0.5, 0.75, 0.9, 0.9999999):
            individual = Mock()
            individual.fitness = Mock(return_value=fitness)
            genetic_algorithm = Mock()
            genetic_algorithm.best = Mock(return_value=individual)

            assert not stop_condition(genetic_algorithm)

    def test_criteria_met_when_higher_fitness(self):
        stop_condition = FitnessBound(bound=1.0)
        for fitness in (1.0, 1.000000001, 1.25, 1.5, 1.75, 2):
            individual = Mock()
            individual.fitness = Mock(return_value=fitness)
            genetic_algorithm = Mock()
            genetic_algorithm.best = Mock(return_value=individual)

            assert stop_condition(genetic_algorithm)


class TestNumSteps(StopConditionTests):
    def get_stop_condition(self):
        return NumSteps(steps=90)

    @pytest.mark.parametrize('steps', [10, 20, 30])
    @pytest.mark.parametrize('generation', [1, 2, 4, 8])
    def test_criteria_not_met_when_lower_steps(self, steps, generation):
        stop_condition = NumSteps(steps=steps)

        genetic_algorithm = Mock()
        genetic_algorithm.generation = generation

        assert not stop_condition(genetic_algorithm)

    @pytest.mark.parametrize('steps', [10, 20, 30])
    @pytest.mark.parametrize('generation', [32, 64, 128])
    def test_criteria_met_when_higher_fitness(self, steps, generation):
        stop_condition = NumSteps(steps=steps)

        genetic_algorithm = Mock()
        genetic_algorithm.generation = generation

        assert stop_condition(genetic_algorithm)
