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
"""Tests for the alphabets implementations."""
from unittest.mock import Mock

import pytest

from pynetics.algorithm import GeneticAlgorithm
from pynetics.exception import NotInitialized
from tests.test_api import EvolutiveAlgorithmTests
from tests.util import build_population


class TestGeneticAlgorithm(EvolutiveAlgorithmTests):
    """Tests for GeneticAlgorithm class."""

    def get_instance(self, **kwargs):
        population_size = kwargs.get('population_size', 10)
        ga = GeneticAlgorithm(
            population_size=population_size,
            initializer=kwargs.get('initializer', Mock()),
            stop_condition=kwargs.get('stop_condition', Mock()),
            fitness=kwargs.get('fitness', Mock()),
            selection=kwargs.get('selection', Mock()),
            recombination=kwargs.get('recombination', Mock()),
            recombination_probability=kwargs.get(
                'recombination_probability', 1.0
            ),
            mutation=kwargs.get('mutation', Mock()),
            mutation_probability=kwargs.get('mutation_probability', 1.0),
            replacement=kwargs.get('replacement', (Mock(), 1.0)),
            callbacks=kwargs.get('callbacks', []),
        )
        if kwargs.get('mock_population', False):
            ga.population = build_population(size=population_size)

        return ga

    def test_cannot_get_best_if_not_initialized(self):
        with pytest.raises(NotInitialized):
            self.get_instance().best()

    @pytest.mark.parametrize('prob, exp', [
        (0, 0), (0.5, 0.5), (1, 1), (-1, 0), (2, 1), ('0.5', 0.5), (None, 0),
    ])
    def test_recombination_probability_always_between_0_and_1(self, prob, exp):
        ga = self.get_instance(recombination_probability=prob)

        assert ga.recombination_probability == exp

    @pytest.mark.parametrize('prob', [p / 10 for p in range(11)])
    def test_recombination_is_optional(self, prob):
        ga = self.get_instance(
            recombination_probability=prob, recombination=None
        )

        parents = Mock(), Mock()
        for child in ga.recombination(*parents):
            assert child is parents[0] or child is parents[1]

    @pytest.mark.parametrize('prob, exp', [
        (0, 0), (0.5, 0.5), (1, 1), (-1, 0), (2, 1), ('0.5', 0.5), (None, 0),
    ])
    def test_mutation_probability_always_between_0_and_1(self, prob, exp):
        ga = self.get_instance(mutation_probability=prob)

        assert ga.mutation_probability == exp

    @pytest.mark.parametrize('prob', [p / 10 for p in range(11)])
    def test_mutation_is_optional(self, prob):
        ga = self.get_instance(mutation_probability=prob, mutation=None)

        genotype = Mock()
        assert ga.mutation(prob, genotype) is genotype

    def test_all_elements_are_called(self):
        def mock_replacement(*, population, offspring):
            return offspring

        population_size = 10
        selection = Mock(side_effect=lambda p, _: p[:2])
        recombination = Mock(side_effect=lambda *gs: gs)
        mutation = Mock(side_effect=lambda p, g: g)
        replacement = Mock(side_effect=mock_replacement)

        ga = self.get_instance(
            population_size=population_size,
            selection=selection,
            recombination=recombination,
            recombination_probability=1.0,
            mutation=mutation,
            mutation_probability=1.0,
            replacement=(replacement, 1.0),
            mock_population=True
        )
        ga.population = build_population(size=population_size)

        ga.step()

        assert selection.call_count == 5
        assert recombination.call_count == 5
        assert mutation.call_count == 10
        assert replacement.call_count == 1
        assert ga.best() == ga.population[-1]
