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
        def mock_replacement(*, population, offspring, max_size):
            return offspring

        population_size = kwargs.get('population_size', 10)
        ga = GeneticAlgorithm(
            population_size=population_size,
            initializer=kwargs.get('initializer', Mock()),
            stop_condition=kwargs.get('stop_condition', Mock()),
            fitness=kwargs.get('fitness', Mock()),
            selection=kwargs.get(
                'selection', Mock(side_effect=lambda p, _: p[:2])
            ),
            recombination=kwargs.get(
                'recombination', Mock(side_effect=lambda *gs: gs)
            ),
            recombination_probability=kwargs.get(
                'recombination_probability', 1.0
            ),
            mutation=kwargs.get(
                'mutation', Mock(side_effect=lambda p, g: g)
            ),
            mutation_probability=kwargs.get('mutation_probability', 1.0),
            replacement=kwargs.get(
                'replacement', Mock(side_effect=mock_replacement)
            ),
            replacement_ratio=kwargs.get('replacement_ratio', 1.0),
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

        parent = Mock()
        assert parent == ga.recombination(parent)

    def test_selection_size_is_correctly_computed(self):
        ga = self.get_instance(recombination=lambda g1: g1)
        assert ga.selection_size == 1
        ga.recombination = lambda g1, g2: (g1, g2)
        assert ga.selection_size == 2
        ga.recombination = lambda g1, g2, g3, g4: (g1, g2, g3, g4)
        assert ga.selection_size == 4

    def test_selection_size_cannot_be_altered(self):
        ga = self.get_instance(recombination=lambda g1, g2: (g1, g2))
        assert ga.selection_size == 2
        with pytest.raises(AttributeError):
            ga.selection_size = 4

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

    def test_no_recombination_probability(self):
        ga = self.get_instance(
            population_size=2,
            recombination_probability=0.0,
            mock_population=True,
        )
        population = ga.population
        ga.step()
        offspring = ga.population

        for genotype in population:
            assert genotype in offspring
        for genotype in offspring:
            assert genotype in population

    @pytest.mark.parametrize('size, ratio, expected', [
        (10, 0, 1), (10, -1, 1),
        (10, 1.1, 10), (10, 2, 10),
        (10, 0.1, 1), (10, 0.25, 2), (10, 0.5, 5), (10, 0.75, 8), (10, 1, 10)
    ])
    def test_offspring_size_is_computed_correctly(self, size, ratio, expected):
        ga = self.get_instance(population_size=size, replacement_ratio=ratio)
        assert ga.offspring_size == expected

    @pytest.mark.parametrize('ratio, size1, exp1, size2, exp2', [
        (-1, 10, 1, 20, 1), (0.1, 10, 1, 20, 2), (2, 10, 10, 15, 15),
    ])
    def test_offspring_size_is_updated_correctly_if_size_varies(
            self, ratio, size1, exp1, size2, exp2
    ):
        ga = self.get_instance(population_size=size1, replacement_ratio=ratio)
        assert ga.offspring_size == exp1
        ga.population_size = size2
        assert ga.offspring_size == exp2

    @pytest.mark.parametrize('size, ratio1, exp1, ratio2, exp2', [
        (10, 0.1, 1, 0.5, 5), (10, 0.2, 2, -1, 1), (10, 0.5, 5, 2, 10),
    ])
    def test_offspring_size_is_updated_correctly_if_ratio_varies(
            self, size, ratio1, exp1, ratio2, exp2
    ):
        ga = self.get_instance(population_size=size, replacement_ratio=ratio1)
        assert ga.offspring_size == exp1
        ga.replacement_ratio = ratio2
        assert ga.offspring_size == exp2

    def test_all_elements_are_called(self):
        ga = self.get_instance(population_size=10, mock_population=True)

        ga.step()

        assert ga.selection.call_count == 5
        assert ga.recombination.call_count == 5
        assert ga.mutation.call_count == 10
        assert ga.replacement.call_count == 1
        assert ga.best() == ga.population[-1]

    def test_change_fitness_function_affects_population(self):
        ga = self.get_instance(population_size=10, mock_population=True)

        old_fitness = ga.fitness
        assert ga.population.fitness == old_fitness
        for genotype in ga.population:
            assert genotype.fitness_function == old_fitness

        new_fitness = Mock()
        ga.fitness = new_fitness
        assert ga.population.fitness != old_fitness
        assert ga.population.fitness == new_fitness
        for genotype in ga.population:
            assert genotype.fitness_function != old_fitness
            assert genotype.fitness_function == new_fitness
