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
"""Generic and specific tests for all the pynetics api."""
import abc
import itertools
from unittest.mock import Mock

import pytest

from pynetics.api import Population
from pynetics.callback import Callback
from pynetics.exception import FullPopulationError
from pynetics.stop import NumSteps
from tests.util import build_population


class GenericTest(metaclass=abc.ABCMeta):
    """Generic class for all of those tests that require subclassing."""

    @abc.abstractmethod
    def get_instance(self, **kwargs):
        """Returns a Diversity instance for this particular test.

        All the arguments on the implementation must be optional.

        :param kwargs: Specific arguments to customize the instance.
        :return: A diversity schema.
        """


class DiversityTests(GenericTest, metaclass=abc.ABCMeta):
    """Base tests for all the genotype implementations."""


class EvolutiveAlgorithmTests(GenericTest, metaclass=abc.ABCMeta):
    """General tests for the implemented evolutive algorithms."""

    @pytest.mark.parametrize('callbacks, expected', [
        (None, []),
        ([None], []),
        ([None, Mock(id=1), Mock(id=2)], [Mock(id=1), Mock(id=2)]),
        ([Mock(id=1), None, Mock(id=2)], [Mock(id=1), Mock(id=2)]),
        ([Mock(id=1), Mock(id=2), None], [Mock(id=1), Mock(id=2)]),
        ([Mock(id=1), Mock(id=2)], [Mock(id=1), Mock(id=2)]),
    ])
    def test_empty_callbacks_are_correctly_stored(self, callbacks, expected):
        algorithm = self.get_instance(callbacks=callbacks)

        for callback in algorithm.callbacks:
            assert callback is not None
            assert callback.id in [c.id for c in expected]

    def test_initialize_and_finalize(self):
        algorithm = self.get_instance()
        algorithm.initialize()
        assert algorithm.generation == 0
        algorithm.finalize()
        assert algorithm.generation == 0

    def test_generation_increases(self):
        algorithm = self.get_instance(stop_condition=NumSteps(1))
        algorithm.step = Mock(return_value=None)
        algorithm.best = Mock(return_value=Mock())
        assert algorithm.generation == 0

        algorithm.run()

        algorithm.step.assert_called_once()
        assert algorithm.generation == 1

    def test_callback_called(self):
        callback = Callback()
        callback.on_algorithm_begin = Mock(side_effect=lambda *args: None)
        callback.on_step_begin = Mock(side_effect=lambda *args: None)
        callback.on_step_end = Mock(side_effect=lambda *args: None)
        callback.on_algorithm_end = Mock(side_effect=lambda *args: None)

        algorithm = self.get_instance(
            stop_condition=NumSteps(2),
            callbacks=[callback]
        )
        algorithm.step = Mock(return_value=None)
        algorithm.best = Mock(return_value=Mock())

        algorithm.run()

        assert algorithm.generation == 2
        callback.on_algorithm_begin.call_count = 1
        callback.on_step_begin.call_count = 2
        callback.on_step_end.call_count = 2
        callback.on_algorithm_end.call_count = 1

    def test_stop_requested(self):
        algorithm = self.get_instance(stop_condition=NumSteps(2))
        algorithm.best = Mock(return_value=Mock())

        assert algorithm.running()

        algorithm.stop()

        assert not algorithm.running()


class GenotypeTests(GenericTest, metaclass=abc.ABCMeta):
    """Base tests for all the genotype implementations."""

    @abc.abstractmethod
    def get_instance(self, **kwargs):
        """Returns a genotype instance for this particular test.

        All the arguments on the implementation must be optional.

        :param kwargs: Named parameters to be used as arguments by the
            instance.
        :return: A genotype.
        """

    def test_no_fitness_function_after_creation(self):
        """Base creation implies a genotype without fitness function."""
        genotype = self.get_instance()
        with pytest.raises(TypeError):
            genotype.fitness()

    @abc.abstractmethod
    def test_check_equality_between_two_genotypes(self):
        """Two instances with the same content are equal."""

    @abc.abstractmethod
    def test_check_inequality_between_two_genotypes(self):
        """Two instances with different content aren't equal."""


class InitializerTests(GenericTest, metaclass=abc.ABCMeta):
    """Test for all the initializer instances"""

    @abc.abstractmethod
    def get_instance(self, **kwargs):
        """Returns an initializer instance for this particular test.

        All the arguments on the implementation must be optional.

        :param kwargs: Specific arguments to customize the instance.
        :return: An initializer.
        """

    @pytest.mark.parametrize('population_size', list(range(1, 11)))
    def test_population_is_filled_correctly(self, population_size):
        initializer = self.get_instance()
        for i in range(population_size + 1):
            population = build_population(population_size, fill=i)
            population = initializer.fill(population)

            assert population.full()
            assert len(population) == population_size


class TestPopulation:
    """Tests to perform over Population instances."""

    @pytest.mark.parametrize('old_size', [2, 4, 6, 8])
    @pytest.mark.parametrize('new_size', [2, 4, 6, 8])
    def test_population_size_update(self, old_size, new_size):
        population = Population(old_size, Mock())
        assert population.max_size == old_size

        population.max_size = new_size
        assert population.max_size == new_size

    @pytest.mark.parametrize('size', [2, 4, 8])
    def test_empty_and_full_population(self, size):
        population = Population(size, Mock())
        assert population.empty()
        assert len(population) == 0

        while not population.full():
            population.append(Mock())
        assert population.full()
        assert len(population) == population.max_size == size

        while not population.empty():
            population.pop()
        assert population.empty()
        assert len(population) == 0

    @pytest.mark.parametrize('p1_size', [2, 4, 8])
    @pytest.mark.parametrize('p1_halved', [True, False])
    @pytest.mark.parametrize('p2_size', [2, 4, 8])
    @pytest.mark.parametrize('p2_halved', [True, False])
    def test_add_two_populations(self, p1_size, p1_halved, p2_size, p2_halved):
        p1 = build_population(p1_size if not p1_halved else (p1_size // 2))
        p1.max_size = p1_size
        p2 = build_population(p2_size if not p2_halved else (p2_size // 2))
        p2.max_size = p2_size
        population = p1 + p2

        assert population.max_size == p1_size + p2_size
        for genotype in itertools.chain(p1, p2):
            assert genotype in population

    @pytest.mark.parametrize('size', [10])
    @pytest.mark.parametrize('key', [None, [], {}, (), ])
    def test_set_item_with_wrong_key_type(self, size, key):
        population = build_population(size=size)
        with pytest.raises(TypeError):
            population[key] = Mock()

    @pytest.mark.parametrize('size', [10])
    @pytest.mark.parametrize('key', [10, 11, 20])
    def test_set_item_in_wrong_position(self, size, key):
        population = build_population(size=size)
        with pytest.raises(IndexError):
            population[key] = Mock()

    @pytest.mark.parametrize('size', [5])
    @pytest.mark.parametrize('key', [0, 1, 2, 3, 4])
    def test_set_item_works_as_expected(self, size, key):
        population = build_population(size=size)
        assert not population.is_sorted

        genotype = Mock()
        genotype.id = 42
        population[key] = genotype

        assert genotype.fitness_function == population.fitness
        assert not population.is_sorted

    @pytest.mark.parametrize('size', [5])
    @pytest.mark.parametrize('index', [0, 1, 2, 3, 4, 5])
    def test_insert_item_in_a_full_population(self, size, index):
        population = build_population(size=size)
        with pytest.raises(FullPopulationError):
            population.insert(index, Mock())

    @pytest.mark.parametrize('size', [5])
    @pytest.mark.parametrize('index', [0, 1, 2, 3, 4, 5])
    def test_insert_item_works_as_expected(self, size, index):
        population = build_population(size=size)
        population.max_size += 1

        assert not population.is_sorted

        genotype = Mock()
        genotype.id = 42
        population.insert(index, genotype)

        assert genotype.fitness_function == population.fitness
        assert not population.is_sorted

    def test_sort_population(self):
        population = build_population(size=2)
        assert not population.is_sorted

        g1, g2 = Mock(id=1), Mock(id=2)
        g1.fitness = Mock(return_value=1)
        g2.fitness = Mock(return_value=2)
        population[0] = g2
        population[1] = g1
        assert not population.is_sorted

        population.sort()
        assert population.is_sorted
        assert population[0] == g1
        assert population[1] == g2

    def test_reverse_population(self):
        population = build_population(size=2)
        population.sort()
        assert population.is_sorted

        g1, g2 = population[0], population[1]
        population.reverse()
        assert not population.is_sorted
        assert population[0] == g2
        assert population[1] == g1

    def test_cannot_append_genotype_in_full_population(self):
        population = build_population(10)
        with pytest.raises(FullPopulationError):
            population.append(Mock())

    def test_cannot_extend_a_full_population(self):
        population = build_population(10)
        with pytest.raises(FullPopulationError):
            population.extend([Mock()])
