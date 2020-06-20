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
"""Tests for all the selection schema."""
import abc
from unittest.mock import Mock

import pytest

from pynetics import api
from pynetics.exception import (
    EmptyPopulation,
    WrongSelectionSize,
    CannotSelectThatMany,
)
from pynetics.selection import (
    MonteCarlo,
    Truncation,
    Tournament,
    RouletteWheel,
    ExponentialRank,
    LinearRank,
)
from tests.util import build_population, FixRandomSeed


# ~~~~~~~~~~~~~
# Generic tests
# ~~~~~~~~~~~~~
class SelectionSchemaTests(metaclass=abc.ABCMeta):
    """Test for the selection schemas common behavior."""

    @abc.abstractmethod
    def get_selection_schema(self, replacement=False):
        """Returns a selection instance for this particular test.

        :param replacement: If it is (or isn't) a selection schema
            without replacement. Defaults to False (i.e. without
            replacement)
        :return: A selection schema instance.
        """

    def selection_testing(
            self, *,
            selection,
            population,
            selection_size,
            expected_selection,
    ):
        """Performs a selection over a population.

        This selection will have the random seed fixed, so it will be
        deterministic given the same arguments over and over.

        :param selection: The selection schema
        :param population: The population to select from.
        :param selection_size: The amount of genotypes to select.
        :param expected_selection: A list with the ids of the expected
            genotypes.
        """
        with FixRandomSeed():
            selected = selection(population=population, n=selection_size)

            current_selection = [genotype.id for genotype in selected]
            assert current_selection == expected_selection

    def test_replacement_attribute_is_correctly_set(self):
        """Checks the replacement attribute is set when initializing."""
        selection = self.get_selection_schema()
        assert not selection.replacement
        selection = self.get_selection_schema(replacement=True)
        assert selection.replacement

    def test_population_cannot_be_none(self):
        """Population cannot be none in order to select from it."""
        selection = self.get_selection_schema()
        with pytest.raises(TypeError):
            selection(population=None)

    def test_population_cannot_be_empty(self):
        """Population cannot be empty in order to select from it."""
        selection = self.get_selection_schema()
        population = api.Population(size=10, fitness=Mock())

        with pytest.raises(EmptyPopulation):
            selection(population=population, n=2)

    def test_sel_size_cannot_be_zero_or_negative(self):
        """We cannot select a negative amount of genotypes."""
        selection = self.get_selection_schema()
        population = build_population(size=42)

        for n in (0, -1, -2):
            with pytest.raises(WrongSelectionSize):
                selection(population=population, n=n)

    def test_sel_size_cannot_be_less_than_pop_size_without_replacement(self):
        """We cannot select more genotypes than existent ones."""
        selection = self.get_selection_schema()

        for size in (2, 3, 4, 5, 6):
            population = build_population(size)

            for n in (size + 1, size + 10):
                with pytest.raises(CannotSelectThatMany):
                    selection(population=population, n=n)

    def test_correct_selected_number_without_replacement(self):
        """Everything works fine if no replacement is set."""
        selection = self.get_selection_schema()

        for size in (2, 3, 4, 5, 6):
            population = build_population(size)

            for n in range(1, size + 1):
                selected = selection(population=population, n=n)
                assert len(selected) == n

    def test_correct_selected_number_with_replacement(self):
        """Everything works fine if replacement is set."""
        selection = self.get_selection_schema(replacement=True)

        for size in (2, 3, 4, 5, 6):
            population = build_population(size)

            for n in range(1, (5 * size) + 1, 2):
                selected = selection(population=population, n=n)
                assert len(selected) == n


class WeightBasedSchemaTests(SelectionSchemaTests, metaclass=abc.ABCMeta):
    """Frequency based selection schemas tests common behavior."""

    def weights_testing(self, populations, weights):
        """Performs a weights computation.

        It is used to test all the selection

        :param populations: A list of populations.
        :param weights: A list of tuples of expected weights for the
            populations specified by the 'populations' argument.
        :return:
        """
        selection = self.get_selection_schema()

        for population, expected_weights in zip(populations, weights):
            actual_weights = selection.get_weights(genotypes=population)

            for actual, expected in zip(actual_weights, expected_weights):
                assert actual == pytest.approx(expected, 10e-4)


# ~~~~~~~~~~~~~~
# Specific tests
# ~~~~~~~~~~~~~~
class TestMonteCarlo(SelectionSchemaTests):
    """Tests for the Monte Carlo selection schema."""

    def get_selection_schema(self, replacement=False):
        return MonteCarlo(replacement=replacement)

    @pytest.mark.parametrize('p_size, s_size, repl, expected', [
        (10, 10, True, [6, 0, 2, 2, 7, 6, 8, 0, 4, 0]),
        (10, 10, False, [1, 0, 4, 9, 6, 5, 8, 2, 3, 7]),
        (10, 5, True, [6, 0, 2, 2, 7]),
        (10, 5, False, [1, 0, 4, 9, 6]),
    ])
    def test_correct_selection(self, p_size, s_size, repl, expected):
        self.selection_testing(
            selection=MonteCarlo(replacement=repl),
            population=build_population(p_size),
            selection_size=s_size,
            expected_selection=expected,
        )


class TestTruncation(SelectionSchemaTests):
    """Tests for the truncation selection schema."""

    def get_selection_schema(self, replacement=False):
        return Truncation(replacement=replacement)

    @pytest.mark.parametrize('p_size, s_size, repl, expected', [
        (10, 10, True, [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]),
        (10, 10, False, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        (10, 5, True, [9, 9, 9, 9, 9]),
        (10, 5, False, [9, 8, 7, 6, 5]),
    ])
    def test_correct_selection(self, p_size, s_size, repl, expected):
        self.selection_testing(
            selection=Truncation(replacement=repl),
            population=build_population(p_size),
            selection_size=s_size,
            expected_selection=expected,
        )


class TestTournament(SelectionSchemaTests):
    def get_selection_schema(self, replacement=False):
        return Tournament(replacement=replacement, m=2)

    @pytest.mark.parametrize('p_size, s_size, m, repl, expected', [
        (10, 10, 2, True, [6, 2, 7, 8, 4, 5, 1, 6, 5, 8]),
        (10, 10, 2, False, [6, 2, 7, 9, 3, 4, 0, 5, 8, 1]),
        (10, 5, 2, True, [6, 2, 7, 8, 4]),
        (10, 5, 2, False, [6, 2, 7, 9, 3]),
        (10, 10, 4, True, [6, 8, 5, 6, 8, 8, 9, 8, 9, 8]),
        (10, 10, 4, False, [6, 9, 4, 5, 7, 8, 3, 2, 1, 0]),
        (10, 5, 4, True, [6, 8, 5, 6, 8]),
        (10, 5, 4, False, [6, 9, 4, 5, 7]),
    ])
    def test_correct_selection(self, p_size, s_size, m, repl, expected):
        self.selection_testing(
            selection=Tournament(replacement=repl, m=m),
            population=build_population(p_size),
            selection_size=s_size,
            expected_selection=expected,
        )


class TestRouletteWheel(WeightBasedSchemaTests):
    """Tests for the roulette wheel selection schema."""

    def get_selection_schema(self, replacement=False):
        return RouletteWheel(replacement=replacement)

    def test_correct_weights_for_selection(self):
        self.weights_testing(
            populations=[build_population(size) for size in (2, 4, 8)],
            weights=[(0, 1), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5, 6, 7)]
        )

    @pytest.mark.parametrize('p_size, s_size, repl, expected', [
        (10, 10, True, [8, 2, 5, 5, 8, 8, 9, 3, 6, 2]),
        (10, 10, False, [8, 1, 5, 4, 9, 7, 6, 2, 3, 0]),
        (10, 5, True, [8, 2, 5, 5, 8]),
        (10, 5, False, [8, 1, 5, 4, 9]),
    ])
    def test_correct_selection(self, p_size, s_size, repl, expected):
        self.selection_testing(
            selection=RouletteWheel(replacement=repl),
            population=build_population(p_size),
            selection_size=s_size,
            expected_selection=expected,
        )


class TestExponentialRank(WeightBasedSchemaTests):
    """Tests for the exponential rank selection schema."""

    def get_selection_schema(self, replacement=False):
        return ExponentialRank(alpha=4, replacement=replacement)

    def test_correct_weights_for_selection(self):
        self.weights_testing(
            populations=[build_population(size) for size in (2, 4, 8)],
            weights=[
                (1, 16),
                (1, 16, 81, 256),
                (1, 16, 81, 256, 625, 1296, 2401, 4096)
            ]
        )

    @pytest.mark.parametrize('p_size, s_size, alpha, repl, expected', [
        (10, 10, 0, True, [6, 0, 2, 2, 7, 6, 8, 0, 4, 0]),
        (10, 10, 0, False, [6, 0, 3, 2, 8, 7, 9, 1, 4, 5]),
        (10, 5, 0, True, [6, 0, 2, 2, 7]),
        (10, 5, 0, False, [6, 0, 3, 2, 8]),
        (10, 10, 4, True, [9, 4, 7, 7, 9, 9, 9, 5, 8, 4]),
        (10, 10, 4, False, [9, 4, 7, 6, 8, 5, 3, 1, 2, 0]),
        (10, 5, 4, True, [9, 4, 7, 7, 9]),
        (10, 5, 4, False, [9, 4, 7, 6, 8]),
    ])
    def test_correct_selection(self, p_size, s_size, alpha, repl, expected):
        self.selection_testing(
            selection=ExponentialRank(alpha=alpha, replacement=repl),
            population=build_population(p_size),
            selection_size=s_size,
            expected_selection=expected,
        )


class TestLinearRank(WeightBasedSchemaTests):
    """Tests for the linear rank selection schema."""

    def get_selection_schema(self, replacement=False):
        return LinearRank(alpha=2, replacement=replacement)

    def test_correct_weights_for_selection(self):
        self.weights_testing(
            populations=[build_population(size) for size in (2, 4, 8)],
            weights=[
                (2, 4),
                (2, 4, 6, 8),
                (2, 4, 6, 8, 10, 12, 14, 16)
            ]
        )

    @pytest.mark.parametrize('p_size, s_size, alpha, repl, expected', [
        (10, 10, 0, True, [6, 0, 2, 2, 7, 6, 8, 0, 4, 0]),
        (10, 10, 0, False, [6, 0, 3, 2, 8, 7, 9, 1, 4, 5]),
        (10, 5, 0, True, [6, 0, 2, 2, 7]),
        (10, 5, 0, False, [6, 0, 3, 2, 8]),
        (10, 10, 4, True, [7, 1, 5, 4, 8, 8, 9, 2, 6, 1]),
        (10, 10, 4, False, [7, 1, 4, 5, 9, 8, 6, 0, 3, 2]),
        (10, 5, 4, True, [7, 1, 5, 4, 8]),
        (10, 5, 4, False, [7, 1, 4, 5, 9]),
    ])
    def test_correct_selection(self, p_size, s_size, alpha, repl, expected):
        self.selection_testing(
            selection=LinearRank(alpha=alpha, replacement=repl),
            population=build_population(p_size),
            selection_size=s_size,
            expected_selection=expected,
        )
