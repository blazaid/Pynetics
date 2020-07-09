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
"""Tests for the different initializers implementations."""
import abc

import pytest

from pynetics.exception import BoundsCannotBeTheSame
from pynetics.list import ListGenotype
from pynetics.list.alphabet import (
    BINARY,
    NIBBLE,
    OCTAL,
    DECIMAL,
    HEXADECIMAL,
    GENETIC_CODE,
)
from pynetics.list.exception import NotEnoughSymbolsInAlphabet
from pynetics.list.initializer import (
    AlphabetInitializer,
    PermutationInitializer,
)
from tests.test_api import InitializerTests


# ~~~~~~~~~~~~~
# Generic tests
# ~~~~~~~~~~~~~
class ListInitializerTests(InitializerTests, metaclass=abc.ABCMeta):

    def test_initialize_with_different_genotype_class(self):
        class MyListGenotype(ListGenotype):
            pass

        initializer = self.get_instance(cls=MyListGenotype)

        assert type(initializer.create()) is MyListGenotype


class IntervalInitializerTests(ListInitializerTests, metaclass=abc.ABCMeta):
    @pytest.mark.parametrize('lower, upper', [(0, 1), (0, 10), (1, 10)])
    def test_correct_lower_and_upper_bounds(self, lower, upper):
        initializer = self.get_instance(lower=lower, upper=upper)
        assert initializer.lower == lower
        assert initializer.upper == upper

    @pytest.mark.parametrize('lower, upper', [(1, 0), (10, 0), (10, 1)])
    def test_switched_lower_and_upper_bounds(self, lower, upper):
        initializer = self.get_instance(lower=lower, upper=upper)
        assert initializer.lower == upper
        assert initializer.upper == lower

    @pytest.mark.parametrize('lower, upper', [(0, 0), (1, 1), (10, 10)])
    def test_bounds_cannot_be_the_same(self, lower, upper):
        with pytest.raises(BoundsCannotBeTheSame):
            self.get_instance(lower=lower, upper=upper)

    @pytest.mark.parametrize('lower, upper, size', [(0, 1, 100)])
    def test_genotypes_have_genes_between_bounds(self, lower, upper, size):
        initializer = self.get_instance(size=size, lower=lower, upper=upper)
        genotype = initializer.create()

        for gene in genotype:
            assert lower <= gene <= upper


# ~~~~~~~~~~~~~~
# Specific tests
# ~~~~~~~~~~~~~~
class TestAlphabetInitializer(ListInitializerTests):
    def get_instance(self, **kwargs):
        size = kwargs.get('size', 4)
        alphabet = kwargs.get('alphabet', GENETIC_CODE)
        cls = kwargs.get('cls', None)
        return AlphabetInitializer(size=size, alphabet=alphabet, cls=cls)

    @pytest.mark.parametrize('alphabet', [
        BINARY, NIBBLE, OCTAL, DECIMAL, HEXADECIMAL, GENETIC_CODE
    ])
    @pytest.mark.parametrize('size', [2, 4, 8, 16, 32, 64])
    def test_genotype_creation(self, alphabet, size):
        initializer = self.get_instance(alphabet=alphabet, size=size)
        genotype = initializer.create()

        assert len(genotype) == size
        for g in genotype:
            assert g in alphabet.genes


class TestPermutationInitializer(ListInitializerTests):
    def get_instance(self, **kwargs):
        size = kwargs.get('size', 4)
        alphabet = kwargs.get('alphabet', GENETIC_CODE)
        cls = kwargs.get('cls', None)
        return PermutationInitializer(size=size, alphabet=alphabet, cls=cls)

    @pytest.mark.parametrize('alphabet, size', [
        (BINARY, len(BINARY) + 1),
        (BINARY, len(BINARY) + 2),
        (NIBBLE, len(NIBBLE) + 1),
        (NIBBLE, len(NIBBLE) + 2),
        (OCTAL, len(OCTAL) + 1),
        (OCTAL, len(OCTAL) + 2),
        (DECIMAL, len(DECIMAL) + 1),
        (DECIMAL, len(DECIMAL) + 2),
        (HEXADECIMAL, len(HEXADECIMAL) + 1),
        (HEXADECIMAL, len(HEXADECIMAL) + 2),
        (GENETIC_CODE, len(GENETIC_CODE) + 1),
        (GENETIC_CODE, len(GENETIC_CODE) + 2),
    ])
    def test_size_cannot_be_bigger_than_alphabet_size(self, alphabet, size):
        with pytest.raises(NotEnoughSymbolsInAlphabet):
            self.get_instance(alphabet=alphabet, size=size)

    @pytest.mark.parametrize('alphabet', [
        BINARY, NIBBLE, OCTAL, DECIMAL, HEXADECIMAL, GENETIC_CODE
    ])
    def test_genotype_creation(self, alphabet):
        for size in range(1, len(alphabet) + 1):
            initializer = self.get_instance(alphabet=alphabet, size=size)
            genotype = initializer.create()

            assert len(genotype) == size
            for g in genotype:
                assert g in alphabet.genes
