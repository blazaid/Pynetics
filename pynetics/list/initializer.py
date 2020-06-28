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
"""API implementations specific of list based genotypes problems.
"""
from __future__ import annotations

import abc
from typing import Type

from .alphabet import Alphabet
from .exception import NotEnoughSymbolsInAlphabet
from .genotype import ListGenotype
from .. import api
from ..exception import BoundsCannotBeTheSame
from ..util import Number


# ~~~~~~~~~~~~~~~~~~~~~
# Abstract initializers
# ~~~~~~~~~~~~~~~~~~~~~
class ListGenotypeInitializer(api.Initializer, metaclass=abc.ABCMeta):
    """Common behaviour between initializers."""

    def __init__(self, *, size: int, cls: ListGenotype.__class__):
        """Initializes this object.

        :param size: The size of the genotypes this initializer will
            generate. Must be greater than 0.
        :param cls: The class of the genotype to create. In not
            specified, it defaults to :py:class`.genotype.ListGenotype`.
        """
        self.size = size
        self.cls = cls


# ~~~~~~~~~~~~
# Initializers
# ~~~~~~~~~~~~
class AlphabetInitializer(ListGenotypeInitializer):
    """Initializer for ListGenotype based on an arbitrary alphabet."""

    def __init__(
            self, *,
            size: int,
            alphabet: Alphabet,
            cls: Type[ListGenotype] = None
    ):
        """Initializes this object.

        :param size: The size of the genotypes to generate.
        :param alphabet: The possible values to fill each gene position.
        """
        super().__init__(size=size, cls=cls or ListGenotype)
        self.alphabet = alphabet

    def create(self) -> ListGenotype:
        """Generates a new random genotype.

        :return: A new ListGenotype instance.
        """
        return self.cls(genes=self.alphabet.get(n=self.size))


class PermutationInitializer(ListGenotypeInitializer):
    """TODO TBD..."""

    def __init__(
            self, *,
            size: int,
            alphabet: Alphabet,
            cls: Type[ListGenotype] = None
    ):
        """TODO TBD..."""
        super().__init__(size=size, cls=cls or ListGenotype)
        self.alphabet = alphabet
        if self.size > len(alphabet):
            raise NotEnoughSymbolsInAlphabet(
                expected=self.size,
                real=len(alphabet)
            )

    def create(self) -> ListGenotype:
        """Generates a new random genotype.

        :return: A new ListGenotype instance.
        """
        genes = self.alphabet.get(n=self.size, rep=False)
        if self.size == 1:
            genes = [genes]
        return self.cls(genes=genes)


class IntervalInitializer(ListGenotypeInitializer, metaclass=abc.ABCMeta):
    """Initializer for genotypes whose genes belong to an interval.

    This is an abstract class that comprises the common behavior for
    genotypes of either integer or real genes.
    """

    def __init__(
            self, *,
            size: int,
            lower: Number,
            upper: Number,
            cls: Type[ListGenotype] = None
    ):
        """Initializes this object.

        :param size: The size of the genotypes to generate.
        :param lower: the lower bound of the interval. If it is greater
            then the upper bound, they will be swapped.
        :param upper: the upper bound of the interval.
        :raise BoundsCannotBeTheSame: If the two bounds have the same
        value.
        """
        super().__init__(size=size, cls=cls or ListGenotype)

        if lower == upper:
            raise BoundsCannotBeTheSame(lower)
        self.lower, self.upper = min(lower, upper), max(lower, upper)

    def create(self) -> ListGenotype:
        """Generates a new random genotype.

        :return: A new genotype instance.
        """
        return self.cls(genes=(
            self.get_value_from_interval() for _ in range(self.size)
        ))

    @abc.abstractmethod
    def get_value_from_interval(self) -> Number:
        """Generates a new value that belongs to the interval.

        :return: A value for the specified interval at initialization.
        """
