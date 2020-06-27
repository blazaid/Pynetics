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
"""Implementations for list based genotypes.
"""
from __future__ import annotations

import collections
from typing import Any, Iterable, List

from .. import api


# ~~~~~~~~~~~~~~~~~~~~~~~~
# Genotype implementations
# ~~~~~~~~~~~~~~~~~~~~~~~~
class ListGenotype(
    api.Genotype,
    collections.abc.MutableSequence,
):
    """A list based genotype, used for various implementations.

    It also behaves as a MutableSequence (provides __init__, __len__,
    __getitem__, __setitem__, __delitem__, and insert), so be used too
    almost like a drop-in replacement for a list.
    """

    def __init__(self, *, genes: Iterable[Any] = None):
        """Initializes this object.

        :param genes: A sequence of genes for this genotype to be
            initialized with.
        """
        super().__init__()
        self.genes = list(genes) if genes is not None else []

    def phenotype(self) -> List[Any]:
        """Returns the list of genes conforming this genotype.

        :return: An ordered list with the elements conforming this
            genotype.
        """
        return self.genes[:]

    def __eq__(self, other: ListGenotype) -> bool:
        """The equality between two list genotypes is True if they:

        1. Have the same length
        2. Any two genes in the same position have the same value.

        :param other: The other genotype to compare with.
        :return: True if both genotypes are considered equal or false
            otherwise.
        """
        if len(self) == len(other):
            # They have the same length, lets check the contents
            return all(x == y for (x, y) in zip(self, other))
        else:
            return False

    def __getitem__(self, index: int) -> Any:
        """Returns the item in position index.

        :param index: The position of the gene to retrieve.
        :return: The item in the given position.
        :raises IndexError: In case the index is out of bounds or
            doesn't exist.
        """
        return self.genes[index]

    def __delitem__(self, index: int):
        """Deletes the item in position index.

        :param index: The index of the element to delete.
        :raise ValueError: The index is out of bounds ot doesn't exist.
        """
        self.genes.remove(index)

    def insert(self, index: int, value: Any):
        """Inserts an element in the given index.

        If the given index is greater or equal to the length of the
        genotype, the value is inserted at the end of it, regardless the
        length of it.

        The elements to the right are shifted one position.

        :param index: The position to insert the value.
        :param value: Which value insert.
        """
        self.genes.insert(index, value)

    def __setitem__(self, index: int, value: Any):
        """Sets the value in the given index, replacing it.

        :param index: The position to set the value.
        :param value: Which value set.
        :raise IndexError: If the index
        """
        self.genes[index] = value

    def __len__(self) -> int:
        """The size iof the genotype (its amount of genes)

        :return: The size in number of genes.
        """
        return len(self.genes)

    def __str__(self) -> str:
        """An user friendly string representation for this genotype.

        :return: A string with the contents of the genotype (in a user
            friendly form).
        """
        return ','.join(str(g) for g in self.genes)
