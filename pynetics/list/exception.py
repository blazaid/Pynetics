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
"""Definition specific errors for list based genetic algorithms.
"""
from typing import Any, Iterable

from ..exception import RecombinationError, PyneticsError


# ~~~~~~~~~~~~~~~
# Alphabet errors
# ~~~~~~~~~~~~~~~
class AlphabetError(PyneticsError):
    """Errors related to the Alphabet classes."""


class TooFewGenes(AlphabetError):
    """Very few genes to fill up the alphabet."""

    def __init__(self, *, genes: Iterable[Any]):
        """Initializes this object.

        :param genes: The genes involved in the error.
        """
        super().__init__(f'Too few genes to conform up the alphabet: {genes}')


class NotEnoughSymbolsInAlphabet(AlphabetError):
    """The alphabet hasn't as many symbols as needed."""

    def __init__(self, *, expected: int, real: int):
        """Initializes this object.

        :param expected: How many different symbols we need.
        :param real: How many different symbols we have.
        """
        super().__init__(f'Asked for {expected} symbols but have {real}')


# ~~~~~~~~~~~~~~~~~~~~
# Recombination errors
# ~~~~~~~~~~~~~~~~~~~~
class EmptyListGenotype(RecombinationError):
    """A genotype with at least one gene was needed, but it was zero."""
