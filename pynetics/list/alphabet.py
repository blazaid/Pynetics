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
"""TODO TBD..."""
import random
from typing import Iterable, Any, Union, Sequence

from .exception import TooFewGenes
from ..exception import MoreGenesRequiredThanExisting


# ~~~~~~~~~~~~~~~~~~~
# Base alphabet class
# ~~~~~~~~~~~~~~~~~~~
class Alphabet:
    """Alleles based on a list of equiprobable elements.

    Although it can be initialized by repeated genes, only one of each
    will be stored.
    """

    # TODO Add a weights parameter
    def __init__(self, *, genes: Iterable[Any]):
        """Initializes this object.

        :param genes: All the possible genes to be selected among this
            alphabet. Must be 2 or higher.
        :raise TooFewGenes: If the number of different genes for this
            alphabet is lesser than two.
        """
        # TODO Add weights
        self.__genes = tuple(sorted(set(genes)))
        if len(self.__genes) < 2:
            raise TooFewGenes(genes=self.__genes)
        self.__size = len(self.__genes)

    @property
    def genes(self):
        """The genes that conform this alphabet."""
        return tuple(self.__genes[:])

    def __len__(self):
        """Returns the amount of available symbols in this alphabet.

        :return: The size of this alphabet.
        """
        return len(self.__genes)

    def get(self, n: int = 1, rep: bool = True) -> Union[Any, Sequence[Any]]:
        """Returns 1-to-n random values among all the possible values.

        The method will return either a random allele from the whole
        genetic pool or a sequence of alleles, with or without
        repetition.

        :param n: The number of genes to return. Defaults to 1.
        :param rep: If n > 1, True means repeated alleles may be
            returned, whereas False means no repeated alleles will be
            returned. Defaults to True.
        :return: An allele in case n = 1 or a sequence of alleles in
            case n > 1.
        :raise MoreGenesRequiredThanExisting: If n > len(genetic pool)
            and rep is False (if we don't allow repetition, it is
            impossible to return more values than there are.
        """
        # Go for the genes!
        if n == 1:
            # Easy, only one gene requested
            return random.choice(self.__genes)
        elif rep:
            # More than one allele, but can be repeated
            return random.choices(self.__genes, k=n)
        elif n <= self.__size:
            # More than one and without repetition, but the pool size is
            # big enough
            return random.sample(self.__genes, k=n)
        else:
            # Well, that's an error. Too many genes requested without
            # repetition
            raise MoreGenesRequiredThanExisting(
                req_size=n,
                pool_size=self.__size
            )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Some standard alphabet implementations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BINARY = Alphabet(genes=range(2))
NIBBLE = Alphabet(genes=range(4))
OCTAL = Alphabet(genes=range(8))
DECIMAL = Alphabet(genes=range(10))
HEXADECIMAL = Alphabet(genes=range(16))
GENETIC_CODE = Alphabet(genes='ACTG')
