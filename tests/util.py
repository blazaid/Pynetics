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
import itertools
import random
from unittest.mock import Mock

from pynetics import api


class FixRandomSeed:
    """Fixes the random seed for a give function or method.

    Any method called with this decorator will have the random seed set
    as specified in the seed argument, so the behaviour of the method
    will be always the same.

    The first 72 numbers for the `random` function after fixing with the
    default seed are the following:

    >>> [0.639, 0.025, 0.275, 0.223, 0.736, 0.677, 0.892, 0.087, 0.422,\
         0.030, 0.219, 0.505, 0.027, 0.199, 0.650, 0.545, 0.220, 0.589,\
         0.809, 0.006, 0.806, 0.698, 0.340, 0.155, 0.957, 0.337, 0.093,\
         0.097, 0.847, 0.604, 0.807, 0.730, 0.536, 0.973, 0.379, 0.552,\
         0.829, 0.619, 0.862, 0.577, 0.705, 0.046, 0.228, 0.289, 0.080,\
         0.233, 0.101, 0.278, 0.636, 0.365, 0.370, 0.210, 0.267, 0.937,\
         0.648, 0.609, 0.171, 0.729, 0.163, 0.379, 0.990, 0.640, 0.557,\
         0.685, 0.843, 0.776, 0.229, 0.032, 0.315, 0.268, 0.211, 0.943]

    After the method call, the random state will be restored to the one
    it had before the method calling.

    :param f: The function, method, ...
    :param seed: Which seed to set prior to the method call. Defaults to
    42 (the answer to life, the universe and everything).
    """

    def __init__(self, seed: int = 42):
        """Initializes this object.

        :param seed: Which seed to set prior to the method call. It is
            set te 42 by default (the answer to life, the universe and
            everything).
        """
        self.seed = seed
        self.previous_state = None

    def __enter__(self):
        """Sets the seed to the fixed state, saving the previous one."""
        self.previous_state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, *args):
        """Restores the saved state."""
        random.setstate(self.previous_state)


def build_population(size, first_id=0, fill=None):
    """Creates a new population of the specified size.

    Each genotype in the population has a fitness equals to their
    position in it, as well as an id property, so it is easier to
    identify them in the tests. The id and the fitnesses are created in
    such a way they are proportional. That is, the bigger the id in a
    population, the fitter the phenotype.

    :param first_id: The first id in the population, the rest will be
        consecutive. Defaults to 0.
    :param fill: How many genotypes to add to the population. Defaults
        to None, which means that will be completely filled with
        genotypes. Any value greater than n or negative will be treated
        as None.
    """
    if fill is None or fill < 0 or fill > size:
        fill = size
    population = api.Population(size=size, fitness=Mock())
    for i in range(fill):
        genotype = Mock()
        genotype.fitness = Mock(return_value=i)
        genotype.id = i + first_id
        population.append(genotype)
    return population


def random_sequence(values):
    """Generator for a random.whatever dummy.

    It will be generating the same sequence starting over and over again
    once the sequence is exhausted.

    Usage:

    1. Create the generator:
    >>> g = random_sequence([1, 2, 3])
    2. Use it in patch:
    >>> with mock.patch('random.choice', side_effect=lambda _: next(g)):

    :param values: The sequence of values to return.
    """
    for value in itertools.cycle(values):
        yield value
