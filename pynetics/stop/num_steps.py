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
from .. import api


class NumSteps:
    """Stop based on the number of iterations.

    The condition is met once the number of iterations has reached an
    specified limit.
    """

    def __init__(self, steps: int):
        """Initializes this object with the number of iterations.

        :param steps: The number of iterations to do before stop.
        """
        self.steps = steps

    def __call__(self, genetic_algorithm: api.EvolutiveAlgorithm) -> bool:
        """Checks if this stop criteria is met.

        It will look at the generation of the genetic algorithm. It's
        expected that, if its generation is greater or equal than the
        specified in initialization method, the criteria is met.

        :param genetic_algorithm: The genetic algorithm where this stop
            condition belongs.
        :return: True if criteria is met, false otherwise.
        :raise: NoGenerationInGeneticAlgorithm if the genetic_algorithm
            argument does not have a "generation" property.
        """
        return genetic_algorithm.generation >= self.steps
