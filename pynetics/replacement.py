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
"""Replacement algorithms.
"""
import abc
import logging
from typing import Optional

from . import api


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Abstract replacement schemas
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ReplacementSchema(metaclass=abc.ABCMeta):
    """Groups common behavior across all the replacement schemas.

    The replacement schema is defined as a class. However, it is enough
    to implement it a replacement method, i.e. a function that receives
    two populations (original and offspring) and returns a population
    resulting from the combination of the previous two.
    """

    def __init__(self):
        """Initializes this object."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def __call__(
            self, *,
            population: api.Population,
            offspring: api.Population,
            max_size: Optional[int] = None,
    ) -> api.Population:
        """Executes the replacement method.

        This method performs some checks to the arguments and delegates
        the implementation to the abstract method
        :func:`~replacement.ReplacementSchema.do`.

        :param population: The original population.
        :param offspring: The population to replace the original one.
        :param max_size: The max size of the new population. If not
            specified or invalid, the size will be set as the one of the
            original population.
        :return: A new Population instance.
        """
        # Set the size for the new population
        if max_size is None:
            max_size = population.max_size
        elif not isinstance(max_size, int):
            self.logger.warning(
                f'Size for the new population not valid ({max_size}); using '
                f'the base population size instead'
            )
            max_size = population.max_size
        elif max_size < 1:
            self.logger.warning(
                f'Size too small for the new population ({max_size}); '
                f'using the base population size instead'
            )
            max_size = population.max_size

        return self.do(
            population=population, offspring=offspring, max_size=max_size
        )

    @abc.abstractmethod
    def do(
            self, *,
            population: api.Population,
            offspring: api.Population,
            max_size: Optional[int] = None,
    ) -> api.Population:
        """Executes this particular implementation of selection.

        This method is not called by the base algorithms implemented,
        but from :func:`~selection.SelectionSchema.__call__` instead.
        It should contain the logic of the specific selection schema.

        :param population: The original population.
        :param offspring: The population to replace the original one.
        :param max_size: The size of the new population. It is
            guaranteed that the value will be a valid int value.
        :return: A new Population instance.
        """


# ~~~~~~~~~~~~~~~~~~~
# Replacement schemas
# ~~~~~~~~~~~~~~~~~~~
# TODO Actually the high and low elitism schemas can be parametrised
#  resulting in even more different replacement schemes.
class HighElitism(ReplacementSchema):
    """Replacement with the fittest among both population and offspring.

    Only those best genotypes among both populations will be selected,
    thus discarding those less fit. This makes this operator extremely
    elitist.
    """

    def do(
            self, *,
            population: api.Population,
            offspring: api.Population,
            max_size: Optional[int] = None,
    ) -> api.Population:
        """Executes this replacement.

        :param population: The original population.
        :param offspring: The population to replace the original one.
        :param max_size: The size of the new population. It is
            guaranteed that the value will be a valid int value.
        :return: A new Population instance.
        """
        # Concat both populations
        new_population = population + offspring

        # Sort it by fitness to have the best ones at the beginning
        new_population.sort()

        # Just delete the works ones until population has the expected
        # size and set it as the maximum size of the new population
        del new_population[:-max_size]
        new_population.max_size = max_size

        # Return the newly created population
        return new_population


high_elitism = HighElitism()


class LowElitism(ReplacementSchema):
    """Replaces the less fit of the population with the fittest of the
    offspring.

    The method will replace the less fit genotypes by the best ones of
    the offspring. This makes this operator elitist, but at least not
    much. Moreover, if the offspring size equals to the population
    size then it's a full replacement (i.e. a generational scheme).
    """

    def do(
            self, *,
            population: api.Population,
            offspring: api.Population,
            max_size: Optional[int] = None,
    ) -> api.Population:
        """Executes this replacement.

        :param population: The original population.
        :param offspring: The population to replace the original one.
        :param max_size: The size of the new population. It is
            guaranteed that the value will be a valid int value.
        :return: A new Population instance.
        """
        # Create the new population to fill with the genotypes
        new_population = api.Population(
            size=max_size, fitness=population.fitness
        )

        # First, we add the best genotypes in the offspring population
        # that fit in the new population.
        num_genotypes = min(max_size, len(offspring))
        offspring.sort()
        new_population.extend(offspring[-num_genotypes:])

        # Second, we fill the new population with the best genotypes
        # from the old population (if needed).
        num_genotypes = max_size - len(new_population)
        if num_genotypes > 0:
            population.sort()
            new_population.extend(population[-num_genotypes:])

        # Return the newly created population
        return new_population


low_elitism = LowElitism()
