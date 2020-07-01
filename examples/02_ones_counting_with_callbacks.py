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
"""Same example as `01_ones_counting.py`, but this time we add a
callback to show how it works.
"""
from pynetics.algorithm import GeneticAlgorithm
from pynetics.callback import Callback
from pynetics.list.alphabet import BINARY
from pynetics.list.initializer import AlphabetInitializer
from pynetics.list.mutation import RandomGene
from pynetics.list.recombination import random_mask
from pynetics.replacement import high_elitism
from pynetics.selection import Tournament
from pynetics.stop import FitnessBound

TARGET_LEN = 50


def fitness(phenotype):
    """Calculated as the sum of the 1's by the length of the chromosome.

    :return: The fitness of the individual.
    """
    return sum(phenotype) / len(phenotype)


class MyCallback(Callback):
    def on_algorithm_begins(self, g):
        print('Start algorithm')

    def on_step_begins(self, g):
        print(f'Generation: {g.generation}\t', end='')

    def on_step_ends(self, g):
        print(f'{g.best().phenotype()}\tfitness: {g.best().fitness():.2f}')

    def on_algorithm_ends(self, g):
        print('End algorithm')


if __name__ == '__main__':
    ga = GeneticAlgorithm(
        population_size=4,
        initializer=AlphabetInitializer(size=TARGET_LEN, alphabet=BINARY),
        stop_condition=FitnessBound(1),
        fitness=fitness,
        selection=Tournament(3),
        replacement=high_elitism,
        replacement_ratio=1.0,
        recombination=random_mask,
        recombination_probability=1.0,
        mutation=RandomGene(BINARY),
        mutation_probability=1 / TARGET_LEN,
        callbacks=[MyCallback()],
    )
    ga.run()
