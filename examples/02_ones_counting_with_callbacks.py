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


def fitness(genotype):
    """The more 1s are in the genotype, the higher the fitness.

    :return: The fitness of the genotype's phenotype.
    """
    return sum(genotype.genes) / len(genotype)


class MyCallback(Callback):
    def on_algorithm_begin(self, g):
        print('Start algorithm')

    def on_step_begin(self, g):
        print('Starting step ... ', end='')

    def on_step_end(self, g):
        print('generation: {}\tfitness: {:.2f}\tIndividual: {}'.format(
            g.generation,
            g.best().fitness(),
            g.best(),
        ))

    def on_algorithm_end(self, g):
        print('End algorithm')


if __name__ == '__main__':
    ga = GeneticAlgorithm(
        population_size=4,
        initializer=AlphabetInitializer(size=TARGET_LEN, alphabet=BINARY),
        stop_condition=FitnessBound(1),
        fitness=fitness,
        selection=Tournament(3),
        recombination=(random_mask, 1.0),
        mutation=(RandomGene(BINARY), 1 / TARGET_LEN),
        replacement=(high_elitism, 1.0),
        callbacks=[MyCallback()],
    )
    ga.run()
