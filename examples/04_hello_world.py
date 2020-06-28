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
"""In this case, we're trying to reach a target sentence. It needs a
different alphabet, but it is almost the same problem as ones counting.
"""
import string

from pynetics.algorithm import GeneticAlgorithm
from pynetics.callback import Callback
from pynetics.list.initializer import Alphabet, AlphabetInitializer
from pynetics.list.mutation import RandomGene
from pynetics.list.recombination import one_point_crossover
from pynetics.replacement import high_elitism
from pynetics.selection import Tournament
from pynetics.stop import FitnessBound

TARGET = 'Hello, World!'
TARGET_LEN = len(TARGET)


def fitness(phenotype):
    """The fitness will be based on the hamming distance error."""
    # Derive the phenotype from the genotype
    sentence = ''.join(str(x) for x in phenotype)
    # Compute the error of this solution
    error = len([i for i in filter(
        lambda x: x[0] != x[1], zip(sentence, TARGET)
    )])
    # Return the fitness according to that error
    return 1 / (1 + error)


class MyCallback(Callback):
    def on_step_ends(self, g):
        print('generation: {}\tfitness: {:.2f}\tIndividual: {}'.format(
            g.generation,
            g.best().fitness(),
            ''.join(g.best()),
        ))


alphabet = Alphabet(
    genes=string.ascii_letters + string.punctuation + ' '
)

if __name__ == '__main__':
    ga = GeneticAlgorithm(
        population_size=10,
        initializer=AlphabetInitializer(size=TARGET_LEN, alphabet=alphabet),
        stop_condition=FitnessBound(1),
        fitness=fitness,
        selection=Tournament(4),
        recombination=(one_point_crossover, 1.0),
        mutation=RandomGene(alphabet),
        mutation_probability=1 / TARGET_LEN,
        replacement=(high_elitism, 1.0),
        callbacks=[MyCallback()]
    )

    history = ga.run()
    print(history)
