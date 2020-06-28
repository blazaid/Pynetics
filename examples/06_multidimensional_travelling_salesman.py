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
"""This problem is like the travelling salesman problem, but in a
n-dimensional space.
"""
import math
import random

from pynetics.algorithm import GeneticAlgorithm
from pynetics.callback import Callback
from pynetics.list.alphabet import Alphabet
from pynetics.list.initializer import PermutationInitializer
from pynetics.list.mutation import swap_genes
from pynetics.list.recombination import pmx
from pynetics.replacement import low_elitism
from pynetics.selection import Tournament
from pynetics.stop import NumSteps

N = 25  # Number of cities to visit
D = 3  # dimensions in our cities universe
CITIES = [tuple(random.uniform(-100, 100) for d in range(D)) for _ in range(N)]


# 1. Encoding
#
# The cities encoding is usually a list containing a permutation of the
# range 1 to N (being N the number of cities). We will use then a
# ListChromosome. The differences will be in the initializer (how to
# create those permutations) and the mating and mutation operations
# (which has to deal with how to combine those chromosome without
# generating non-valid ones).
#
# We are going to write a helper method in order to represent the
# individual

def route_and_distance(phenotype):
    # Extract the values of the cards on each deck
    def dist(x, y):
        return sum([math.pow(a - b, 2) for a, b in zip(x, y)])

    total_distance = 0
    for i in range(len(phenotype) - 1):
        total_distance += dist(
            CITIES[phenotype[i]],
            CITIES[phenotype[i + 1]]
        )

    return '->'.join(str(g) for g in phenotype), total_distance


# 2. Fitness
#
# Now that we have a encoding, lets compute the fitness. The idea is to
# cover the minimum possible euclidean distance so the higher the
# distance, the lower the individual's fitness.
def fitness(phenotype):
    """Computes the fitness given a genotype.

    Computes the distance and returns a fitness inversely proportional
    to that distance. We use the squared euclidean as is less expensive.
    """
    _, total_distance = route_and_distance(phenotype)

    # Compute the fitness according to that error
    return 1 / (1 + total_distance)


class MyCallback(Callback):
    def on_step_ends(self, g):
        best = g.best()

        path, total_distance = route_and_distance(best.phenotype())

        if g.generation % 10 == 0:
            print(f'{g.generation}\tfitness: {best.fitness():.9f}'
                  f'\tDistance: {total_distance:.2f}\tPath: {path}')


alphabet = Alphabet(genes=range(N))

if __name__ == '__main__':
    ga = GeneticAlgorithm(
        population_size=10,
        initializer=PermutationInitializer(size=N, alphabet=alphabet),
        stop_condition=NumSteps(100),
        fitness=fitness,
        selection=Tournament(2),
        recombination=pmx,
        recombination_probability=1.0,
        mutation=swap_genes,
        mutation_probability=1 / N,
        replacement=(low_elitism, 0.9),
        callbacks=[MyCallback()]
    )

    history = ga.run()
