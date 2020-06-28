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
"""The card problem: You've got a deck of N cards numbered from 1 to N.
The idea is to divide the deck in two piles where the cards in one pile
sum as close as possible to I and the ones in the other multiply as
close as possible to J.
"""
import operator
from functools import reduce

from pynetics.algorithm import GeneticAlgorithm
from pynetics.callback import Callback
from pynetics.list import alphabet
from pynetics.list.genotype import ListGenotype
from pynetics.list.initializer import AlphabetInitializer
from pynetics.list.mutation import RandomGene
from pynetics.list.recombination import random_mask
from pynetics.replacement import low_elitism
from pynetics.selection import Tournament
from pynetics.stop import FitnessBound

N = 10  # How many cards in our deck
TARGET_I = 36  # Sum target
TARGET_J = 360  # Product target


# Encoding
#
# One way to encode this problem is as binary genotypes. This way, a
# size 10 genotype can represent each card position (indices 0 to 9 are
# values 1 to 10) and each gene value can represent the deck where the
# card is placed (values 0 or 1 to decks 1 or 2).
#
# So a ListGenotype is enough, but we're gonna show how to extend its
# behaviour by providing it a new phenotype method.
class CardsGenotype(ListGenotype):
    def phenotype(self):
        """The phenotype will be a tuple of two decks containing a list
        of the cards included on that decks."""
        # Those positions marked as 0 will belong to the deck 1 (sum)
        deck1 = [i for (i, g) in enumerate(self, start=1) if g == 0]
        # Those positions marked as 1 will belong to the deck 2 (productS)
        deck2 = [i for (i, g) in enumerate(self, start=1) if g == 1]

        return deck1, deck2


def fitness(phenotype):
    """Fitness will be based on the distance error to the target values.

    The error will be the sum of both errors, which will be computed as
    the difference to the targets.
    """
    deck1, deck2 = phenotype

    error1 = abs((sum(deck1) - TARGET_I) / TARGET_I)
    error2 = abs((reduce(operator.mul, deck2, 1) - TARGET_J) / TARGET_J)

    error = error1 + error2

    # Return the fitness according to that error
    return 1 / (1 + error)


class MyCallback(Callback):
    def on_step_ends(self, g):
        print('generation: {}\tfitness: {:.2f}\tIndividual: {}'.format(
            g.generation,
            g.best().fitness(),
            g.best().phenotype(),
        ))


if __name__ == '__main__':
    ga = GeneticAlgorithm(
        population_size=10,
        initializer=AlphabetInitializer(
            size=N, alphabet=alphabet.BINARY, cls=CardsGenotype
        ),
        stop_condition=FitnessBound(1),
        fitness=fitness,
        selection=Tournament(2),
        recombination=(random_mask, 1.0),
        mutation=(RandomGene(alphabet.BINARY), 1 / N),
        replacement=(low_elitism, 0.9),
        callbacks=[MyCallback()]
    )

    history = ga.run()
