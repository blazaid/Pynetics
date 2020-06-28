.. _examples_two_decks_problem:

Two decks problem
=================

In the two decks problem there's a deck of N cards, numbered from 1 to N, and
they have to be divided in two decks in such a way that the cards in one pile
sum as close as possible to :code:`I` and the cards in the other multiply as
close as possible to :code:`J`.

We can use a ListGenotype with a binary alphabet, so a length of N represents
N cards ranging from 1 to N, and each gene value (i.e. 0 or 1) maps that card
to one deck (i.e. 1 or 2 respectively).

Customizing the phenotype
-------------------------

We're gonna introduce a new concept. A :code:`ListGenotype` is enough, yes, but
it will be nice to vary it's genotype. We dont care about the internal
representation; we only want the actual individual, that is, the two decks.

It requires two things; first, overriding the :code:`phenotype` method
(obviously), and second, telling to the initializer that the class of the
:code:`ListGenotype` instances is not the standard, but is the new one created
by us. So let's get to it.

First, our :code:`ListGenotype` subclass:

.. code-block:: python

    class CardsGenotype(ListGenotype):
        def phenotype(self):
            """The phenotype will be a tuple of two decks containing a list
            of the cards included on that decks."""
            # Those positions marked as 0 will belong to the deck 1 (sum)
            deck1 = [i for (i, g) in enumerate(self, start=1) if g == 0]
            # Those positions marked as 1 will belong to the deck 2 (productS)
            deck2 = [i for (i, g) in enumerate(self, start=1) if g == 1]

            return deck1, deck2

Once we have our class (and it's a :code:`ListGenotype` subclass), we just need
to create the initializer advising it to use our custom class instead of the
default :code:`ListGenotype`.

.. code-block:: python

    initializer = AlphabetInitializer(
            size=N, alphabet=alphabet.BINARY, cls=CardsGenotype
        )

We can now continue with the fitness implementation.

Fitness
-------

The idea is to reach two targets. In the example are specified as two
variables, :code:`TARGET_I` and :code:`TARGET_J`:

.. code-block:: python

    TARGET_I = 36   # Sum target
    TARGET_J = 360  # Product target

In our fitness, we will compute the error to each target (using the
proportional difference) and then add it. Maybe there is a better solution, but
this one does the trick.

.. code-block:: python

    def fitness(phenotype):
        deck1, deck2 = phenotype

        error1 = abs((sum(deck1) - TARGET_I) / TARGET_I)
        error2 = abs((reduce(operator.mul, deck2, 1) - TARGET_J) / TARGET_J)

        error = error1 + error2

        # Return the fitness according to that error
        return 1 / (1 + error)

Check the first line of the function. Do you see how the custom phenotype is
being used? Nice, now we're prepared to configure and run the algorithm as
usual.

Algorithm configuration
-----------------------

Let's configure out algorithm. As we did before, it is enough to use the
default operators, but this time telling to the :code:`AlphabetInitializer`
that the :code:`ListGenotype` class to use is out implementation.

.. code-block:: python

    ga = GeneticAlgorithm(
        # ... Stuff ...
        initializer=AlphabetInitializer(
            size=N, alphabet=alphabet.BINARY, cls=CardsGenotype
        ),
        # ... More stuff ...
    )
