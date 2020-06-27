.. _examples_ones_counting_finish_abruptly:

Ones counting finishing abruptly
================================

The callbacks in a genetic algorithm allows us to alter its behaviour while it
is running. This means we can change any operator depending on other factors
(e.g. altering mutation probability as a function of diversity).

In this example, we're gonna replicate the previous :ref:`Ones counting with
callbacks <examples_ones_counting_with_callbacks>`, but it'll be stopped once a
series of :code:`m` consecutive ones are generated.

The new callback
----------------

Lacking a better name, we decided to call the new callback
:code:`StopBecauseIAmWorthIt`. It'll be implemented as follows:

.. code-block:: python

    class StopBecauseIAmWorthIt(Callback):
        """Stops if there are n consecutive 1's in the genotype"""

        def __init__(self, m):
            self.m = m
            self.ones = '1' * self.m

        def on_step_ends(self, g):
            for genotype in g.population:
                if self.ones in ''.join(map(str, genotype)):
                    print(f"Stopping because we found {self.m} consecutive 1's")
                    g.stop()

In short, it will stop as soon as at least one genotype in the population has
:code:`m` consecutive ones. This'll be done by calling the genetic algorithm
:code:`stop` method.

Algorithm configuration
-----------------------

The configuration is the same as the previous example, but using an instance of
our new callback:

.. code-block:: python

    ga = GeneticAlgorithm(
        # ...
        # Stuff
        # ...
        callbacks=[StopBecauseIAmWorthIt(m=42)]
    )

The callback is configured to alter the genetic algorithm behavior once 42
consecutive ones are found in any genotype.