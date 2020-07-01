.. _examples_ones_counting_with_callbacks:

Ones counting with callbacks
============================

In this example, we introduce the Callback instances. When creating a new
instance of a GeneticAlgorithm, the  user has the option of providing a list of
callbacks, objects that are called whenever an event is triggered. The previous
:ref:`Ones counting <examples_ones_counting>` examples will be used as base for
this one.

There is a class in the library called Callback, which defines the possible
events, but actually any object with the required methods is valid as a
callback. The methods, corresponding to the events, are self-explanatory, and
are the following:

- :code:`on_algorithm_begins(self, ga: api.EvolutiveAlgorithm)`
- :code:`on_algorithm_ends(self, ga: api.EvolutiveAlgorithm)`
- :code:`on_step_begins(self, ga: api.EvolutiveAlgorithm)`
- :code:`on_step_ends(self, ga: api.EvolutiveAlgorithm)`

Defining a callback
-------------------

Our example defines a callback called :code:`MyCallback` which inherits from
the :code:`Callback` helper class, redefining the four methods:

.. code-block:: python

 class MyCallback(Callback):
    def on_algorithm_begins(self, g):
        print('Start algorithm')

    def on_step_begins(self, g):
        print(f'Generation: {g.generation}\t', end='')

    def on_step_ends(self, g):
        print(f'{g.best().phenotype()}\tfitness: {g.best().fitness():.2f}')

    def on_algorithm_ends(self, g):
        print('End algorithm')

Algorithm configuration
-----------------------

The configuration is the same as the previous example with the difference that
we've added a new argument, :code:`callback` with a list of objects to be used
as callbacks; in our case, only one, an instance of :code:`MyCallback`:

.. code-block:: python

    ga = GeneticAlgorithm(
        # ...
        # Stuff
        # ...
        callbacks=[MyCallback()],
    )

Then, during the execution of the algorithm, each time an event is triggered
(e.g. a new algorithm iteration or step begins), the corresponding methods from
each callback in the list (e.g. :code:`on_step_begins`) are called.