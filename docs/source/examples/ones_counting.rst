.. _examples_ones_counting:

Ones counting
=============

This is an example to show how to create a basic Genetic Algorithm. the idea is
to achieve an all-1s sequence from a population of random binary sequences.

With this example it is possible to explore the basic elements of a genetic
algorithm (e.g. genotypes, operators, etc.) and the existing components in the
Pynetics library to create and use them.

In the example, we've established the genotype size via the next sentence:

.. code-block:: python

   GENOTYPE_LEN = 50

We don't really need to code almost anything else. Pynetics has a set of
components for list-based genetic algorithms that can be used for this kind of
problems. One of them is the ListGenotype, an implementation of a Genotype that
behaves as a list of genes. For now, that's all we need to know.

Fitness
-------

The dependent aspect of our problem is the answer to the question: how can we
tell our algorithm how good an individual is? This is where the fitness comes
into play.

The fitness can be any function that receives an object (representing an
individual) and returns a float value where, the higher the value, the fitter
the individual. We have implemented ours as follows.

.. code-block:: python

def fitness(phenotype):
    return sum(phenotype) / len(phenotype)

This function returns a value that goes from 0 to 1 being 0 the worst that
evolution has been able to create and 1 totally fit. In fact, any function that
admits an object of the Genotype class as an argument and returns a float whose
value is larger the fitter the individual is, would be sufficient.

Algorithm configuration
-----------------------

This is actually the most we're going to program specifically for our problem.
The rest will be to configure the genetic algorithm to perform the iterative
process that it consists of and provide us with a solution.

We're going to write down all the settings and comment on them in parts:

.. code-block:: python

    def fitness(genotype):
        return sum(genotype.genes) / len(genotype)

    ga = GeneticAlgorithm(
        population_size=10,
        initializer=AlphabetInitializer(size=GENOTYPE_LEN, alphabet=BINARY),
        stop_condition=FitnessBound(1),
        fitness=fitness,
        selection=Tournament(3),
        recombination=(random_mask, 1.0),
        mutation=(RandomGene(BINARY), 1 / GENOTYPE_LEN),
        replacement=(high_elitism, 1.0),
    )

:code:`GeneticAlgorithm` is the main class we're going to use. It represents a
genetic algorithm configured with a set of settings to drive its behavior.
Those settings are mandatory and are specified here:

- :code:`population_size`: How many gentoypes will exist in our population at a
  time.
- :code:`initializer': The object in charge of creating individuals when needed
  (almost always in the initialization step of the genetic algorithm), where as
  many as specified in the population size will be created. In our case we will
  use an :code:`AlphabetInitializer`, which will create individuals using a
  :code:`BINARY` alphabet of the specified length.
- :code:`stop_condition`: When the algorithm should stop. In our case, we're
  using an stopper that deals with the genotype fitness.
- :code:`fitness`: Which function we'll use to evaluate the individuals. This
  is the one we defined before.
- :code:`selection`: Which operator the algorithm will use in order to select
  individuals from the population. We use a `tournament selection` schema with
  3 random genotypes.
- :code:`recombination`: A tuple with a recombination algorithm and the actual
  probability for the individuals to actually match. In this case, we're using a
  `random mask` recombination schema with a probability of 1 (there'll always be
  a match).
- :code:`mutation`: Like the recombination, but with the mutation. In our case
  we sill work with a `random gene` schema with a low probability.
- :code:`replacement`: Which replacement schema to use and the replacement
  ratio (percentage of how many individuals are needed in the offspring to
  actually replace populations. We will use a `high elitism` schema with a 100%
  replacement rate (the offspring will have the same length as the original
  population.

Running the algorithm
---------------------

Once the algorithm has been configured, we just need to run it:

.. code-block:: python

    history = ga.run()

It may take a while (I hope not, at least this example), but one it finishes it
will return a history object with information about the execution. In our case,
we recover the best genotype of the last generation and print it, along with
the generation it appeared and its fitness.

.. code-block:: python

    best = history.data['Best genotype'][-1]
    print(history.generation, best, best.fitness())

And that's all! I hope you were under the impression of how to work with the
library.