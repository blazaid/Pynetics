.. _examples_hello_world:

Hello world
===========

In the previous :ref:`ones <examples_ones_counting_with_callbacks>`
:ref:`counting <examples_ones_counting_with_callbacks>` :ref:`examples
<examples_ones_counting_with_callbacks>` we made use of a binary alphabet to
encode our genotypes.

Although a binary alphabet can encode virtually any problem with finite
alphabets, sometimes more complex alphabets are more useful. In this example we
will try to achieve a predefined phrase with an alphabet tailored to our
problem.

The problem is as follows: Let's assume we have an unknown sentence consisting
of letters, numbers, punctuation marks and spaces, and we want to build an
algorithm that finds that sentence.

We just need to program a new fitness function and provide a genotype
initializer with our alphabet.

Fitness
-------

Our target sentence will be in the :code:`TARGET` variable, and will be of
length :code:`TARGET_LEN`. For the fitness we will compare the phenotype (i.e.
the resulting sentence from the genotype) with the target sentence using the
`Hamming distance <https://en.wikipedia.org/wiki/Hamming_distance>`_. The
smaller the distance, the closer the sentence is to the best solution, thus the
higher the fitness.

An implementation for this fitness could be as follows:

.. code-block:: python

   def fitness(genotype):
    """The fitness will be based on the hamming distance error."""
    # Derive the phenotype from the genotype
    phenotype = ''.join(str(x) for x in genotype)
    # Compute the error of this solution
    error = len([i for i in filter(
        lambda x: x[0] != x[1], zip(phenotype, TARGET)
    )])
    # Return the fitness according to that error
    return 1 / (1 + error)

This is not the only implementation, but it is simple enough to see what's
happening. We are assuming that our genotypes are composed of genes belonging
to an specific alphabet. Let's create the initializer that will generate
those genotypes.

Alphabet and the genotype initializer
-------------------------------------

In the previous examples, we used the class :code:`AlphabetInitializer` with a
binary alphabet. Actually, this class works with any alphabet, provided that it
consists of a finite number of elements.

We are going to use exactly the same object, but with a different alphabet, the
one that encodes our solution:

.. code-block:: python

    alphabet = Alphabet(
        genes=string.ascii_letters + string.punctuation + ' '
    )

Now we have an :code:`Alphabet` that codifies our solutions. We can use it to
create our initializer:

.. code-block:: python

    initializer = AlphabetInitializer(size=TARGET_LEN, alphabet=alphabet)

Being :code:`TARGET_LEN` the length of the sentence we'll try to discover and
:code:`alphabet` the alphabet that codifies our solutions.

Algorithm configuration
-----------------------

Let's configure out algorithm. As we did before, it is enough to use the
default operators.

.. code-block:: python

    ga = GeneticAlgorithm(
        population_size=10,
        initializer=AlphabetInitializer(size=TARGET_LEN, alphabet=alphabet),
        stop_condition=FitnessBound(1),
        fitness=fitness,
        selection=Tournament(4),
        recombination=(one_point_crossover, 1.0),
        mutation=(RandomGene(alphabet), 1 / TARGET_LEN),
        replacement=(high_elitism, 1.0),
        callbacks=[MyCallback()]
    )
