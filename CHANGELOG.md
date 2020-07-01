# Changelog

This document holds all the changes in the project.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning]

## Unreleased
- There were some classes that made impossible to change their behaviour in run
time. They've been update. Those classes are:
    - `SelectionSchema`: Now it is possible to change the `replacement`
    attribute, and switch it from a without-replacement schema to a
    with-replacement one.
    - `GeneticAlgorithm`: Here there are a number of changes:
        - Although`recombination` could be dynamically modified, the selection
        size didn't change with it, so it may lead to an error. Now, the
        selection size changes according to the new recombination parameter.
        - The `population_size` has also some lateral effects that have to do
        with the `offspring_size`. It has also been changed.
        - Also the `fitness`, which affects to the `population` and the
        `genotypes` inside it.
    - Don't remember more changes, but from now on, anything detected that may
    be made modifiable in runtime, it'll have its own issue.

## 0.7.0 - 2020-06-28
- The `Genotype` class now has a `phenotype` abstract method that must be
overrode to obtain the individual. This phenotype is the object to be used as
argument in the `fitness` method.
- The ListGenotype class has been modified to provide a default phenotype
implementation.
- The `Initializer` subclasses that deal with `ListGenotype` instances now
accept a class that indicates which `ListGenotype` subclass to use. If not
specified, the base `ListGenotype` is used.
- Arguments `recombination` and `mutation` are now optional parameters in the
`GeneticAlgorithm` class.

## 0.6.1 - 2020-06-20
- Fixed CI/CD configuration for deployment on PyPi.

## 0.6.0 - 2020-06-18

- Recovered all the old API and rewritten part of the code to provide a stable
base on which to work.
- Restructured and rewritten the majority of tests to provide a 100% coverage.
- MIT Licensed.
- Created this changelog.


[Keep a Changelog]: https://keepachangelog.com/en/1.0.0
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html

