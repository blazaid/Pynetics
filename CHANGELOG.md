# Changelog

This document holds all the changes in the project.

The format is based on [Keep a Changelog], and this project adheres to
[Semantic Versioning]

## Unreleased
- The `Genotype` has a property that lists its progenitors (it can be retrieved
through the `parents` property), so it is possible to trace back all the family
tree of a given genotype.

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

- Recovered all the old API and rewritten part of the code to provide a
 stable base on which to work.
- Restructured and rewitten the majority of tests to provide a 100%
 coverage.
- MIT Licensed.
- Created this changelog.


[Keep a Changelog]: https://keepachangelog.com/en/1.0.0
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html

