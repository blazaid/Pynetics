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
"""Tests for all the callbacks."""
from unittest.mock import Mock

from pynetics.callback import Callback, History


class TestCallback:
    """Tests over the base callback instances.

    Possibly, the stupidiest test class, because none of the methods do
    anything by default.
    """

    def test_on_algorithm_begin(self):
        callback = Callback()
        callback.on_algorithm_begin(Mock())

    def test_on_algorithm_end(self):
        callback = Callback()
        callback.on_algorithm_end(Mock())

    def test_on_step_begin(self):
        callback = Callback()
        callback.on_step_begin(Mock())

    def test_on_step_end(self):
        callback = Callback()
        callback.on_step_end(Mock())


class TestHistory:
    """Tests for the History callback instances."""

    def test_no_data_nor_generation_after_initialization(self):
        history = History()
        assert history.generation == 0
        assert history.data == {}

    def test_on_algorithm_begin_resets_everything(self):
        history = History()
        history.generation = 10
        history.data = {i: i for i in range(10)}

        assert history.generation != 0
        assert history.data != {}

        history.on_algorithm_begin(Mock())

        assert history.generation == 0
        assert history.data == {}

    def test_on_step_end(self):
        # The best genotype returned by the algorithm
        best_fitness = 1
        best_genotype = Mock()
        best_genotype.fitness = Mock(return_value=best_fitness)
        # The algorithm
        ga = Mock()
        ga.best = Mock(return_value=best_genotype)
        ga.generation = 1

        history = History()
        assert history.generation == 0
        assert history.data == {}

        history.on_step_end(ga)

        assert history.generation == 1
        assert history.data == {
            'Best genotype': [best_genotype],
            'Best fitness': [best_fitness]
        }
