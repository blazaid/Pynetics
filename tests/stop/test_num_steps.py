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
from unittest.mock import Mock

import pytest

from pynetics.stop import NumSteps


class TestNumSteps:
    """Test for the `NumSteps` stop condition."""

    @pytest.mark.parametrize('steps', [10])
    def test_criteria_not_met_when_lower_steps(self, steps):
        """Not met when generations are lower than expected."""
        stop_condition = NumSteps(steps=steps)

        genetic_algorithm = Mock()
        for generation in range(steps):
            genetic_algorithm.generation = generation
            assert not stop_condition(genetic_algorithm)

    @pytest.mark.parametrize('steps', [10])
    def test_criteria_met_when_higher_fitness(self, steps):
        """Met when generations are greater or equal than expected."""
        stop_condition = NumSteps(steps=steps)

        genetic_algorithm = Mock()
        for generation in range(steps + 1, steps + 10):
            genetic_algorithm.generation = generation
            assert stop_condition(genetic_algorithm)
