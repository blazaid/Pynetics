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
"""Tests for all the utility code."""
from unittest import mock

import pytest

from pynetics.util import take_chances
from tests.util import random_sequence


@pytest.mark.parametrize('probability, rand, expected', [
    (0, 0.00001, False),
    (0, 0.1, False),
    (0, 0.9, False),
    (0.1, 0, True),
    (0.1, 0.09999, True),
    (0.1, 0.1, False),
    (0.1, 0.9, False),
    (0.5, 0, True),
    (0.5, 0.49999, True),
    (0.5, 0.5, False),
    (0.5, 0.9, False),
    (1, 0, True),
    (1, 0.5, True),
    (1, 0.99999, True),
])
def test_take_chances(probability, rand, expected):
    seq = random_sequence([rand])
    with mock.patch('random.random', side_effect=lambda: next(seq)):
        assert take_chances(probability) == expected
