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
"""Tests for the alphabets implementations."""
from unittest.mock import Mock

import pytest

from pynetics.exception import MoreGenesRequiredThanExisting
from pynetics.list.alphabet import Alphabet
from pynetics.list.exception import TooFewGenes


class TestAlphabet:

    @pytest.mark.parametrize('genes', [[], [Mock()]])
    def test_init_wrong_genes(self, genes):
        with pytest.raises(TooFewGenes):
            Alphabet(genes=genes)

    @pytest.mark.parametrize('genes, expected', [
        ([0, 1], (0, 1)),
        ([0, 1, 2, 3], (0, 1, 2, 3)),
        ('actg', ('a', 'c', 't', 'g')),
        ([0, 0, 0, 1, 0, 0, 1], (0, 1)),
        ([0, 1, 1, 1, 2, 3, 1], (0, 1, 2, 3)),
        ('actgacacactg', ('a', 'c', 't', 'g')),
    ])
    def test_init_correct(self, genes, expected):
        alphabet = Alphabet(genes=genes)

        assert sorted(alphabet.genes) == sorted(expected)

    @pytest.mark.parametrize('genes', [(0, 1)])
    @pytest.mark.parametrize('n', [4, 8, 16, 32, 64])
    def test_no_rep_cannot_get_more_than_total_genes(self, genes, n):
        alphabet = Alphabet(genes=genes)

        with pytest.raises(MoreGenesRequiredThanExisting):
            alphabet.get(n, rep=False)

    @pytest.mark.parametrize('genes', ['actg'])
    @pytest.mark.parametrize('n', [1, 2, 3, 4])
    def test_get_amounts_without_repetition(self, genes, n):
        alphabet = Alphabet(genes=genes)

        result = alphabet.get(n, rep=False)

        assert len(result) == n
        assert all(g in genes for g in result)

    @pytest.mark.parametrize('genes', ['actg'])
    @pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 7, 8, 9])
    def test_get_amounts_with_repetition(self, genes, n):
        alphabet = Alphabet(genes=genes)

        result = alphabet.get(n, rep=True)

        assert len(result) == n
        assert all(g in genes for g in result)
