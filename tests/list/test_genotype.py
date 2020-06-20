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
"""Tests for all the selection schema."""

import pytest

from pynetics.list.genotype import ListGenotype
from tests.test_api import GenotypeTests


class TestListGenotype(GenotypeTests):
    """Base tests for all the ListGenotype implementations."""

    def get_instance(self, genes=None):
        return ListGenotype(genes=genes)

    @pytest.mark.parametrize('genes_1, genes_2', [
        ([], []),
        ([1], [1]),
        ([1, 2], [1, 2]),
        ([1, 2, 3], [1, 2, 3]),
    ])
    def test_check_equality_between_two_genotypes(self, genes_1, genes_2):
        genotype_1 = self.get_instance(genes=genes_1)
        genotype_2 = self.get_instance(genes=genes_2)

        assert genotype_1 == genotype_2

    @pytest.mark.parametrize('genes_1, genes_2', [
        ([], [1]),
        ([1], []),
        ([1, 2], [1]),
        ([1], [1, 2]),
        ([1, 2],
         [2, 1]),
    ])
    def test_check_inequality_between_two_genotypes(self, genes_1, genes_2):
        genotype_1 = self.get_instance(genes=genes_1)
        genotype_2 = self.get_instance(genes=genes_2)

        assert genotype_1 != genotype_2

    def test_no_content_for_empty_initialized(self):
        """No content for empty initialized genotypes."""
        genotype = self.get_instance()

        assert len(genotype) == 0
        assert genotype == []

    @pytest.mark.parametrize('genotype_size', [0, 2, 4, 6, 8])
    def test_get_item_out_of_position(self, genotype_size):
        genotype = self.get_instance(genes=range(genotype_size))

        for i in range(genotype_size, genotype_size + 4):
            with pytest.raises(IndexError):
                assert genotype[i]

    @pytest.mark.parametrize('genotype_size', [0, 2, 4, 6, 8])
    def test_get_item_in_position(self, genotype_size):
        genotype = self.get_instance(genes=range(genotype_size))

        for i, gene in enumerate(range(genotype_size)):
            assert gene == genotype[i]

    @pytest.mark.parametrize('genotype_size', [0, 2, 4, 6, 8])
    def test_try_to_delete_non_existent_item(self, genotype_size):
        genotype = self.get_instance(genes=range(genotype_size))

        for i in range(genotype_size, genotype_size + 4):
            with pytest.raises(ValueError):
                del genotype[i]

    @pytest.mark.parametrize('genes, index, expected_genes', [
        ([0, 1, 2, 3], 0, [1, 2, 3]),
        ([0, 1, 2, 3], 1, [0, 2, 3]),
        ([0, 1, 2, 3], 2, [0, 1, 3]),
        ([0, 1, 2, 3], 3, [0, 1, 2]),
    ])
    def test_correct_deleted_item(self, genes, index, expected_genes):
        genotype = self.get_instance(genes=genes)
        expected_genotype = self.get_instance(genes=expected_genes)

        del genotype[index]

        assert genotype == expected_genotype

    @pytest.mark.parametrize('genes, where, what, expected_genes', [
        ([0, 1, 2, 3], 0, 9, [9, 0, 1, 2, 3]),
        ([0, 1, 2, 3], 1, 9, [0, 9, 1, 2, 3]),
        ([0, 1, 2, 3], 2, 9, [0, 1, 9, 2, 3]),
        ([0, 1, 2, 3], 3, 9, [0, 1, 2, 9, 3]),
        ([0, 1, 2, 3], 4, 9, [0, 1, 2, 3, 9]),
        ([0, 1, 2, 3], 9, 9, [0, 1, 2, 3, 9]),
        ([0, 1, 2, 3], -1, 9, [0, 1, 2, 9, 3]),
    ])
    def test_correct_genotype_size(self, genes, where, what, expected_genes):
        genotype = self.get_instance(genes=genes)
        expected_genotype = self.get_instance(genes=expected_genes)

        genotype.insert(where, what)
        assert genotype == expected_genotype

    @pytest.mark.parametrize('genes, expected_size', [
        ([], 0), ([0], 1), ([0, 1], 2), ([0, 1, 2], 3), ([0, 1, 2, 3], 4),
    ])
    def test_correct_insert_item(self, genes, expected_size):
        genotype = self.get_instance(genes=genes)

        assert len(genotype) == expected_size

    @pytest.mark.parametrize('genes, expected_string', [
        (ListGenotype(genes=[]), ''),
        (ListGenotype(genes=[0]), '0'),
        (ListGenotype(genes=[0, 1]), '0,1'),
        (ListGenotype(genes=[0, 1, 2]), '0,1,2'),
        (ListGenotype(genes=[0, 1, 2, 3]), '0,1,2,3'),
    ])
    def test_correct_string_representation(self, genes, expected_string):
        genotype = self.get_instance(genes=genes)

        assert str(genotype) == expected_string
