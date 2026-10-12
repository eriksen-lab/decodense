#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import pytest
from decodense.tools import (
    dim,
    make_natorb,
    make_rdm1,
    mf_info,
    orbsym,
    res_add,
    res_sub,
)


# mock molecule
class MockMol:
    def __init__(self, natm, ao_labels, ovlp):
        self.natm = natm
        self._ao_labels = ao_labels
        self._ovlp = ovlp

    def ao_labels(self, fmt=None):
        return self._ao_labels

    def intor_symmetric(self, key):
        return self._ovlp


# mock mean-field object
class MockMf:
    def __init__(self, mo_coeff, mo_occ):
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ


# dim
# molecular dimensions must match hand-derived values
def test_dim():
    mo_occ = (np.array([1.0, 0.0, -0.5, 1.0]), np.array([0.0, 0.8, 0.0]))
    alpha, beta = dim(mo_occ)
    assert np.array_equal(alpha, [0, 2, 3])
    assert np.array_equal(beta, [1])


# mf_info
# mo coefficients and occupations must match hand-derived values for restricted closed-shell reference
def test_mf_info_restricted():
    mo_coeff = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    mo_occ = np.array([2.0, 2.0, 0.0])
    coeff, occ = mf_info(MockMf(mo_coeff, mo_occ))
    assert np.array_equal(coeff[0], [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
    assert np.array_equal(coeff[1], [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
    assert np.array_equal(occ[0], [1.0, 1.0])
    assert np.array_equal(occ[1], [1.0, 1.0])


# mo coefficients and occupations must match hand-derived values for restricted open-shell reference
def test_mf_info_restricted_open_shell():
    mo_coeff = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    mo_occ = np.array([2.0, 1.0, 0.0])
    coeff, occ = mf_info(MockMf(mo_coeff, mo_occ))
    assert np.array_equal(coeff[0], [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
    assert np.array_equal(coeff[1], [[1.0], [4.0], [7.0]])
    assert np.array_equal(occ[0], [1.0, 1.0])
    assert np.array_equal(occ[1], [1.0])


# mo coefficients and occupations must match hand-derived values for unrestricted reference
def test_mf_info_unrestricted():
    mo_coeff = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    mo_occ = np.array([[1.0, 1.0], [1.0, 0.0]])
    coeff, occ = mf_info(MockMf(mo_coeff, mo_occ))
    assert np.array_equal(coeff[0], [[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(coeff[1], [[5.0], [7.0]])
    assert np.array_equal(occ[0], [1.0, 1.0])
    assert np.array_equal(occ[1], [1.0])


# orbsym
# orbital symmetry labels must match pyscf's irrep labels for a symmetric molecule
def test_orbsym(mf_h2o_rhf):
    mo = mf_h2o_rhf.mo_coeff[:, mf_h2o_rhf.mo_occ > 0.0]
    assert list(orbsym(mf_h2o_rhf.mol, mo)) == ["A1", "A1", "B2", "A1", "B1"]


# orbsym must fall back to "A" for every orbital without symmetry info
@pytest.mark.parametrize(
    "mo_coeff,expected",
    [
        (np.zeros((3, 4)), ["A", "A", "A", "A"]),
        (np.zeros((2, 3, 2)), [["A", "A"], ["A", "A"]]),
        ((np.zeros((3, 2)), np.zeros((3, 1))), [["A", "A"], ["A"]]),
    ],
)
def test_orbsym_fallback(mo_coeff, expected):
    assert np.array_equal(orbsym(None, mo_coeff), np.array(expected, dtype=object))


# make_rdm1
# one-electron reduced density matrix must match hand-derived value
def test_make_rdm1():
    mo = np.array([[1.0, 0.0], [1.0, 2.0]])
    occup = np.array([2.0, 1.0])
    rdm = make_rdm1(mo, occup)
    assert np.array_equal(rdm, [[2.0, 2.0], [2.0, 6.0]])


# trace of the one-electron reduced density matrix with the overlap matrix must equal the total number of electrons
def test_make_rdm1_equal_electron_count(mf_h2o_rhf):
    mo = mf_h2o_rhf.mo_coeff[:, :5]
    occup = mf_h2o_rhf.mo_occ[:5]
    S = mf_h2o_rhf.get_ovlp()
    D = make_rdm1(mo, occup)
    assert np.isclose(np.trace(D @ S), 10.0)


# make_natorb
# natural orbitals and occupations must match hand-derived values
def test_make_natorb():
    mol = MockMol(natm=1, ao_labels=[], ovlp=np.eye(2))
    mo_coeff = np.eye(2)
    rdm1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    no, occ = make_natorb(mol, mo_coeff, rdm1)
    assert np.allclose(occ[0], [1.0])
    assert np.allclose(occ[1], [1.0])
    assert np.allclose(np.abs(no[0]), [[0.5**0.5], [0.5**0.5]])
    assert np.allclose(np.abs(no[1]), [[0.5**0.5], [0.5**0.5]])


# res_add
# adding two results with res_add must match hand-derived values
def test_res_add():
    res_a = {"el": 1.0, "Symm.": ["A1", "B2"], "Occup.": [2.0], "struct": [1.0, 2.0]}
    res_b = {"el": 0.5, "Symm.": ["A1", "A1"], "Occup.": [1.0], "struct": [3.0, 4.0]}
    res = res_add(res_a, res_b)
    assert res["el"] == 1.5
    # labels and occupations are kept side by side rather than combined
    assert res["Symm."] == (["A1", "B2"], ["A1", "A1"])
    assert res["Occup."] == ([2.0], [1.0])
    assert res["struct"] == [4.0, 6.0]


# res_sub
# subtracting two results with res_sub must match hand-derived values
def test_res_sub():
    res_a = {"el": 1.0, "struct": [1.0, 2.0]}
    res_b = {"el": 0.5, "struct": [3.0, 4.0]}
    res = res_sub(res_a, res_b)
    assert res["el"] == 0.5
    assert res["struct"] == [-2.0, -2.0]


# combining results must raise ValueError for mismatched keys
def test_res_combine_key_mismatch():
    with pytest.raises(ValueError):
        res_add({"el": 1.0}, {"struct": 1.0})
