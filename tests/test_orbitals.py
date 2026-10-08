#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import pytest
from decodense.orbitals import _population_becke, _population_mul, assign_rdm1s
from pyscf import gto, scf


# mf fixtures
# uhf OH
@pytest.fixture
def mf_oh():
    mol = gto.M(
        verbose=0,
        output=None,
        spin=1,
        basis="sto-3g",
        atom="O 0 0 0; H 0 0 1.8",
        unit="bohr",
    )
    mf = scf.UHF(mol).run()
    return mf


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


# assign_rdm1s
# mulliken weights must match hand-derived values
def test_assign_rdm1s_mulliken():
    ovlp = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    ao_labels = [(0, "", "", ""), (0, "", "", ""), (1, "", "", "")]
    mol = MockMol(natm=2, ao_labels=ao_labels, ovlp=ovlp)
    mo = np.array([[2.0, 1.0], [3.0, 0.0], [1.0, 4.0]])
    mo_occ = (np.array([1.5, 0.7]), np.array([1.5, 0.7]))
    weights = assign_rdm1s(mol, None, (mo, mo), mo_occ, "MINAO", "mulliken", False, 0)
    assert np.allclose(weights[0], [[26.1, 2.7], [0.98, 11.48]])
    assert np.array_equal(weights[1], weights[0])


# mulliken weights must match pyscf for restricted reference
def test_assign_rdm1s_h2o(mf_h2o_rhf):
    mo = mf_h2o_rhf.mo_coeff[:, mf_h2o_rhf.mo_occ > 0.0]
    mo_occ = (np.ones(mo.shape[1]), np.ones(mo.shape[1]))
    weights = assign_rdm1s(
        mf_h2o_rhf.mol, mf_h2o_rhf, (mo, mo), mo_occ, "MINAO", "mulliken", False, 0
    )
    assert np.array_equal(weights[1], weights[0])
    assert np.allclose(weights[0].sum(axis=1), 1.0)
    pop_ref = mf_h2o_rhf.mol.atom_charges() - mf_h2o_rhf.mulliken_pop(verbose=0)[1]
    assert np.allclose(weights[0].sum(axis=0) + weights[1].sum(axis=0), pop_ref)


# iao weights and the partial charges derived from them must be sensible for restricted reference
def test_assign_rdm1s_h2o_iao(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    mo = mf_h2o_rhf.mo_coeff[:, mf_h2o_rhf.mo_occ > 0.0]
    mo_occ = (np.ones(mo.shape[1]), np.ones(mo.shape[1]))
    weights = assign_rdm1s(mol, mf_h2o_rhf, (mo, mo), mo_occ, "MINAO", "iao", False, 0)
    assert np.array_equal(weights[1], weights[0])
    assert np.allclose(weights[0].sum(axis=1), 1.0)
    population = np.sum(weights[0], axis=0) + np.sum(weights[1], axis=0)
    charge_atom = mol.atom_charges() - population
    assert np.isclose(charge_atom.sum(), 0.0)
    assert charge_atom[0] < 0.0
    assert charge_atom[1] > 0.0
    assert charge_atom[2] > 0.0
    assert np.isclose(charge_atom[2], charge_atom[1])


# mulliken weights must match pyscf for unrestricted reference
def test_assign_rdm1s_oh(mf_oh):
    alpha = np.where(mf_oh.mo_occ[0] > 0.0)[0]
    beta = np.where(mf_oh.mo_occ[1] > 0.0)[0]
    mo_coeff = (mf_oh.mo_coeff[0][:, alpha], mf_oh.mo_coeff[1][:, beta])
    mo_occ = (np.ones(alpha.size), np.ones(beta.size))
    weights = assign_rdm1s(
        mf_oh.mol, mf_oh, mo_coeff, mo_occ, "MINAO", "mulliken", False, 0
    )
    assert np.allclose(weights[0].sum(axis=1), 1.0)
    assert np.allclose(weights[1].sum(axis=1), 1.0)
    assert weights[0].shape == (5, 2)
    assert weights[1].shape == (4, 2)
    # per-atom values checked against pyscf values
    ao_labels = mf_oh.mol.ao_labels(fmt=None)
    alpha_pop, beta_pop = mf_oh.mulliken_pop(verbose=0)[0]
    alpha_ref = np.zeros(mf_oh.mol.natm)
    beta_ref = np.zeros(mf_oh.mol.natm)
    for p, label in zip(alpha_pop, ao_labels):
        alpha_ref[label[0]] += p
    for p, label in zip(beta_pop, ao_labels):
        beta_ref[label[0]] += p
    assert np.allclose(weights[0].sum(axis=0), alpha_ref)
    assert np.allclose(weights[1].sum(axis=0), beta_ref)


# iao weights and the partial charges derived from them must be sensible for unrestricted reference
def test_assign_rdm1s_oh_iao(mf_oh):
    alpha = np.where(mf_oh.mo_occ[0] > 0.0)[0]
    beta = np.where(mf_oh.mo_occ[1] > 0.0)[0]
    mo_coeff = (mf_oh.mo_coeff[0][:, alpha], mf_oh.mo_coeff[1][:, beta])
    mo_occ = (np.ones(alpha.size), np.ones(beta.size))
    weights = assign_rdm1s(mf_oh.mol, mf_oh, mo_coeff, mo_occ, "MINAO", "iao", False, 0)
    assert np.allclose(weights[0].sum(axis=1), 1.0)
    assert np.allclose(weights[1].sum(axis=1), 1.0)
    assert weights[0].shape == (5, 2)
    assert weights[1].shape == (4, 2)
    assert not np.allclose(weights[0].sum(axis=0), weights[1].sum(axis=0))


# lowdin, meta_lowdin and becke weights must each sum to the electron count
@pytest.mark.parametrize(
    "pop_method,atol",
    [
        ("lowdin", 1e-10),
        ("meta_lowdin", 1e-10),
        ("becke", 1e-6),
    ],
)
def test_assign_rdm1s_h2o_pop_methods(mf_h2o_rhf, pop_method, atol):
    mol = mf_h2o_rhf.mol
    mo = mf_h2o_rhf.mo_coeff[:, mf_h2o_rhf.mo_occ > 0.0]
    mo_occ = (np.ones(mo.shape[1]), np.ones(mo.shape[1]))
    weights = assign_rdm1s(
        mol, mf_h2o_rhf, (mo, mo), mo_occ, "MINAO", pop_method, False, 0
    )
    assert np.allclose(weights[0].sum(axis=1), 1.0, atol=atol)
    population = weights[0].sum(axis=0) + weights[1].sum(axis=0)
    assert np.isclose(population.sum(), 10.0, atol=atol)


# _population_mul
# Mulliken populations must match hand-derived values
def test_population_mul():
    natm = 4
    ao_labels = (
        [(0, "", "", "")] * 2
        + [(1, "", "", "")] * 3
        + [(2, "", "", "")]
        + [(3, "", "", "")] * 2
    )
    pop = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
            [7.0, 70.0],
            [8.0, 80.0],
        ]
    )
    populations = _population_mul(natm, ao_labels, pop)
    assert np.array_equal(
        populations, [[3.0, 12.0, 6.0, 15.0], [30.0, 120.0, 60.0, 150.0]]
    )


# _population_becke
# Becke populations must match hand-derived values
def test_population_becke():
    orbs = np.array([[1.0, 2.0], [3.0, 1.0]])
    charge_matrix = np.array([[[1.0, 0.5], [0.5, 2.0]], [[2.0, 0.0], [0.0, 1.0]]])
    populations = _population_becke(charge_matrix, orbs)
    assert np.array_equal(populations, [[22.0, 11.0], [8.0, 9.0]])
