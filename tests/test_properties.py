#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import pytest
from unittest.mock import patch
from decodense.properties import (
    _dip_nuc,
    _get_nuc,
    _h_core,
    _make_rho,
    _make_rho_interm2,
    _trace,
    _e_xc,
    _xc_ao_deriv,
    _e_nuc,
    _point_charges,
    _pcm,
)
from pyscf import gto, scf, dft, solvent
from pyscf.dft import numint


# mf fixtures
@pytest.fixture
def mf_h2o_rhf():
    mol = gto.M(
        verbose=0, output=None, basis="sto-3g", symmetry=True, atom="geom/h2o.xyz"
    )
    mf = scf.RHF(mol).run()
    return mf


@pytest.fixture
def mf_h2o_dft():
    mol = gto.M(
        verbose=0, output=None, basis="sto-3g", symmetry=True, atom="geom/h2o.xyz"
    )
    mf = dft.RKS(mol, xc="pbe0").run()
    return mf


# mock molecule
class MockMolNuc:
    def __init__(self, charges, coords):
        self._charges = np.array(charges)
        self._coords = np.array(coords)

    def atom_charges(self):
        return self._charges

    def atom_coords(self):
        return self._coords


# _e_nuc
# check against hand-derived value
def test_e_nuc():
    mol = MockMolNuc(
        charges=[2.0, 1.0, 3.0],
        coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
    )
    e_nuc = _e_nuc(mol)
    assert np.allclose(e_nuc, [2.5, 1.67082039, 2.17082039])


# check against pyscf value for nuclear repulsion
def test_e_nuc_pyscf(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    nuc_energy = _e_nuc(mol).sum()
    expected_nuc_energy = mol.energy_nuc()
    assert np.isclose(nuc_energy, expected_nuc_energy, atol=1e-10)


# _dip_nuc
# check against hand-derived value for the nuclear dipole contribution
def test_dip_nuc():
    mol = MockMolNuc(charges=[1.5, 2.5], coords=[[2.0, 0.0, 1.0], [0.0, 3.0, -1.0]])
    gauge_origin = np.array([1.0, 1.0, 0.0])
    hdip_nuc = _dip_nuc(mol, gauge_origin)
    assert np.allclose(hdip_nuc, [[1.5, -1.5, 1.5], [-2.5, 5.0, -2.5]])


# check gauge-origin shift formula
def test_dip_nuc_gauge_origin_shift():
    mol = MockMolNuc(
        charges=[2.0, 1.0, 3.0],
        coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
    )
    gauge_origin1 = np.array([0.0, 0.0, 1.0])
    dip_nuc1 = _dip_nuc(mol, gauge_origin1)
    gauge_origin2 = np.array([0.0, 1.0, 2.0])
    dip_nuc2 = _dip_nuc(mol, gauge_origin2)
    dip_nuc1 = dip_nuc1.sum(axis=0)
    dip_nuc2 = dip_nuc2.sum(axis=0)
    total_charge = mol.atom_charges().sum()
    assert np.allclose(
        dip_nuc1 - dip_nuc2,
        -1 * total_charge * (gauge_origin1 - gauge_origin2),
        atol=1e-10,
    )


# _h_core
# check against pyscf value for one-electron Hamiltonian
def test_h_core(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    kin, nuc, sub_nuc = _h_core(mol, mf_h2o_rhf)
    hcore = mf_h2o_rhf.get_hcore()
    assert np.allclose(hcore, kin + nuc, atol=1e-10)
    assert np.all(np.diag(kin) > 0.0)
    assert np.all(np.diag(nuc) < 0.0)


# _get_nuc
# check individual atomic potentials sum to pyscf's total nuclear potential
def test_get_nuc_matches_pyscf():
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.0", basis="sto-3g", unit="bohr", verbose=0)
    sub_nuc = _get_nuc(mol)
    total = sub_nuc.sum(axis=0)
    assert np.allclose(total, mol.intor("int1e_nuc"), atol=1e-10)
    assert sub_nuc.shape == (mol.natm, mol.nao_nr(), mol.nao_nr())


# _point_charges
# check against hand-derived nuc_solv and pyscf value for mm_pot
def test_point_charges():
    mol = gto.M(atom="Li 0 0 0; H 0 0 1", basis="sto-3g", unit="bohr", verbose=0)
    mm_mol = gto.M(
        atom="O 0 0 3; H 0 0 5", basis="sto-3g", spin=1, unit="bohr", verbose=0
    )
    mm_pot, nuc_solv = _point_charges(mol, mm_mol)
    assert nuc_solv.shape == (mol.natm,)
    assert np.allclose(nuc_solv, [3 * (8 / 3 + 1 / 5), 1 * (8 / 2 + 1 / 4)])
    ref = np.zeros_like(mm_pot)
    for coord, charge in zip(mm_mol.atom_coords(), mm_mol.atom_charges()):
        with mol.with_rinv_origin(coord):
            ref += -1.0 * mol.intor("int1e_rinv") * charge
    assert np.allclose(mm_pot, ref, atol=1e-10)


# _pcm
# check against pyscf value for the total pcm solvation energy
def test_pcm():
    mol = gto.M(
        verbose=0, output=None, basis="sto-3g", symmetry=True, atom="geom/h2o.xyz"
    )
    mf = scf.RHF(mol)
    mf = solvent.PCM(mf)
    mf.with_solvent.eps = 78.3553
    mf.conv_tol = 1.0e-10
    mf.kernel()
    rdm1 = mf.make_rdm1()
    vmat_e, nuc_solv_pcm = _pcm(mol, rdm1, mf.with_solvent)
    decodense_total = np.einsum("ij,ij->", vmat_e, rdm1) + nuc_solv_pcm.sum()
    assert np.isclose(decodense_total, mf.with_solvent.e, atol=1e-10)
    assert np.allclose(vmat_e, vmat_e.T)
    assert nuc_solv_pcm.shape == (mol.natm,)


# _xc_ao_deriv
# check xc type and ao derivative level needed
@pytest.mark.parametrize(
    "xc_func,expected",
    [
        ("lda,vwn", ("LDA", 0)),
        ("hf", ("HF", 0)),
        ("pbe", ("GGA", 1)),
        ("tpss", ("MGGA", 2)),
    ],
)
def test_xc_ao_deriv(xc_func, expected):
    assert _xc_ao_deriv(xc_func) == expected


# unrecognised xc_type raises unboundlocalerror
def test_xc_ao_deriv_unknown_type():
    with patch("pyscf.dft.libxc.xc_type", return_value="UNKNOWN"):
        with pytest.raises(UnboundLocalError):
            _xc_ao_deriv("anything")


# _make_rho
# check against hand-derived values for rho (lda)
def test_make_rho():
    # 2 grid points, 3 aos
    ao_value = np.array([[1.0, 2.0, 0.0], [0.5, 1.0, 2.0]])
    rdm1 = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 1.0], [0.0, 1.0, 3.0]])
    c0, c1, rho = _make_rho(ao_value, rdm1, "LDA")
    assert np.allclose(c0, [[3.0, 2.5, 2.0], [1.5, 3.25, 7.0]], atol=1e-10)
    assert c1 is None
    assert np.allclose(rho, [8.0, 18.0], atol=1e-10)


# check against pyscf value for rho (gga)
def test_make_rho_gga(mf_h2o_dft):
    mol = mf_h2o_dft.mol
    grids = dft.Grids(mol)
    grids.build()
    ao_value = numint.eval_ao(mol, grids.coords, deriv=1)
    rdm1 = mf_h2o_dft.make_rdm1()
    c0, c1, rho = _make_rho(ao_value, rdm1, "GGA")
    rho_ref = numint.eval_rho(mol, ao_value, rdm1, xctype="GGA")
    assert np.allclose(rho, rho_ref, atol=1e-10)


# _make_rho_interm2
# check if rho per atom sums back to total rho
def test_make_rho_atom_slicing():
    ao_value = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])
    c0 = np.array([[2.0, 1.0, 0.5, 1.5], [1.0, 2.0, 1.0, 0.5]])
    ao_atom_idx = np.array([0, 0, 1, 1])
    rho_total = _make_rho_interm2(c0, None, ao_value, "LDA")
    assert np.allclose(rho_total, [11.5, 7.75], atol=1e-10)
    rho_sum = np.zeros_like(rho_total)
    for atom_idx in range(2):
        select = np.where(ao_atom_idx == atom_idx)[0]
        rho_sum += _make_rho_interm2(c0[:, select], None, ao_value[:, select], "LDA")
    assert np.allclose(rho_sum, rho_total, atol=1e-10)


# _trace
# check against hand-derived value
def test_trace_identity():
    op = np.array([[1.0, 2.0], [3.0, 4.0]])
    rdm1 = np.eye(2)
    assert _trace(op, rdm1) == 5.0
    assert _trace(op, rdm1, scaling=0.5) == 2.5


# check against numpy-derived value (symmetric rdm1)
def test_trace_symmetric():
    op = np.array([[1.0, 2.0], [3.0, 4.0]])
    rdm1 = np.array([[2.0, 1.0], [1.0, 3.0]])
    assert _trace(op, rdm1) == np.trace(op @ rdm1)


# check against hand-derived value (3d matrix)
def test_trace_3d():
    op = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    rdm1 = np.array([[2.0, 1.0], [1.0, 3.0]])
    assert np.array_equal(_trace(op, rdm1), [19.0, 2.0, 5.0])
    assert np.array_equal(_trace(op, rdm1, scaling=0.5), [9.5, 1.0, 2.5])


# _e_xc
# check against hand-derived value (1d matrix)
def test_e_xc_1d():
    eps_xc = np.array([1.0, 2.0, 3.0])
    grid_weights = np.array([2.0, 1.0, 0.5])
    rho = np.array([1.0, 2.0, 3.0])
    assert _e_xc(eps_xc, grid_weights, rho) == 10.5


# check against hand-derived value (2d matrix)
def test_e_xc_2d():
    eps_xc = np.array([1.0, 1.0, 1.0])
    grid_weights = np.array([1.0, 1.0, 1.0])
    rho = np.array([[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]])
    assert _e_xc(eps_xc, grid_weights, rho) == 6.0
