#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import pytest
from unittest.mock import patch
from decodense.prop_tools import (
    _ao_val,
    _dip_nuc,
    _e_nuc,
    _get_nuc,
    _h_core,
    _make_rho,
    _make_rho_interm2,
    _mo_diag,
    _orb_atom_occ,
    _solvent,
    _trace,
    _vk_dft,
    _xc_ao_deriv,
    _xc_blocks,
    _xc_energies,
)
from pyscf import gto, scf, dft, qmmm
from pyscf.dft import numint


# mf fixtures
# dft water
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
# nuclear repulsion energy must match hand-derived value
def test_e_nuc():
    mol = MockMolNuc(
        charges=[2.0, 1.0, 3.0],
        coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
    )
    e_nuc = _e_nuc(mol)
    assert np.allclose(e_nuc, [2.5, 1.67082039, 2.17082039])


# per-atom nuclear repulsion energy must sum to pyscf's reference value
def test_e_nuc_pyscf(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    nuc_energy = _e_nuc(mol).sum()
    expected_nuc_energy = mol.energy_nuc()
    assert np.isclose(nuc_energy, expected_nuc_energy, atol=1e-10)


# _dip_nuc
# nuclear contribution to the molecular dipole moment must match hand-derived value
def test_dip_nuc():
    mol = MockMolNuc(charges=[1.5, 2.5], coords=[[2.0, 0.0, 1.0], [0.0, 3.0, -1.0]])
    gauge_origin = np.array([1.0, 1.0, 0.0])
    hdip_nuc = _dip_nuc(mol, gauge_origin)
    assert np.allclose(hdip_nuc, [[1.5, -1.5, 1.5], [-2.5, 5.0, -2.5]])


# nuclear contribution to the molecular dipole moment must respond correctly to a gauge-origin shift
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
# components of the core hamiltonian must sum to pyscf's reference value and must have correct signs
# the per-atom nuclear attraction energies must sum to tr(V_nuc D)
def test_h_core(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    rdm1 = mf_h2o_rhf.make_rdm1()
    kin, nuc, e_nuc = _h_core(mol, mf_h2o_rhf, rdm1)
    hcore = mf_h2o_rhf.get_hcore()
    assert np.allclose(hcore, kin + nuc, atol=1e-10)
    assert np.all(np.diag(kin) > 0.0)
    assert np.all(np.diag(nuc) < 0.0)
    assert e_nuc.shape == (mol.natm,)
    assert np.all(e_nuc < 0.0)
    assert np.isclose(e_nuc.sum(), np.einsum("ij,ij", nuc, rdm1), atol=1e-10)


# _get_nuc
# individual atomic potentials must sum to pyscf's total nuclear potential
# and per-atom energies tr(V_K D) must match an explicit calculation
def test_get_nuc_matches_pyscf():
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.0", basis="sto-3g", unit="bohr", verbose=0)
    nao = mol.nao_nr()
    rng = np.random.default_rng(0)
    # test should succeed with any random symmetric matrx
    rdm1 = rng.random((nao, nao))
    rdm1 = rdm1 + rdm1.T
    nuc, e_nuc = _get_nuc(mol, rdm1)
    assert np.allclose(nuc, mol.intor("int1e_nuc"), atol=1e-10)
    assert e_nuc.shape == (mol.natm,)
    for k in range(mol.natm):
        with mol.with_rinv_origin(mol.atom_coords()[k]):
            v_k = -mol.atom_charges()[k] * mol.intor("int1e_rinv")
        assert np.isclose(e_nuc[k], np.einsum("ij,ij", v_k, rdm1), atol=1e-10)


# _xc_ao_deriv
# each functional must give the correct xc type and ao derivative level
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


# unrecognised xc_type must raise UnboundLocalError
def test_xc_ao_deriv_unknown_type():
    with patch("pyscf.dft.libxc.xc_type", return_value="UNKNOWN"):
        with pytest.raises(NotImplementedError):
            _xc_ao_deriv("anything")


# _make_rho
# electron density and its intermediate must match hand-derived values for lda
def test_make_rho():
    ao_value = np.array([[1.0, 2.0, 0.0], [0.5, 1.0, 2.0]])
    rdm1 = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 1.0], [0.0, 1.0, 3.0]])
    c0, c1, rho = _make_rho(ao_value, rdm1, "LDA")
    assert np.allclose(c0, [[3.0, 2.5, 2.0], [1.5, 3.25, 7.0]], atol=1e-10)
    assert c1 is None
    assert np.allclose(rho, [8.0, 18.0], atol=1e-10)


# electron density must match pyscf's reference value for gga
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
# electron density must match hand-derived value
# per-atom electron density must sum up to the total electron density
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
# computed trace must match hand-derived value
def test_trace_identity():
    op = np.array([[1.0, 2.0], [3.0, 4.0]])
    rdm1 = np.eye(2)
    assert _trace(op, rdm1) == 5.0
    assert _trace(op, rdm1, scaling=0.5) == 2.5


# computed trace must match numpy's reference value for a symmetric rdm1
def test_trace_symmetric():
    op = np.array([[1.0, 2.0], [3.0, 4.0]])
    rdm1 = np.array([[2.0, 1.0], [1.0, 3.0]])
    assert _trace(op, rdm1) == np.trace(op @ rdm1)


# computed trace of a 3d operator must match hand-derived value
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


# _mo_diag
# c_m^T op c_m must match the diagonal of C^T op C, also for multi-component operators
def test_mo_diag():
    rng = np.random.default_rng(1)
    op = rng.random((5, 5))
    c = rng.random((5, 3))
    assert np.allclose(_mo_diag(op, c), np.diag(c.T @ op @ c))
    op3 = rng.random((3, 5, 5))
    ref = np.array([np.diag(c.T @ o @ c) for o in op3])
    assert np.allclose(_mo_diag(op3, c), ref)


# _orb_atom_occ
# occupation factors must match hand-derived values and sum to the occupations
def test_orb_atom_occ():
    mo_occ = (np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    spin_mos = (np.array([0, 1]), np.array([0]))
    weights = [np.array([[1.0, 3.0], [2.0, 2.0]]), np.array([[0.5, 0.5]])]
    F = _orb_atom_occ(mo_occ, weights, spin_mos, 2)
    assert np.allclose(F[0], [[0.25, 0.75], [0.5, 0.5]])
    assert np.allclose(F[1], [[0.5, 0.5]])
    assert np.allclose(F[0].sum(axis=1), mo_occ[0][spin_mos[0]])


# an emptry spin channel must give an empty (0, natm) array
def test_orb_atom_occ_empty_spin():
    mo_occ = (np.array([1.0]), np.array([0.0]))
    spin_mos = (np.array([0]), np.array([], dtype=int))
    weights = [np.array([[1.0, 1.0]]), np.zeros((0, 2))]
    F = _orb_atom_occ(mo_occ, weights, spin_mos, 2)
    assert F[1].shape == (0, 2)


# _ao_val
# ao values (and derivatives) must match pyscf and have the expected shape
@pytest.mark.parametrize("deriv,ncomp", [(0, None), (1, 4), (2, 10)])
def test_ao_val(mf_h2o_dft, deriv, ncomp):
    mol = mf_h2o_dft.mol
    coords = mf_h2o_dft.grids.coords[:100]
    ao = _ao_val(mol, coords, deriv)
    nao = mol.nao_nr()
    assert ao.shape == ((100, nao) if ncomp is None else (ncomp, 100, nao))
    assert np.allclose(ao, numint.eval_ao(mol, coords, deriv=deriv))


# _make_rho
# density, gradient, laplacian and tau must match pyscf for meta-gga
def test_make_rho_mgga(mf_h2o_dft):
    mol = mf_h2o_dft.mol
    ao = numint.eval_ao(mol, mf_h2o_dft.grids.coords, deriv=2)
    dm = mf_h2o_dft.make_rdm1()
    _, c1, rho = _make_rho(ao, dm, "MGGA")
    rho_ref = numint.eval_rho(mol, ao, dm, xctype="MGGA", with_lapl=True)
    assert c1.shape == (3,) + ao.shape[1:]
    assert np.allclose(rho, rho_ref, atol=1e-10)


# spin-resolved densities must match pyscf, and c0 must use the spin-summed rdm1
def test_make_rho_unrestricted(mf_h2o_dft):
    mol = mf_h2o_dft.mol
    ao = numint.eval_ao(mol, mf_h2o_dft.grids.coords, deriv=1)
    dm = mf_h2o_dft.make_rdm1()
    rng = np.random.default_rng(2)
    pert = rng.random(dm.shape) * 1e-2
    dm_a = 0.5 * dm + (pert + pert.T)
    dm_b = 0.5 * dm - (pert + pert.T)
    c0, c1, rho = _make_rho(ao, np.array([dm_a, dm_b]), "GGA")
    assert rho.shape == (2, 4, ao.shape[1])
    assert np.allclose(rho[0], numint.eval_rho(mol, ao, dm_a, xctype="GGA"), atol=1e-10)
    assert np.allclose(rho[1], numint.eval_rho(mol, ao, dm_b, xctype="GGA"), atol=1e-10)
    assert np.allclose(c0, ao[0] @ (dm_a + dm_b), atol=1e-10)
    assert c1 is None


# identical spin rdm1s must give the same density as the spin-summed rdm1
def test_make_rho_equal_spins(mf_h2o_dft):
    ao = numint.eval_ao(mf_h2o_dft.mol, mf_h2o_dft.grids.coords, deriv=1)
    dm = mf_h2o_dft.make_rdm1()
    rho = _make_rho(ao, np.array([dm, dm]) * 0.5, "GGA")[2]
    rho_ref = _make_rho(ao, dm, "GGA")[2]
    assert np.allclose(rho, rho_ref, atol=1e-10)


# _vk_dft
# exchange operator must be zero (pure), scaled (hybrid) or include the long-range
# part (range-separated); a precomputed unscaled vk must give the same result
@pytest.mark.parametrize("xc", ["pbe", "pbe0", "camb3lyp"])
def test_vk_dft(mf_h2o_dft, xc):
    mol = mf_h2o_dft.mol
    mf = dft.RKS(mol, xc=xc)
    dm = mf_h2o_dft.make_rdm1()
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(xc)
    vk_ref = hyb * mf.get_k(mol, dm)
    if abs(omega) > 1e-10:
        vk_ref += (alpha - hyb) * mf.get_k(mol, dm, omega=omega)
    assert np.allclose(_vk_dft(mol, mf, xc, dm, None), vk_ref, atol=1e-10)
    if abs(hyb) > 1e-10:
        vk_in = mf.get_k(mol, dm)
        assert np.allclose(_vk_dft(mol, mf, xc, dm, vk_in), vk_ref, atol=1e-10)


# _xc_blocks
# hartree-fock must not yield any xc blocks
def test_xc_blocks_hf(mf_h2o_dft):
    dm = mf_h2o_dft.make_rdm1()
    rdm1 = np.array([dm, dm]) * 0.5
    assert list(_xc_blocks(mf_h2o_dft.mol, mf_h2o_dft, rdm1, "HF", 0)) == []


# _xc_energies (incl. _xc_blocks and _xc_partition)
# atom- and orbital-wise xc energies must sum to pyscf's total xc energy
@pytest.mark.parametrize("part_method", ["mo", "ao", None])
def test_xc_energies_sum(mf_h2o_dft, part_method):
    mf = mf_h2o_dft
    mol = mf.mol
    occ = np.where(mf.mo_occ > 0)[0]
    c = mf.mo_coeff[:, occ]
    dm = mf.make_rdm1()
    rng = np.random.default_rng(3)
    w = rng.random((occ.size, mol.natm))
    e_xc, e_xc_nlc = _xc_energies(
        mol,
        mf,
        (c, c),
        (np.ones(occ.size), np.ones(occ.size)),
        np.array([dm, dm]) * 0.5,
        [w, w],
        part_method,
        mol.natm,
    )
    exc_ref = mf._numint.nr_rks(mol, mf.grids, mf.xc, dm)[1]
    assert np.isclose(np.sum(e_xc), exc_ref, atol=1e-8)
    assert e_xc_nlc is None


# _xc_energies, unrestricted (open-shell) reference
# atom- and orbital-wise xc energies must sum to pyscf's total xc energy
@pytest.mark.parametrize("xc", ["pbe0", "tpss"])
@pytest.mark.parametrize("part_method", ["mo", "ao", None])
def test_xc_energies_sum_uks(xc, part_method):
    mol = gto.M(
        verbose=0, output=None, basis="sto-3g", atom="geom/h2o.xyz", charge=1, spin=1
    )
    mf = dft.UKS(mol, xc=xc).run()
    dm = mf.make_rdm1()
    occ = [np.where(mf.mo_occ[i] > 0)[0] for i in range(2)]
    mo_coeff = tuple(mf.mo_coeff[i][:, occ[i]] for i in range(2))
    mo_occ = tuple(np.ones(o.size) for o in occ)
    rng = np.random.default_rng(4)
    weights = [rng.random((o.size, mol.natm)) for o in occ]
    e_xc, _ = _xc_energies(
        mol, mf, mo_coeff, mo_occ, dm, weights, part_method, mol.natm
    )
    exc_ref = mf._numint.nr_uks(mol, mf.grids, mf.xc, dm)[1]
    # orbital-wise: one array per spin, with different lengths (5 alpha, 4 beta)
    total = sum(np.sum(x) for x in e_xc) if isinstance(e_xc, list) else np.sum(e_xc)
    assert np.isclose(total, exc_ref, atol=1e-8)


# _solvent
# without solvent, no contributions must be returned
def test_solvent_none(mf_h2o_rhf):
    res = _solvent(mf_h2o_rhf.mol, mf_h2o_rhf, mf_h2o_rhf.make_rdm1())
    assert res == (None, None, None)


# pcm: electronic + nuclear parts must sum to pyscf's pcm energy
def test_solvent_pcm(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    mf = scf.RHF(mol).PCM().run()
    dm = mf.make_rdm1()
    pot_solv, nuc_solv, vdw_solv = _solvent(mol, mf, dm)
    epcm = mf.with_solvent._get_vind(dm)[0]
    assert vdw_solv is None
    assert np.isclose(
        np.einsum("ij,ij", pot_solv, dm) + nuc_solv.sum(), epcm, atol=1e-8
    )


# non-pcm solvent models must raise NotImplementedError
def test_solvent_not_pcm(mf_h2o_rhf):
    mf = scf.RHF(mf_h2o_rhf.mol).ddCOSMO()
    with pytest.raises(NotImplementedError):
        _solvent(mf.mol, mf, mf_h2o_rhf.make_rdm1())


# point charges: mm potential and nuclear interaction must match pyscf's qmmm
def test_solvent_point_charges(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    coords = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 1.0]])
    charges = np.array([0.5, -0.3])
    mf = qmmm.mm_charge(scf.RHF(mol), coords, charges)
    pot_solv, nuc_solv, vdw_solv = _solvent(mol, mf, mf_h2o_rhf.make_rdm1())
    h_mm = (
        mf.get_hcore()
        - mol.intor_symmetric("int1e_kin")
        - mol.intor_symmetric("int1e_nuc")
    )
    assert vdw_solv is None
    assert np.allclose(pot_solv, h_mm, atol=1e-10)
    assert np.isclose(nuc_solv.sum(), mf.energy_nuc() - mol.energy_nuc(), atol=1e-10)


# helper: decodense's _nlc_blocks needs the pyscf interface where nlc_coeff returns
# ((b, c), factor) pairs
def _has_new_nlc_interface(ni, xc):
    try:
        (b, c), fac = ni.nlc_coeff(xc)[0]
        return True
    except Exception:
        return False


# _xc_energies with nlc (vv10)
# atom- and orbital-wise xc and nlc energies must sum to pyscf's total values
@pytest.mark.parametrize("part_method", ["mo", "ao", None])
def test_xc_energies_nlc_sum(mf_h2o_dft, part_method):
    mol = mf_h2o_dft.mol
    mf = dft.RKS(mol, xc="b97m_v")
    if not _has_new_nlc_interface(mf._numint, mf.xc):
        pytest.skip("installed pyscf lacks the nlc interface used by _nlc_blocks")
    # any density works for the sum rule; use the pbe0 one of the fixture
    occ = np.where(mf_h2o_dft.mo_occ > 0)[0]
    c = mf_h2o_dft.mo_coeff[:, occ]
    dm = mf_h2o_dft.make_rdm1()
    rng = np.random.default_rng(5)
    w = rng.random((occ.size, mol.natm))
    e_xc, e_xc_nlc = _xc_energies(
        mol,
        mf,
        (c, c),
        (np.ones(occ.size), np.ones(occ.size)),
        np.array([dm, dm]) * 0.5,
        [w, w],
        part_method,
        mol.natm,
    )
    # reference values (grids were built inside _xc_energies)
    exc_ref = mf._numint.nr_rks(mol, mf.grids, mf.xc, dm)[1]
    nlc_ref = mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, mf.xc, dm)[1]
    assert e_xc_nlc is not None
    assert np.isclose(np.sum(e_xc), exc_ref, atol=1e-8)
    assert np.isclose(np.sum(e_xc_nlc), nlc_ref, atol=1e-8)
