#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import pytest
from unittest.mock import patch
from decodense.decodense import main
from decodense.orbitals import _population_mul, assign_rdm1s
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
    _vk_dft,
    _point_charges,
)
from decodense.tools import dim, make_rdm1, mf_info
from decodense.decomp import CompKeys, DecompCls, sanity_check
from decodense.results import atoms, orbs
from pyscf import gto, scf, dft
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
def mf_h2o_permute():
    mol = gto.M(
        verbose=0,
        output=None,
        basis="sto-3g",
        symmetry=True,
        atom="geom/h2o_permute.xyz",
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


@pytest.fixture
def mf_li_uhf():
    mol = gto.M(
        verbose=0, output=None, spin=1, basis="sto-3g", symmetry=True, atom="Li 0 0 0"
    )
    mf = scf.UHF(mol).run()
    return mf


@pytest.fixture
def mf_li_rohf():
    mol = gto.M(
        verbose=0, output=None, spin=1, basis="sto-3g", symmetry=True, atom="Li 0 0 0"
    )
    mf = scf.ROHF(mol).run()
    return mf


# open shell with more than one atom
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


# unit tests for functions in decodense.py


# main
# permutation invariance
@pytest.mark.parametrize("part_method", ["mo", "ao"])
def test_permute(mf_h2o_rhf, mf_h2o_permute, part_method):
    mol = mf_h2o_rhf.mol
    mo_coeff, mo_occ = mf_info(mf_h2o_rhf)
    decomp = DecompCls(pop_method="iao", part="atoms", part_method=part_method)
    res = main(mol, decomp, mf_h2o_rhf, mo_coeff, mo_occ)
    mol_perm = mf_h2o_permute.mol
    mo_coeff_perm, mo_occ_perm = mf_info(mf_h2o_permute)
    decomp_perm = DecompCls(pop_method="iao", part="atoms", part_method=part_method)
    res_perm = main(mol_perm, decomp_perm, mf_h2o_permute, mo_coeff_perm, mo_occ_perm)
    ## every contribution must match
    for key, val in res.res_dict.items():
        val_perm = res_perm.res_dict[key]
        assert np.isclose(val[0], val_perm[2])
        assert np.isclose(val[1], val_perm[0])
        assert np.isclose(val[2], val_perm[1])


# symmetry-equivalent atoms must match for every contribution
@pytest.mark.parametrize("part_method", ["mo", "ao"])
def test_symmetry_equivalent_atoms(mf_h2o_rhf, part_method):
    mol = mf_h2o_rhf.mol
    mo_coeff, mo_occ = mf_info(mf_h2o_rhf)
    decomp = DecompCls(pop_method="iao", part="atoms", part_method=part_method)
    res = main(mol, decomp, mf_h2o_rhf, mo_coeff, mo_occ)
    for val in res.res_dict.values():
        assert np.isclose(val[1], val[2])


# pyscf's mo_coeff/mo_occ must match the ones from  mf_info
def test_main_rohf_mo_occ(mf_li_rohf):
    mol = mf_li_rohf.mol
    mo_coeff, mo_occ = mf_info(mf_li_rohf)
    decomp = DecompCls(pop_method="iao", part="atoms")
    res = main(mol, decomp, mf_li_rohf, mo_coeff, mo_occ)
    decomp_raw = DecompCls(pop_method="iao", part="atoms")
    res_raw = main(mol, decomp_raw, mf_li_rohf, mf_li_rohf.mo_coeff, mf_li_rohf.mo_occ)
    assert np.allclose(res.el, res_raw.el)


# orbitals partitioning (non-aufbau occupation)
def test_main_orbitals_non_aufbau(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    mf_u = scf.UHF(mol).run()
    occ_a = mf_u.mo_occ[0].copy()
    occ_a[4] = 0.0
    occ_a[5] = 1.0
    occ_b = mf_u.mo_occ[1].copy()
    decomp = DecompCls(pop_method="iao", part="orbitals")
    res = main(mol, decomp, mf_u, mf_u.mo_coeff, (occ_a, occ_b))
    idx_a = np.where(occ_a > 0.0)[0]
    idx_b = np.where(occ_b > 0.0)[0]
    mo_coeff_sliced = (mf_u.mo_coeff[0][:, idx_a], mf_u.mo_coeff[1][:, idx_b])
    mo_occ_sliced = (np.ones(idx_a.size), np.ones(idx_b.size))
    decomp_sliced = DecompCls(pop_method="iao", part="orbitals")
    res_sliced = main(mol, decomp_sliced, mf_u, mo_coeff_sliced, mo_occ_sliced)
    assert res.el[0].size == 5
    assert np.allclose(res.el[0], res_sliced.el[0])
    assert np.allclose(res.el[1], res_sliced.el[1])


# nuc_att_glob and nuc_att_loc sum to the total
@pytest.mark.parametrize("part_method", ["mo", "ao"])
def test_nuc_att_sum(mf_h2o_rhf, part_method):
    mol = mf_h2o_rhf.mol
    mo_coeff, mo_occ = mf_info(mf_h2o_rhf)
    decomp = DecompCls(pop_method="iao", part="atoms", part_method=part_method)
    res = main(mol, decomp, mf_h2o_rhf, mo_coeff, mo_occ)
    rdm1 = mf_h2o_rhf.make_rdm1()
    total_nuc_att = np.sum(mol.intor("int1e_nuc") * rdm1)
    assert np.isclose((res.nuc_att_glob + res.nuc_att_loc).sum(), total_nuc_att)


# dipole must be gauge-origin independent for neutral molecule and match pyscf
@pytest.mark.parametrize("part_method", ["mo", "ao"])
def test_main_dipole(mf_h2o_rhf, part_method):
    mol = mf_h2o_rhf.mol
    mo_coeff, mo_occ = mf_info(mf_h2o_rhf)
    decomp1 = DecompCls(
        pop_method="iao", part="atoms", part_method=part_method, prop="dipole"
    )
    res1 = main(mol, decomp1, mf_h2o_rhf, mo_coeff, mo_occ)
    decomp2 = DecompCls(
        pop_method="iao",
        part="atoms",
        part_method=part_method,
        prop="dipole",
        gauge_origin=[1.0, 0.0, 0.0],
    )
    res2 = main(mol, decomp2, mf_h2o_rhf, mo_coeff, mo_occ)
    assert np.allclose(res1.tot.sum(axis=0), res2.tot.sum(axis=0))
    assert np.allclose(
        res1.tot.sum(axis=0), mf_h2o_rhf.dip_moment(unit="au", verbose=0)
    )


# unit tests for functions in decomp.py


# DecompCls
# checkingpart/part_method defaults
def test_decomp_cls_part_method_defaults():
    decomp_eda = DecompCls(part="eda")
    assert decomp_eda.part == "atoms"
    assert decomp_eda.part_method == "ao"
    decomp_atoms = DecompCls(part="atoms")
    assert decomp_atoms.part_method == "mo"
    decomp_orb = DecompCls(part="orbitals")
    assert decomp_orb.part_method is None


# sanity_check
# input validation
@pytest.mark.parametrize(
    "attr,bad_value,exception,message",
    [
        ("minao", "bad", ValueError, "invalid minao basis"),
        ("mo_basis", "bad", ValueError, "invalid MO basis"),
        ("pop_method", "bad", ValueError, "invalid population scheme"),
        ("mo_init", "bad", ValueError, "invalid MO start guess"),
        ("loc_exp", 3, ValueError, "invalid localization exponent"),
        ("part", "bad", ValueError, "invalid partitioning"),
        ("part_method", "bad", ValueError, "invalid partitioning method"),
        ("prop", "bad", ValueError, "invalid property"),
        ("unit", "bad", ValueError, "invalid unit"),
        ("verbose", -1, ValueError, "invalid verbosity"),
        ("verbose", 6, ValueError, "invalid verbosity"),
        ("ndo", "not_a_bool", TypeError, "invalid NDO argument"),
        ("write", "bad", ValueError, "invalid write format"),
        ("write", 123, TypeError, "invalid write format argument"),
        ("writename", 123, TypeError, "invalid write name argument"),
        ("verbose", "not_an_int", TypeError, "invalid verbosity"),
        ("unit", 123, TypeError, "invalid unit"),
    ],
)
def test_sanity_check_rejects_invalid_attr(attr, bad_value, exception, message):
    decomp = DecompCls(**{attr: bad_value})
    with pytest.raises(exception, match=message):
        sanity_check(None, None, decomp, None, None)


# invalid gauge_origin
@pytest.mark.parametrize(
    "bad_gauge_origin,exception",
    [
        ([0.0, 0.0], ValueError),  # too short
        ([0.0, 0.0, 0.0, 0.0], ValueError),  # too long
        (["a", "b", "c"], ValueError),  # right length, wrong element type
        ("not a list", TypeError),  # wrong container type entirely
    ],
)
def test_sanity_check_gauge_origin_invalid(bad_gauge_origin, exception):
    decomp = DecompCls(gauge_origin=bad_gauge_origin)
    with pytest.raises(exception, match="invalid gauge origin"):
        sanity_check(None, None, decomp, None, None)


# invalid mo_coeff
@pytest.mark.parametrize(
    "mo_coeff,exception,message",
    [
        ([1, 2, 3], TypeError, "invalid mo coefficients"),
        (np.zeros(5), ValueError, "invalid mo coefficients"),
        (
            (np.zeros((2, 2)),),
            TypeError,
            "invalid mo coefficients",
        ),
    ],
)
def test_sanity_check_rejects_invalid_mo_coeff(mo_coeff, exception, message):
    decomp = DecompCls()
    with pytest.raises(exception, match=message):
        sanity_check(None, None, decomp, mo_coeff, None)


# invalid mo_occ
@pytest.mark.parametrize(
    "mo_occ,exception,message",
    [
        ("not valid", TypeError, "invalid mo occupation"),
        (
            np.zeros((2, 2, 2)),
            ValueError,
            "invalid mo occupation",
        ),
        ((np.zeros(2),), TypeError, "invalid mo occupation"),
    ],
)
def test_sanity_check_rejects_invalid_mo_occ(mo_occ, exception, message):
    decomp = DecompCls()
    mo_coeff = np.zeros((2, 2))
    with pytest.raises(exception, match=message):
        sanity_check(None, None, decomp, mo_coeff, mo_occ)


# valid input
@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # fully default
        {"verbose": 0},
        {"verbose": 5},
        {"gauge_origin": [0.0, 0.0, 0.0]},
        {"part": "orbitals"},
        {"part": "atoms", "part_method": "ao"},
        {"part": "eda"},
    ],
)
def test_sanity_check_accepts_valid_input(kwargs):
    mo_coeff = np.zeros((2, 2))
    decomp = DecompCls(**kwargs)
    sanity_check(None, None, decomp, mo_coeff, None)


# unit tests for functions in orbitals.py


# assign_rdm1s
# mulliken weights vs pyscf reference
def test_assign_rdm1s_h2o(mf_h2o_rhf):
    mo_coeff, mo_occ = mf_info(mf_h2o_rhf)
    weights = assign_rdm1s(
        mf_h2o_rhf.mol, mf_h2o_rhf, mo_coeff, mo_occ, "MINAO", "mulliken", False, 0
    )
    assert len(weights) == 2
    assert np.array_equal(weights[1], weights[0])
    assert np.allclose(weights[0].sum(axis=1), 1.0)
    # per-atom values checked against pyscf's mulliken_pop
    pop_ref = mf_h2o_rhf.mol.atom_charges() - mf_h2o_rhf.mulliken_pop(verbose=0)[1]
    assert np.allclose(weights[0].sum(axis=0) + weights[1].sum(axis=0), pop_ref)


# iao weights
def test_assign_rdm1s_h2o_iao(mf_h2o_rhf):
    mo_coeff, mo_occ = mf_info(mf_h2o_rhf)
    weights = assign_rdm1s(
        mf_h2o_rhf.mol, mf_h2o_rhf, mo_coeff, mo_occ, "MINAO", "iao", False, 0
    )
    assert len(weights) == 2
    assert np.array_equal(weights[1], weights[0])
    assert np.allclose(weights[0].sum(axis=1), 1.0)


# partial charges from the iao population weights
def test_rdm1_charge_conservation(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    mo_coeff, mo_occ = mf_info(mf_h2o_rhf)
    weights = assign_rdm1s(mol, mf_h2o_rhf, mo_coeff, mo_occ, "MINAO", "iao", False, 0)
    population = np.sum(weights[0], axis=0) + np.sum(weights[1], axis=0)
    charge_atom = mol.atom_charges() - population
    total_charge = charge_atom.sum()
    assert np.isclose(total_charge, 0.0)
    assert charge_atom[0] < 0.0
    assert charge_atom[1] > 0.0
    assert charge_atom[2] > 0.0
    assert np.isclose(charge_atom[2], charge_atom[1])


# oh radical (alpha != beta): mulliken weights
def test_assign_rdm1s_oh(mf_oh):
    mo_coeff, mo_occ = mf_info(mf_oh)
    weights = assign_rdm1s(
        mf_oh.mol, mf_oh, mo_coeff, mo_occ, "MINAO", "mulliken", False, 0
    )
    assert len(weights) == 2
    assert np.allclose(weights[0].sum(axis=1), 1.0)
    assert np.allclose(weights[1].sum(axis=1), 1.0)
    assert weights[0].shape == (5, 2)
    assert weights[1].shape == (4, 2)
    assert not np.allclose(weights[0].sum(axis=0), weights[1].sum(axis=0))


# oh radical (alpha != beta): iao weights
def test_assign_rdm1s_oh_iao(mf_oh):
    mo_coeff, mo_occ = mf_info(mf_oh)
    weights = assign_rdm1s(mf_oh.mol, mf_oh, mo_coeff, mo_occ, "MINAO", "iao", False, 0)
    assert len(weights) == 2
    assert np.allclose(weights[0].sum(axis=1), 1.0)
    assert np.allclose(weights[1].sum(axis=1), 1.0)
    assert weights[0].shape == (5, 2)
    assert weights[1].shape == (4, 2)
    assert not np.allclose(weights[0].sum(axis=0), weights[1].sum(axis=0))


# _population_mul (checking against made up values)
def test_population_mul():
    mol = gto.M(verbose=0, basis="sto-3g", symmetry=True, atom="geom/h2o.xyz")
    pop = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
            [7.0, 70.0],
        ]
    )
    populations = _population_mul(mol.natm, mol.ao_labels(fmt=None), pop)
    assert np.array_equal(populations, [[15.0, 6.0, 7.0], [150.0, 60.0, 70.0]])


# unit tests for functions in properties.py


# _e_nuc
# check against hand-derived value for nuclear repulsion
def test_e_nuc_h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.0", basis="sto-3g", unit="bohr", verbose=0)
    h2nuc = _e_nuc(mol)
    assert np.array_equal(h2nuc, [0.5, 0.5])


# check against pyscf value for nuclear repulsion
def test_e_nuc(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    nuc_energy = _e_nuc(mol).sum()
    expected_nuc_energy = mol.energy_nuc()
    assert np.isclose(nuc_energy, expected_nuc_energy, atol=1e-10)


# _dip_nuc
# check against hand-derived value for nuclear contribution to the molecular dipole moment
def test_dip_nuc():
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", unit="bohr", spin=1, verbose=0)
    gauge_origin = np.array([0.0, 0.0, 1.0])
    hdip_nuc = _dip_nuc(mol, gauge_origin)
    assert np.array_equal(hdip_nuc, np.array([[0, 0, -1]]))


# check gauge-origin shift formula
def test_dip_nuc_gauge_origin_shift(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
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
# check point charges against hand-derived nuc_solv and pyscf reference for mm_pot
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


# _xc_ao_deriv
#  xc type and ao derivative level needed
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
# check against pyscf value for rho (lda)
def test_make_rho(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    grids = dft.Grids(mol)
    grids.build()
    ao_value = numint.eval_ao(mol, grids.coords, deriv=0)
    rdm1 = mf_h2o_rhf.make_rdm1()
    c0, c1, rho = _make_rho(ao_value, rdm1, "LDA")
    rho_ref = numint.eval_rho(mol, ao_value, rdm1, xctype="LDA")
    assert np.allclose(rho, rho_ref, atol=1e-10)


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
def test_make_rho_atom_slicing(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    grids = dft.Grids(mol)
    grids.build()
    ao_value = numint.eval_ao(mol, grids.coords, deriv=0)
    rdm1 = mf_h2o_rhf.make_rdm1()
    c0, c1, rho_total = _make_rho(ao_value, rdm1, "LDA")
    ao_labels = mol.ao_labels(fmt=None)
    rho_sum = np.zeros_like(rho_total)
    for atom_idx in range(mol.natm):
        select = np.where([label[0] == atom_idx for label in ao_labels])[0]
        rho_atom = _make_rho_interm2(c0[:, select], None, ao_value[:, select], "LDA")
        rho_sum += rho_atom
    assert np.allclose(rho_sum, rho_total, atol=1e-10)


# _trace
# check against hand-derived value
def test_trace_identity():
    op = np.array([[1.0, 2.0], [3.0, 4.0]])
    rdm1 = np.eye(2)
    assert _trace(op, rdm1) == 5.0


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


# unit tests for functions in tools.py


# dim
# check against hand-derived values for molecular dimensions
def test_dim():
    mo_occ = (np.array([1.0, 0.0, -0.5, 1.0]), np.array([0.0, 0.8, 0.0]))
    alpha, beta = dim(mo_occ)
    assert np.array_equal(alpha, [0, 2, 3])
    assert np.array_equal(beta, [1])


# mf_info
# check against pyscf's mo_coeff and mo_occ for rhf
def test_mf_info_h2o(mf_h2o_rhf):
    mo_coeff, mo_occ = mf_info(mf_h2o_rhf)
    assert np.array_equal(mo_occ[0], np.ones(5))
    assert np.array_equal(mo_occ[1], np.ones(5))
    assert np.array_equal(mo_coeff[0], mf_h2o_rhf.mo_coeff[:, mf_h2o_rhf.mo_occ > 0.0])
    assert np.array_equal(mo_coeff[1], mf_h2o_rhf.mo_coeff[:, mf_h2o_rhf.mo_occ > 1.0])


# check against pyscf's mo_coeff and mo_occ for uhf
def test_mf_info_li(mf_li_uhf):
    mo_coeff, mo_occ = mf_info(mf_li_uhf)
    assert np.array_equal(mo_occ[0], np.ones(2))
    assert np.array_equal(mo_occ[1], np.ones(1))
    assert np.array_equal(
        mo_coeff[0], mf_li_uhf.mo_coeff[0][:, mf_li_uhf.mo_occ[0] > 0.0]
    )
    assert np.array_equal(
        mo_coeff[1], mf_li_uhf.mo_coeff[1][:, mf_li_uhf.mo_occ[1] > 0.0]
    )


# check against pyscf's mo_coeff and mo_occ for rohf
def test_mf_info_li_rohf(mf_li_rohf):
    mo_coeff, mo_occ = mf_info(mf_li_rohf)
    assert np.array_equal(mo_occ[0], np.ones(2))
    assert np.array_equal(mo_occ[1], np.ones(1))
    assert np.array_equal(mo_coeff[0], mf_li_rohf.mo_coeff[:, mf_li_rohf.mo_occ > 0.0])
    assert np.array_equal(mo_coeff[1], mf_li_rohf.mo_coeff[:, mf_li_rohf.mo_occ > 1.0])


# make_rdm1
# check against hand-derived value for rdm1
def test_make_rdm1():
    mo = np.array([[1.0, 0.0], [1.0, 2.0]])
    occup = np.array([2.0, 1.0])
    rdm = make_rdm1(mo, occup)
    assert np.array_equal(rdm, [[2.0, 2.0], [2.0, 6.0]])


# check against total number of electrons
def test_make_rdm1_equal_electron_count(mf_h2o_rhf):
    mo = mf_h2o_rhf.mo_coeff[:, :5]
    occup = mf_h2o_rhf.mo_occ[:5]
    S = mf_h2o_rhf.get_ovlp()
    D = make_rdm1(mo, occup)
    assert np.isclose(np.trace(D @ S), 10.0)


# unit tests for functions in results.py


# to_dataframe
# check row counts, string output, and unit conversion are correct
def test_to_dataframe(mf_h2o_rhf):
    mol = mf_h2o_rhf.mol
    mo_coeff, mo_occ = mf_info(mf_h2o_rhf)
    decomp1 = DecompCls(pop_method="iao", part="atoms")
    res1 = main(mol, decomp1, mf_h2o_rhf, mo_coeff, mo_occ)
    decomp2 = DecompCls(pop_method="iao", part="orbitals")
    res2 = main(mol, decomp2, mf_h2o_rhf, mo_coeff, mo_occ)
    decomp3 = DecompCls(pop_method="iao", part="atoms", unit="ev")
    res3 = main(mol, decomp3, mf_h2o_rhf, mo_coeff, mo_occ)
    assert len(res1.to_dataframe()) == 3
    assert len(res2.to_dataframe()) == 10
    assert "O0" in str(res1)
    assert np.allclose(res1.to_dataframe()[CompKeys.tot], res1.tot)
    assert np.allclose(res3.to_dataframe()[CompKeys.tot], res1.tot * 27.211386245988)


# atoms
# check dataframe structure and unit scaling for energy
def test_atoms():
    mol = gto.M(
        atom="H 0 0 0; O 0 0 1.0", basis="sto-3g", spin=1, unit="bohr", verbose=0
    )
    res = {
        CompKeys.el: np.array([1.0, 4.0]),
        CompKeys.kin: np.array([2.0, 8.0]),
    }
    df = atoms(mol, res, "au")
    assert len(df) == 2
    assert list(df.index) == ["H0", "O1"]
    assert np.allclose(df[CompKeys.el], [1.0, 4.0])
    assert np.allclose(df[CompKeys.kin], [2.0, 8.0])
    df_ev = atoms(mol, res, "ev")
    assert np.allclose(df_ev[CompKeys.el], np.array([1.0, 4.0]) * 27.211386245988)


# check dataframe structure and column naming for dipole
def test_atoms_dipole():
    mol = gto.M(
        atom="H 0 0 0; O 0 0 1.0", basis="sto-3g", spin=1, unit="bohr", verbose=0
    )
    res = {CompKeys.el: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    df = atoms(mol, res, "au")
    assert list(df.columns) == ["Elect. (x)", "Elect. (y)", "Elect. (z)"]
    assert np.allclose(df.values, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


# orbs
# check ndo pairing order and unit scaling
def test_orbs_ndo():
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, unit="bohr", verbose=0)
    res = {
        CompKeys.el: [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([])],
        CompKeys.mo_occ: (np.array([-0.8, -0.3, 0.0, 0.3, 0.8]), np.array([])),
        CompKeys.orbsym: (
            np.array(list("abcde"), dtype=object),
            np.array([], dtype=object),
        ),
    }
    df = orbs(mol, res, "au", ndo=True)
    assert np.allclose(df[CompKeys.el], [1.0, 5.0, 2.0, 4.0, 3.0])
    assert np.allclose(df[CompKeys.mo_occ], [-0.8, 0.8, -0.3, 0.3, 0.0])
    df_ev = orbs(mol, res, "ev", ndo=True)
    assert np.allclose(
        df_ev[CompKeys.el], np.array([1.0, 5.0, 2.0, 4.0, 3.0]) * 27.211386245988
    )


# no unit tests for functions in pbctools.py yet
