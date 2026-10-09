#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import pytest
from decodense.decodense import main
from decodense.tools import mf_info
from decodense.decomp import DecompCls
from pyscf import gto, scf


# mf fixtures
# rhf water with permuted order of atoms
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


# rohf lithium
@pytest.fixture
def mf_li_rohf():
    mol = gto.M(
        verbose=0, output=None, spin=1, basis="sto-3g", symmetry=True, atom="Li 0 0 0"
    )
    mf = scf.ROHF(mol).run()
    return mf


# main
# atomic partitioning must be invariant to permutation of atoms
@pytest.mark.parametrize("part_method", ["mo", "ao"])
def test_permute(mf_h2o_rhf, mf_h2o_permute, part_method):
    mo = mf_h2o_rhf.mo_coeff[:, mf_h2o_rhf.mo_occ > 0.0]
    mo_occ = (np.ones(mo.shape[1]), np.ones(mo.shape[1]))
    decomp = DecompCls(pop_method="iao", part="atoms", part_method=part_method)
    res = main(mf_h2o_rhf.mol, decomp, mf_h2o_rhf, (mo, mo), mo_occ)
    mo_perm = mf_h2o_permute.mo_coeff[:, mf_h2o_permute.mo_occ > 0.0]
    mo_occ_perm = (np.ones(mo_perm.shape[1]), np.ones(mo_perm.shape[1]))
    decomp_perm = DecompCls(pop_method="iao", part="atoms", part_method=part_method)
    res_perm = main(
        mf_h2o_permute.mol, decomp_perm, mf_h2o_permute, (mo_perm, mo_perm), mo_occ_perm
    )
    # every contribution must match
    for key, val in res.res_dict.items():
        val_perm = res_perm.res_dict[key]
        assert np.allclose(val[0], val_perm[2], atol=1e-10)
        assert np.allclose(val[1], val_perm[0], atol=1e-10)
        assert np.allclose(val[2], val_perm[1], atol=1e-10)


# every contribution must match for symmetry-equivalent atoms
@pytest.mark.parametrize("part_method", ["mo", "ao"])
def test_symmetry_equivalent_atoms(mf_h2o_rhf, part_method):
    mol = mf_h2o_rhf.mol
    mo = mf_h2o_rhf.mo_coeff[:, mf_h2o_rhf.mo_occ > 0.0]
    mo_occ = (np.ones(mo.shape[1]), np.ones(mo.shape[1]))
    decomp = DecompCls(pop_method="iao", part="atoms", part_method=part_method)
    res = main(mol, decomp, mf_h2o_rhf, (mo, mo), mo_occ)
    for val in res.res_dict.values():
        assert np.allclose(val[1], val[2], atol=1e-10)


# pyscf arrays and mf_info's output must give the same result
def test_main_rohf_mo_occ(mf_li_rohf):
    mol = mf_li_rohf.mol
    mo_coeff, mo_occ = mf_info(mf_li_rohf)
    decomp = DecompCls(pop_method="iao", part="atoms")
    res = main(mol, decomp, mf_li_rohf, mo_coeff, mo_occ)
    decomp_raw = DecompCls(pop_method="iao", part="atoms")
    res_raw = main(mol, decomp_raw, mf_li_rohf, mf_li_rohf.mo_coeff, mf_li_rohf.mo_occ)
    for key, val in res.res_dict.items():
        assert np.allclose(val, res_raw.res_dict[key], atol=1e-10)


# orbitals partitioning must give the same result for non-Aufbau occupation
def test_main_orbitals_non_aufbau():
    mol = gto.M(
        verbose=0, output=None, basis="sto-3g", symmetry=True, atom="geom/h2o.xyz"
    )
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
    assert np.allclose(res.el[0], res_sliced.el[0], atol=1e-10)
    assert np.allclose(res.el[1], res_sliced.el[1], atol=1e-10)


# nuc_att_glob and nuc_att_loc must sum to the total nuclear attraction energy
@pytest.mark.parametrize("part_method", ["mo", "ao"])
def test_nuc_att_sum(mf_h2o_rhf, part_method):
    mol = mf_h2o_rhf.mol
    mo = mf_h2o_rhf.mo_coeff[:, mf_h2o_rhf.mo_occ > 0.0]
    mo_occ = (np.ones(mo.shape[1]), np.ones(mo.shape[1]))
    decomp = DecompCls(pop_method="iao", part="atoms", part_method=part_method)
    res = main(mol, decomp, mf_h2o_rhf, (mo, mo), mo_occ)
    rdm1 = mf_h2o_rhf.make_rdm1()
    total_nuc_att = np.sum(mol.intor("int1e_nuc") * rdm1)
    assert np.isclose(
        (res.nuc_att_glob + res.nuc_att_loc).sum(), total_nuc_att, atol=1e-10
    )


# dipole moment must be gauge-origin independent for neutral molecule and must match pyscf reference
@pytest.mark.parametrize("part_method", ["mo", "ao"])
def test_main_dipole(mf_h2o_rhf, part_method):
    mol = mf_h2o_rhf.mol
    mo = mf_h2o_rhf.mo_coeff[:, mf_h2o_rhf.mo_occ > 0.0]
    mo_occ = (np.ones(mo.shape[1]), np.ones(mo.shape[1]))
    decomp1 = DecompCls(
        pop_method="iao", part="atoms", part_method=part_method, prop="dipole"
    )
    res1 = main(mol, decomp1, mf_h2o_rhf, (mo, mo), mo_occ)
    decomp2 = DecompCls(
        pop_method="iao",
        part="atoms",
        part_method=part_method,
        prop="dipole",
        gauge_origin=[1.0, -2.0, 0.5],
    )
    res2 = main(mol, decomp2, mf_h2o_rhf, (mo, mo), mo_occ)
    assert np.allclose(res1.tot.sum(axis=0), res2.tot.sum(axis=0), atol=1e-10)
    assert np.allclose(
        res1.tot.sum(axis=0), mf_h2o_rhf.dip_moment(unit="au", verbose=0), atol=1e-10
    )
