#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import pytest
from pyscf import gto, scf, dft

from decodense.decomp import CompKeys
from decodense.properties import prop_tot
from decodense.tools import make_rdm1


# helper: converged hf/dft calculation of h2o (restricted) or h2o+ (unrestricted)
def _mf_h2o(restrict, xc):
    charge, spin = (0, 0) if restrict else (1, 1)
    mol = gto.M(
        verbose=0,
        output=None,
        basis="sto-3g",
        atom="geom/h2o.xyz",
        charge=charge,
        spin=spin,
    )
    if xc == "hf":
        mf = scf.RHF(mol) if restrict else scf.UHF(mol)
    else:
        mf = dft.RKS(mol, xc=xc) if restrict else dft.UKS(mol, xc=xc)
    return mf.run(conv_tol=1e-10)


# helper: occupied mo coefficients and occupations in the format of decodense.main,
# and random atom weights (identical for both spins if restricted)
def _decomp_input(mf, restrict):
    if restrict:
        mo_coeff_mf, mo_occ_mf = (mf.mo_coeff, mf.mo_coeff), (mf.mo_occ / 2,) * 2
    else:
        mo_coeff_mf, mo_occ_mf = mf.mo_coeff, mf.mo_occ
    occ = [np.where(o > 0)[0] for o in mo_occ_mf]
    mo_coeff = tuple(mo_coeff_mf[s][:, occ[s]] for s in range(2))
    mo_occ = tuple(np.ones(o.size) for o in occ)
    rng = np.random.default_rng(8)
    weights = [rng.random((o.size, mf.mol.natm)) for o in occ]
    if restrict:
        weights[1] = weights[0]
    return mo_coeff, mo_occ, weights


# helper: kinetic, nuclear attraction and coulomb energies of the total density
def _e_ref_components(mf):
    mol = mf.mol
    dm = mf.make_rdm1()
    dm_tot = dm if dm.ndim == 2 else dm[0] + dm[1]
    e_kin = np.einsum("ij,ij", mol.intor("int1e_kin"), dm_tot)
    e_nuc_att = np.einsum("ij,ij", mol.intor("int1e_nuc"), dm_tot)
    e_coul = 0.5 * np.einsum("ij,ij", scf.hf.get_jk(mol, dm_tot)[0], dm_tot)
    return e_kin, e_nuc_att, e_coul


# helper: sum of an atom-wise array or of orbital-wise arrays (one per spin)
def _total(v):
    return sum(np.sum(x) for x in v) if isinstance(v, list) else np.sum(v)


# prop_tot, energies
# atom-wise ("mo"/"ao") totals must sum to pyscf's total energy, orbital-wise totals
# to the electronic energy; the kinetic, nuclear attraction (local + global for atoms)
# and coulomb parts must sum to their values for the total density
@pytest.mark.parametrize("part_method", ["mo", "ao", None])
@pytest.mark.parametrize("xc", ["hf", "pbe0", "camb3lyp"])
@pytest.mark.parametrize("restrict", [True, False])
def test_prop_tot_energy(restrict, xc, part_method):
    mf = _mf_h2o(restrict, xc)
    mol = mf.mol
    mo_coeff, mo_occ, weights = _decomp_input(mf, restrict)
    res = prop_tot(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        None,
        "minao",
        "mulliken",
        "energy",
        part_method,
        False,
        np.zeros(3),
        weights if part_method == "mo" else None,
    )
    e_ref = mf.e_tot if part_method is not None else mf.e_tot - mol.energy_nuc()
    assert np.isclose(_total(res[CompKeys.tot]), e_ref, atol=1e-8)
    e_kin, e_nuc_att, e_coul = _e_ref_components(mf)
    nuc_att_keys = (CompKeys.nuc_att, CompKeys.nuc_att_loc, CompKeys.nuc_att_glob)
    assert np.isclose(_total(res[CompKeys.kin]), e_kin, atol=1e-8)
    assert np.isclose(
        sum(_total(res[k]) for k in nuc_att_keys if k in res), e_nuc_att, atol=1e-8
    )
    assert np.isclose(_total(res[CompKeys.coul]), e_coul, atol=1e-8)


# prop_tot, dipole moments
# atom-wise totals must sum to pyscf's dipole moment (gauge origin at zero, au)
@pytest.mark.parametrize("part_method", ["mo", "ao"])
@pytest.mark.parametrize("restrict", [True, False])
def test_prop_tot_dipole(restrict, part_method):
    mf = _mf_h2o(restrict, "hf")
    mol = mf.mol
    mo_coeff, mo_occ, weights = _decomp_input(mf, restrict)
    res = prop_tot(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        None,
        "minao",
        "mulliken",
        "dipole",
        part_method,
        False,
        np.zeros(3),
        weights if part_method == "mo" else None,
    )
    assert res[CompKeys.tot].shape == (mol.natm, 3)
    dip_ref = mf.dip_moment(unit="au", verbose=0)
    assert np.allclose(res[CompKeys.tot].sum(axis=0), dip_ref, atol=1e-8)


# prop_tot, atomic reference energies
# atom-wise results must contain the reference energies and the total energy relative
# to them
def test_prop_tot_atom_ref():
    mf = _mf_h2o(True, "hf")
    mo_coeff, mo_occ, weights = _decomp_input(mf, True)
    atom_ref = np.array([-74.0, -0.5, -0.5])
    res = prop_tot(
        mf.mol,
        mf,
        mo_coeff,
        mo_occ,
        None,
        "minao",
        "mulliken",
        "energy",
        "mo",
        False,
        np.zeros(3),
        weights,
        atom_ref,
    )
    assert np.array_equal(res[CompKeys.atom_ref], atom_ref)
    assert np.allclose(res[CompKeys.tot_rel], res[CompKeys.tot] - atom_ref)
