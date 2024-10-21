#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
properties module
"""

__author__ = "Janus Juul Eriksen, Technical University of Denmark, DK"
__maintainer__ = "Janus Juul Eriksen"
__email__ = "janus@kemi.dtu.dk"
__status__ = "Development"

import copy
import numpy as np
from itertools import starmap
from pyscf import gto, scf, dft, df, lo, lib, solvent
from pyscf.dft import numint
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.dft import numint as pbc_numint
from typing import List, Tuple, Dict, Union, Any, Optional

from .pbctools import ewald_e_nuc, get_nuc_pbc
from .tools import dim, make_rdm1, orbsym, contract
from .decomp import CompKeys

# block size in _mm_pot()
BLKSIZE = 200


def prop_tot(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[
        scf.hf.SCF,
        dft.rks.KohnShamDFT,
        pbc_scf.hf.RHF,
        pbc_scf.uhf.UHF,
        pbc_dft.rks.RKS,
        pbc_dft.uks.UKS,
    ],
    mo_coeff: Tuple[np.ndarray, np.ndarray],
    mo_occ: Tuple[np.ndarray, np.ndarray],
    rdm1: Optional[np.ndarray],
    minao: str,
    pop_method: str,
    prop_type: str,
    part: str,
    ndo: bool,
    gauge_origin: np.ndarray,
    weights: List[np.ndarray],
) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
    """
    this function returns atom-decomposed mean-field properties
    """
    # declare nested kernel functions in global scope
    global prop_atom
    global prop_eda
    global prop_orb

    # restricted reference
    if mo_occ[0].size == mo_occ[1].size:
        restrict = np.allclose(mo_coeff[0], mo_coeff[1]) and np.allclose(
            mo_occ[0], mo_occ[1]
        )
    else:
        restrict = False

    # dft logical
    dft_calc = isinstance(mf, dft.rks.KohnShamDFT)

    # ao dipole integrals with specified gauge origin
    if prop_type == "dipole":
        with mol.with_common_origin(gauge_origin):
            ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    else:
        ao_dip = None

    # compute total 1-RDMs (AO basis)
    if rdm1 is None:
        rdm1 = np.array(
            [make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])]
        )
    if rdm1.ndim == 2:
        rdm1 = np.array([rdm1, rdm1]) * 0.5
    rdm1_tot = np.array(
        [make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])]
    )

    # mol object projected into minao basis
    if pop_method == "iao":
        pmol = lo.iao.reference_mol(mol, minao=minao)
    else:
        pmol = mol

    # effective atomic charges
    if part in ["atoms", "eda"]:
        charge_atom = (
            -(np.sum(weights[0], axis=0) + np.sum(weights[1], axis=0))
            + pmol.atom_charges()
        )
    else:
        charge_atom = 0.0

    # possible mm region
    mm_mol = getattr(mf, "mm_mol", None)

    # possible cosmo/pcm solvent model
    if getattr(mf, "with_solvent", None):
        e_solvent = _solvent(mol, np.sum(rdm1, axis=0), mf.with_solvent)
    else:
        e_solvent = None

    # nuclear repulsion property
    if prop_type == "energy":
        if isinstance(mol, pbc_gto.Cell):
            prop_nuc_rep = ewald_e_nuc(mol)
        else:
            prop_nuc_rep = _e_nuc(pmol, mm_mol)
    elif prop_type == "dipole":
        prop_nuc_rep = _dip_nuc(pmol, gauge_origin)

    # core hamiltonian
    kin, nuc, sub_nuc, mm_pot = _h_core(mol, mm_mol, mf)
    # fock potential
    if hasattr(mf, "vj"):
        vj = copy.copy(mf.vj)
        if hasattr(mf, "vk"):
            vk = copy.copy(mf.vk)
        else:
            _, vk = mf.get_jk(
                mol=mol,
                dm=np.sum(rdm1, axis=0) if restrict else rdm1,
                with_j=False,
                with_k=not dft_calc,
            )
    else:
        vj, vk = mf.get_jk(
            mol=mol, dm=np.sum(rdm1, axis=0) if restrict else rdm1, with_k=not dft_calc
        )

    # class for xc parameters
    class XCParams:
        grid_weights: np.ndarray
        grid_weights_nlc: np.ndarray
        ao_value: np.ndarray
        ao_value_nlc: np.ndarray
        eps_xc: np.ndarray
        eps_xc_nlc: Optional[np.ndarray]
        c0_tot: np.ndarray
        c1_tot: Optional[np.ndarray]
        c0_vv10: np.ndarray
        c1_vv10: Optional[np.ndarray]

    xc_params: Optional[XCParams] = None

    # calculate xc energy density
    if dft_calc:
        # inititialize class for xc paramters
        xc_params = XCParams()
        # ndo assertion
        if ndo:
            raise NotImplementedError(
                "NDOs for KS-DFT do not yield a lossless decomposition"
            )
        # xc-type and ao_deriv
        xc_type, ao_deriv = _xc_ao_deriv(mf.xc)
        # update exchange operator wrt range-separated parameter and exact exchange
        # components
        vk = _vk_dft(mol, mf, mf.xc, np.sum(rdm1, axis=0) if restrict else rdm1, vk, vj)
        # ao function values on given grid
        xc_params.ao_value = _ao_val(mol, mf.grids.coords, ao_deriv)
        # grid weights
        xc_params.grid_weights = mf.grids.weights
        # compute all intermediates
        xc_params.c0_tot, xc_params.c1_tot, rho_tot = _make_rho(
            xc_params.ao_value, rdm1, xc_type
        )
        # evaluate xc energy density
        xc_params.eps_xc = dft.libxc.eval_xc(
            mf.xc, rho_tot, spin=0 if rho_tot.ndim == 2 else 1
        )[0]
        # nlc (vv10)
        if isinstance(mol, pbc_gto.Cell):
            xc_params.eps_xc_nlc = None
        else:
            if mf.nlc.upper() == "VV10":
                nlc_pars = dft.libxc.nlc_coeff(mf.xc)[0][0]
                xc_params.ao_value_nlc = _ao_val(mol, mf.nlcgrids.coords, 1)
                xc_params.grid_weights_nlc = mf.nlcgrids.weights
                xc_params.c0_vv10, xc_params.c1_vv10, rho_vv10 = _make_rho(
                    xc_params.ao_value_nlc, np.sum(rdm1, axis=0), "GGA"
                )
                xc_params.eps_xc_nlc = numint._vv10nlc(
                    rho_vv10,
                    mf.nlcgrids.coords,
                    rho_vv10,
                    xc_params.grid_weights_nlc,
                    mf.nlcgrids.coords,
                    nlc_pars,
                )[0]
            else:
                xc_params.eps_xc_nlc = None

    # molecular dimensions
    alpha, beta = dim(mo_occ)

    # atomic labels
    if part == "eda":
        ao_labels = mol.ao_labels(fmt=None)

    def prop_atom(atom_idx: int) -> Dict[str, Any]:
        """
        this function returns atom-wise energy/dipole contributions
        """
        # init results
        res: Dict[str, Union[float, np.ndarray]] = {}
        # atom-specific rdm1
        rdm1_atom = np.zeros_like(rdm1_tot)
        # loop over spins
        if prop_type == "energy" and not restrict:
            res[CompKeys.coul] = 0.0
            res[CompKeys.exch] = 0.0
        for i, spin_mo in enumerate((alpha, beta)):
            # loop over spin-orbitals
            for m, j in enumerate(spin_mo):
                # get orbital(s)
                orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], -1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                # weighted contribution to rdm1_atom
                rdm1_atom[i] += (
                    rdm1_orb * weights[i][m][atom_idx] / np.sum(weights[i][m])
                )
            # coulumb & exchange energy associated with given atom
            if prop_type == "energy" and not restrict:
                res[CompKeys.coul] += _trace(
                    np.sum(vj, axis=0), rdm1_atom[i], scaling=0.5
                )
                res[CompKeys.exch] -= _trace(vk[i], rdm1_atom[i], scaling=0.5)
        # common energy contributions associated with given atom
        if prop_type == "energy":
            if restrict:
                res[CompKeys.coul] = _trace(vj, np.sum(rdm1_atom, axis=0), scaling=0.5)
                res[CompKeys.exch] = -_trace(
                    vk, np.sum(rdm1_atom, axis=0), scaling=0.25
                )
            res[CompKeys.kin] = _trace(kin, np.sum(rdm1_atom, axis=0))
            res[CompKeys.nuc_att_loc] = _trace(
                nuc, np.sum(rdm1_atom, axis=0), scaling=0.5
            )
            res[CompKeys.nuc_att_glob] = _trace(
                sub_nuc[atom_idx], np.sum(rdm1_tot, axis=0), scaling=0.5
            )
            if mm_pot is not None:
                res[CompKeys.solvent] = _trace(mm_pot, np.sum(rdm1_atom, axis=0))
            if e_solvent is not None:
                res[CompKeys.solvent] = e_solvent[atom_idx]
            # additional xc energy contribution
            if dft_calc and xc_params is not None:
                # atom-specific rho
                _, _, rho_atom = _make_rho(
                    xc_params.ao_value, np.sum(rdm1_atom, axis=0), xc_type
                )
                # energy from individual atoms
                res[CompKeys.xc] = _e_xc(
                    xc_params.eps_xc, xc_params.grid_weights, rho_atom
                )
                # nlc (vv10)
                if xc_params.eps_xc_nlc is not None:
                    _, _, rho_atom_vv10 = _make_rho(
                        xc_params.ao_value_nlc, np.sum(rdm1_atom, axis=0), "GGA"
                    )
                    res[CompKeys.xc] += _e_xc(
                        xc_params.eps_xc_nlc, xc_params.grid_weights_nlc, rho_atom_vv10
                    )
        elif prop_type == "dipole":
            res[CompKeys.el] = -_trace(ao_dip, np.sum(rdm1_atom, axis=0))
        # sum up electronic contributions
        if prop_type == "energy":
            res[CompKeys.el] = sum(res.values())
        return res

    def prop_eda(atom_idx: int) -> Dict[str, Any]:
        """
        this function returns EDA energy/dipole contributions
        """
        # init results
        res = {}
        # get AOs on atom k
        select = np.where([atom[0] == atom_idx for atom in ao_labels])[0]
        # common energy contributions associated with given atom
        if prop_type == "energy":
            if restrict:
                res[CompKeys.coul] = _trace(
                    vj[select], np.sum(rdm1_tot, axis=0)[select], scaling=0.5
                )
                res[CompKeys.exch] = -_trace(
                    vk[select], np.sum(rdm1_tot, axis=0)[select], scaling=0.25
                )
            else:
                res[CompKeys.coul] = 0.0
                res[CompKeys.exch] = 0.0
                # loop over spins
                for i, _ in enumerate((alpha, beta)):
                    res[CompKeys.coul] += _trace(
                        np.sum(vj, axis=0)[select], rdm1_tot[i][select], scaling=0.5
                    )
                    res[CompKeys.exch] -= _trace(
                        vk[i][select], rdm1_tot[i][select], scaling=0.5
                    )
            res[CompKeys.kin] = _trace(kin[select], np.sum(rdm1_tot, axis=0)[select])
            res[CompKeys.nuc_att_loc] = _trace(
                nuc[select], np.sum(rdm1_tot, axis=0)[select], scaling=0.5
            )
            res[CompKeys.nuc_att_glob] = _trace(
                sub_nuc[atom_idx], np.sum(rdm1_tot, axis=0), scaling=0.5
            )
            if mm_pot is not None:
                res[CompKeys.solvent] = _trace(
                    mm_pot[select], np.sum(rdm1_tot, axis=0)[select]
                )
            if e_solvent is not None:
                res[CompKeys.solvent] = e_solvent[atom_idx]
            # additional xc energy contribution
            if dft_calc and xc_params is not None:
                # atom-specific rho
                rho_atom = _make_rho_interm2(
                    xc_params.c0_tot[:, select],
                    (
                        xc_params.c1_tot[:, :, select]
                        if xc_params.c1_tot is not None
                        else xc_params.c1_tot
                    ),
                    xc_params.ao_value[:, :, select],
                    xc_type,
                )
                # energy from individual atoms
                res[CompKeys.xc] = _e_xc(
                    xc_params.eps_xc, xc_params.grid_weights, rho_atom
                )
                # nlc (vv10)
                if xc_params.eps_xc_nlc is not None:
                    rho_atom_vv10 = _make_rho_interm2(
                        xc_params.c0_vv10[:, select],
                        (
                            xc_params.c1_vv10[:, :, select]
                            if xc_params.c1_vv10 is not None
                            else xc_params.c1_vv10
                        ),
                        xc_params.ao_value_nlc[:, :, select],
                        "GGA",
                    )
                    res[CompKeys.xc] += _e_xc(
                        xc_params.eps_xc_nlc, xc_params.grid_weights_nlc, rho_atom_vv10
                    )
        elif prop_type == "dipole":
            res[CompKeys.el] = -_trace(
                ao_dip[:, select], np.sum(rdm1_tot, axis=0)[select]
            )
        # sum up electronic contributions
        if prop_type == "energy":
            res[CompKeys.el] = sum(res.values())
        return res

    def prop_orb(spin_idx: int, orb_idx: int) -> Dict[str, Any]:
        """
        this function returns bond-wise energy/dipole contributions
        """
        # init res
        res = {}
        # get orbital(s)
        orb = mo_coeff[spin_idx][:, orb_idx].reshape(mo_coeff[spin_idx].shape[0], -1)
        # orbital-specific rdm1
        rdm1_orb = make_rdm1(orb, mo_occ[spin_idx][orb_idx])
        # total energy or dipole moment associated with given spin-orbital
        if prop_type == "energy":
            if restrict:
                res[CompKeys.coul] = _trace(vj, rdm1_orb, scaling=0.5)
                res[CompKeys.exch] = -_trace(vk, rdm1_orb, scaling=0.25)
            else:
                res[CompKeys.coul] = _trace(np.sum(vj, axis=0), rdm1_orb, scaling=0.5)
                res[CompKeys.exch] = -_trace(vk[spin_idx], rdm1_orb, scaling=0.5)
            res[CompKeys.kin] = _trace(kin, rdm1_orb)
            res[CompKeys.nuc_att] = _trace(nuc, rdm1_orb)
            if mm_pot is not None:
                res[CompKeys.solvent] = _trace(mm_pot, rdm1_orb)
            # additional xc energy contribution
            if dft_calc and xc_params is not None:
                # orbital-specific rho
                _, _, rho_orb = _make_rho(xc_params.ao_value, rdm1_orb, xc_type)
                # xc energy from individual orbitals
                res[CompKeys.xc] = _e_xc(
                    xc_params.eps_xc, xc_params.grid_weights, rho_orb
                )
                # nlc (vv10)
                if xc_params.eps_xc_nlc is not None:
                    _, _, rho_orb_vv10 = _make_rho(
                        xc_params.ao_value_nlc, rdm1_orb, "GGA"
                    )
                    res[CompKeys.xc] += _e_xc(
                        xc_params.eps_xc_nlc, xc_params.grid_weights_nlc, rho_orb_vv10
                    )
        elif prop_type == "dipole":
            res[CompKeys.el] = -_trace(ao_dip, rdm1_orb)
        # sum up electronic contributions
        if prop_type == "energy":
            res[CompKeys.el] = sum(res.values())
        return res

    # perform decomposition
    prop: Dict[str, Union[np.ndarray, List[np.ndarray]]]
    if part in ["atoms", "eda"]:
        # domain
        domain = np.arange(pmol.natm)
        # execute kernel
        res = list(map(prop_atom if part == "atoms" else prop_eda, domain))
        # init atom-specific energy or dipole arrays
        if prop_type == "energy":
            prop = {
                comp_key: np.zeros(pmol.natm, dtype=np.float64)
                for comp_key in res[0].keys()
            }
        elif prop_type == "dipole":
            prop = {
                comp_key: np.zeros([pmol.natm, 3], dtype=np.float64)
                for comp_key in res[0].keys()
            }
        # collect results
        for k, r in enumerate(res):
            for key, val in r.items():
                prop[key][k] = val
        if ndo:
            prop[CompKeys.struct] = np.zeros_like(prop_nuc_rep)
        else:
            prop[CompKeys.struct] = prop_nuc_rep
        return {**prop, CompKeys.charge_atom: charge_atom}
    else:  # orbs
        # domain
        domain = np.array(
            [(i, j) for i, orbs in enumerate((alpha, beta)) for j in orbs]
        )
        # execute kernel
        res = list(starmap(prop_orb, domain))  # type: ignore
        # init orbital-specific energy or dipole array
        if prop_type == "energy":
            prop = {
                comp_key: [np.zeros(alpha.size), np.zeros(beta.size)]
                for comp_key in res[0].keys()
            }
        elif prop_type == "dipole":
            prop = {
                comp_key: [
                    np.zeros([alpha.size, 3], dtype=np.float64),
                    np.zeros([beta.size, 3], dtype=np.float64),
                ]
                for comp_key in res[0].keys()
            }
        # collect results
        for k, r in enumerate(res):
            for key, val in r.items():
                prop[key][domain[k, 0]][domain[k, 1]] = val
        if ndo:
            prop[CompKeys.struct] = np.zeros_like(prop_nuc_rep)
        else:
            prop[CompKeys.struct] = prop_nuc_rep
        return {
            **prop,
            CompKeys.mo_occ: list(mo_occ),
            CompKeys.orbsym: orbsym(mol, mo_coeff),
        }


def _e_nuc(mol: gto.Mole, mm_mol: Optional[gto.Mole]) -> np.ndarray:
    """
    this function returns the nuclear repulsion energy
    """
    # coordinates and charges of nuclei
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    # internuclear distances (with self-repulsion removed)
    dist = gto.inter_distance(mol)
    dist[np.diag_indices_from(dist)] = 1e200
    e_nuc = contract("i,ij,j->i", charges, 1.0 / dist, charges) * 0.5
    # possible interaction with mm sites
    if mm_mol is not None:
        mm_coords = mm_mol.atom_coords()
        mm_charges = mm_mol.atom_charges()
        for j in range(mol.natm):
            q2, r2 = charges[j], coords[j]
            r = lib.norm(r2 - mm_coords, axis=1)
            e_nuc[j] += q2 * np.sum(mm_charges / r)
    return e_nuc


def _dip_nuc(mol: gto.Mole, gauge_origin: np.ndarray) -> np.ndarray:
    """
    this function returns the nuclear contribution to the molecular dipole moment
    """
    # coordinates and formal/actual charges of nuclei
    coords = mol.atom_coords()
    form_charges = mol.atom_charges()
    return contract("i,ix->ix", form_charges, coords - gauge_origin)


def _h_core(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mm_mol: Optional[gto.Mole],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    this function returns the components of the core hamiltonian
    """
    if isinstance(mol, pbc_gto.Cell) and isinstance(
        mf, (pbc_scf.hf.RHF, pbc_scf.uhf.UHF)
    ):
        # kinetic integrals
        kin = mol.pbc_intor("int1e_kin")
        mydf = mf.with_df
        # individual atomic potentials
        sub_nuc = get_nuc_pbc(mol, mydf)
    else:
        # kinetic integrals
        kin = mol.intor_symmetric("int1e_kin")
        # individual atomic potentials
        sub_nuc = _get_nuc(mol)
    # total nuclear potential
    nuc = np.sum(sub_nuc, axis=0)
    # possible mm potential
    if mm_mol is not None:
        mm_pot = _mm_pot(mol, mm_mol)
    else:
        mm_pot = None
    return kin, nuc, sub_nuc, mm_pot


def _get_nuc(mol: gto.Mole) -> np.ndarray:
    """
    individual atomic potentials for molecules
    """
    # coordinates and charges of nuclei
    coords = mol.atom_coords()
    charges = mol.atom_charges()
    # individual atomic potentials
    sub_nuc = np.zeros([mol.natm, mol.nao_nr(), mol.nao_nr()], dtype=np.float64)
    for k in range(mol.natm):
        with mol.with_rinv_origin(coords[k]):
            sub_nuc[k] = -1.0 * mol.intor("int1e_rinv") * charges[k]
    return sub_nuc


def _mm_pot(mol: gto.Mole, mm_mol: gto.Mole) -> np.ndarray:
    """
    this function returns the full mm potential
    (adapted from: qmmm/itrf.py:get_hcore() in PySCF)
    """
    # settings
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    blksize = BLKSIZE
    # integrals
    intor = "int3c2e_cart" if mol.cart else "int3c2e_sph"
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, intor)
    # compute interaction potential
    nao = mol.nao_nr()
    mm_pot = np.zeros(nao * (nao + 1) // 2, dtype=np.float64)
    for i0, i1 in lib.prange(0, charges.size, blksize):
        fakemol = gto.fakemol_for_charges(coords[i0:i1])
        j3c = df.incore.aux_e2(mol, fakemol, intor=intor, aosym="s2ij", cintopt=cintopt)
        mm_pot += np.einsum("xk,k->x", j3c, -charges[i0:i1])
    mm_pot = lib.unpack_tril(mm_pot)
    return mm_pot


def _solvent(
    mol: gto.Mole, rdm1: np.ndarray, solvent_model: solvent.ddcosmo.DDCOSMO
) -> np.ndarray:
    """
    this function return atom-specific PCM/COSMO contributions
    (adapted from: solvent/ddcosmo.py:_get_vind() in PySCF)
    """
    # settings
    r_vdw = solvent_model._intermediates["r_vdw"]
    ylm_1sph = solvent_model._intermediates["ylm_1sph"]
    ui = solvent_model._intermediates["ui"]
    Lmat = solvent_model._intermediates["Lmat"]
    cached_pol = solvent_model._intermediates["cached_pol"]
    dielectric = solvent_model.eps
    f_epsilon = (dielectric - 1.0) / dielectric if dielectric > 0.0 else 1.0
    # electrostatic potential
    phi = solvent.ddcosmo.make_phi(solvent_model, rdm1, r_vdw, ui, ylm_1sph)
    # X and psi
    # (cf. https://github.com/filippolipparini/ddPCM/blob/master/reference.pdf)
    Xvec = np.linalg.solve(Lmat, phi.ravel()).reshape(mol.natm, -1)
    psi = solvent.ddcosmo.make_psi_vmat(
        solvent_model, rdm1, r_vdw, ui, ylm_1sph, cached_pol, Xvec, Lmat
    )[0]
    return 0.5 * f_epsilon * np.einsum("jx,jx->j", psi, Xvec)


def _xc_ao_deriv(xc_func: str) -> Tuple[str, int]:
    """
    this function returns the type of xc functional and the level of ao derivatives
    needed
    """
    xc_type = dft.libxc.xc_type(xc_func)
    if xc_type == "LDA":
        ao_deriv = 0
    elif xc_type in ["GGA", "NLC"]:
        ao_deriv = 1
    elif xc_type == "MGGA":
        ao_deriv = 2
    return xc_type, ao_deriv


def _make_rho_interm1(
    ao_value: np.ndarray, rdm1: np.ndarray, xc_type: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    this function returns the rho intermediates (c0, c1) needed in _make_rho()
    (adpated from: dft/numint.py:eval_rho() in PySCF)
    """
    # determine dimensions based on xctype
    xctype = xc_type.upper()
    if xctype == "LDA" or xctype == "HF":
        ngrids, nao = ao_value.shape
    else:
        ngrids, nao = ao_value[0].shape
    # compute rho intermediate based on xctype
    if xctype == "LDA" or xctype == "HF":
        c0 = contract("ik,kj->ij", ao_value, rdm1)
        c1 = None
    elif xctype in ("GGA", "NLC"):
        c0 = contract("ik,kj->ij", ao_value[0], rdm1)
        c1 = None
    else:  # meta-GGA
        c0 = contract("ik,kj->ij", ao_value[0], rdm1)
        c1 = np.empty((3, ngrids, nao), dtype=np.float64)
        for i in range(1, 4):
            c1[i - 1] = contract("ik,jk->ij", ao_value[i], rdm1)
    return c0, c1


def _make_rho_interm2(
    c0: np.ndarray, c1: Optional[np.ndarray], ao_value: np.ndarray, xc_type: str
) -> np.ndarray:
    """
    this function returns rho from intermediates (c0, c1)
    (adpated from: dft/numint.py:eval_rho() in PySCF)
    """
    # determine dimensions based on xctype
    xctype = xc_type.upper()
    if xctype == "LDA" or xctype == "HF":
        ngrids = ao_value.shape[0]
    else:
        ngrids = ao_value[0].shape[0]
    # compute rho intermediate based on xctype
    if xctype == "LDA" or xctype == "HF":
        rho = contract("pi,pi->p", ao_value, c0)
    elif xctype in ("GGA", "NLC"):
        rho = np.empty((4, ngrids), dtype=np.float64)
        rho[0] = contract("pi,pi->p", c0, ao_value[0])
        for i in range(1, 4):
            rho[i] = contract("pi,pi->p", c0, ao_value[i]) * 2.0
    else:  # meta-GGA
        assert c1 is not None
        rho = np.empty((6, ngrids), dtype=np.float64)
        rho[0] = contract("pi,pi->p", ao_value[0], c0)
        rho[5] = 0.0
        for i in range(1, 4):
            rho[i] = contract("pi,pi->p", c0, ao_value[i]) * 2.0
            rho[5] += contract("pi,pi->p", c1[i - 1], ao_value[i])
        XX, YY, ZZ = 4, 7, 9
        ao_value_2 = ao_value[XX] + ao_value[YY] + ao_value[ZZ]
        rho[4] = contract("pi,pi->p", c0, ao_value_2)
        rho[4] += rho[5]
        rho[4] *= 2.0
        rho[5] *= 0.5
    return rho


def _make_rho(
    ao_value: np.ndarray, rdm1: np.ndarray, xc_type: str
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    this function returns important dft intermediates, e.g., energy density, grid
    weights, etc.
    """
    # rho corresponding to given 1-RDM
    if rdm1.ndim == 2:
        c0, c1 = _make_rho_interm1(ao_value, rdm1, xc_type)
        rho = _make_rho_interm2(c0, c1, ao_value, xc_type)
    else:
        if np.allclose(rdm1[0], rdm1[1]):
            c0, c1 = _make_rho_interm1(ao_value, rdm1[0] * 2.0, xc_type)
            rho = _make_rho_interm2(c0, c1, ao_value, xc_type)
        else:
            c0_a, c1_a = _make_rho_interm1(ao_value, rdm1[0], xc_type)
            c0_b, c1_b = _make_rho_interm1(ao_value, rdm1[1], xc_type)
            rho = np.stack(
                (
                    _make_rho_interm2(c0_a, c1_a, ao_value, xc_type),
                    _make_rho_interm2(c0_b, c1_b, ao_value, xc_type),
                )
            )
            c0 = c0_a + c0_b
            if c1_a is not None and c1_b is not None:
                c1 = c1_a + c1_b
            else:
                c1 = None
    return c0, c1, rho


def _vk_dft(
    mol: gto.Mole,
    mf: dft.rks.KohnShamDFT,
    xc_func: str,
    rdm1: np.ndarray,
    vk: np.ndarray,
    vj: np.ndarray,
) -> np.ndarray:
    """
    this function returns the appropriate dft exchange operator
    """
    # range-separated and exact exchange parameters
    ks_omega, ks_alpha, ks_hyb = mf._numint.rsh_and_hybrid_coeff(xc_func)
    # if hybrid func: compute vk
    if abs(ks_hyb) > 1e-10:
        if not hasattr(mf, "vk"):
            vk = mf.get_k(mol=mol, dm=rdm1)
        # scale amount of exact exchange
        vk *= ks_hyb
    else:
        vk = np.zeros_like(vj)
    # range separated coulomb operator
    if abs(ks_omega) > 1e-10:
        vk_lr = mf.get_k(mol, rdm1, omega=ks_omega)
        vk_lr *= ks_alpha - ks_hyb
        if not hasattr(mf, "vk"):
            vk += vk_lr
        else:
            vk += np.sum(vk_lr, axis=0)
    return vk


def _ao_val(mol: gto.Mole, grids_coords: np.ndarray, ao_deriv: int) -> np.ndarray:
    """
    this function returns ao function values on the given grid
    """
    if not isinstance(mol, pbc_gto.Cell):
        return numint.eval_ao(mol, grids_coords, deriv=ao_deriv)
    else:
        return pbc_numint.eval_ao(mol, grids_coords, deriv=ao_deriv)


def _trace(
    op: np.ndarray, rdm1: np.ndarray, scaling: float = 1.0
) -> Union[float, np.ndarray]:
    """
    this function returns the trace between an operator and an rdm1
    """
    if op.ndim == 2:
        return contract("ij,ij", op, rdm1) * scaling
    else:
        return contract("xij,ij->x", op, rdm1) * scaling


def _e_xc(eps_xc: np.ndarray, grid_weights: np.ndarray, rho: np.ndarray) -> float:
    """
    this function returns a contribution to the exchange-correlation energy from given
    rmd1 (via rho)
    """
    return contract("i,i,i->", eps_xc, rho if rho.ndim == 1 else rho[0], grid_weights)
