#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
energy module containing all functions related to energy calculations in mf_decomp
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import gto, scf, dft
from pyscf.dft import numint
from pyscf import tools as pyscf_tools
from typing import List, Tuple, Union

import results


def e_tot(mol: gto.Mole, \
            mf: Union[scf.hf.RHF, scf.hf_symm.RHF, dft.rks.RKS, dft.rks_symm.RKS], \
            orb_type: str, ao_dip: np.ndarray, mo_coeff: np.ndarray, \
            rep_idx: List[np.ndarray], cube: bool) -> Union[np.ndarray, np.ndarray]:
    """
    this function returns sorted an orbital-decomposed mean-field energy and dipole moment for a given orbital variant
    """
    # compute total 1-RDM (AO basis)
    rdm1 = _rdm1(mo_coeff)

    # core hamiltonian
    h_core = _h_core(mol)
    # fock potential
    vj, vk = mf.get_jk(mol, rdm1)

    if isinstance(mf, (dft.rks.RKS, dft.rks_symm.RKS)):
        # xc-type and ao_deriv
        xc_type, ao_deriv = _xc_ao_deriv(mf)
        # update exchange operator wrt range-separated parameter and exact exchange components
        vk = _vk_dft(mol, mf, rdm1, vk)
        # ao function values on given grid
        ao_value = _ao_val(mol, mf, ao_deriv)
        # rho corresponding to total 1-RDM
        rho = numint.eval_rho(mol, ao_value, rdm1, xctype=xc_type)
        # evaluate eps_xc
        eps_xc = dft.libxc.eval_xc(mf.xc, rho)[0]

    # init orbital-specific energy array
    e_orb = np.zeros(len(rep_idx), dtype=np.float64)
    # init orbital-specific dipole array
    dip_orb = np.zeros([len(rep_idx), 3], dtype=np.float64)

    # loop over orbitals
    for i, j in enumerate(rep_idx):
        # get orbital(s)
        orb = mo_coeff[:, j].reshape(mo_coeff.shape[0], -1)
        # orbital-specific rdm1
        rdm1_orb = _rdm1(orb)
        # energy from individual orbitals
        e_orb[i] = _e_elec(h_core, vj, vk, rdm1_orb)
        # dipole from individual orbitals
        dip_orb[i] = np.einsum('xij,ji->x', ao_dip, rdm1_orb).real
        if isinstance(mf, (dft.rks.RKS, dft.rks_symm.RKS)):
            # orbital-specific rho
            rho_orb = numint.eval_rho(mol, ao_value, rdm1_orb, xctype=xc_type)
            # energy from individual orbitals
            e_orb[i] += _e_xc(eps_xc, mf.grids.weights, rho_orb)
        # write cube file
        if cube:
            pyscf_tools.cubegen.density(mol, results.OUT + orb_type + '/rdm1_{:}_tmp.cube'.format(i), rdm1_orb)

    return e_orb, dip_orb


def _h_core(mol: gto.Mole) -> np.ndarray:
    """
    this function returns the core hamiltonian
    """
    return mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')


def _rdm1(mo: np.ndarray) -> np.ndarray:
    """
    this function returns an 1-RDM (in ao basis) corresponding to given mo(s)
    """
    return np.einsum('ip,jp->ij', mo, mo) * 2.


def _xc_ao_deriv(mf: Union[dft.rks.RKS, dft.rks_symm.RKS]) -> Tuple[str, int]:
    """
    this function returns the type of xc functional and the level of ao derivatives needed
    """
    xc_type = dft.libxc.xc_type(mf.xc)
    if xc_type == 'LDA':
        ao_deriv = 0
    elif xc_type in ['GGA', 'NLC']:
        ao_deriv = 1
    elif xc_type == 'MGGA':
        ao_deriv = 2
    return xc_type, ao_deriv


def _vk_dft(mol: gto.Mole, mf: Union[dft.rks.RKS, dft.rks_symm.RKS], rdm1: np.ndarray, vk: np.ndarray) -> np.ndarray:
    """
    this function returns the appropriate dft exchange operator
    """
    # range-separated and exact exchange parameters
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc)
    # scale amount of exact exchange
    vk *= hyb
    # range separated coulomb operator
    if abs(omega) > 1e-10:
        vk_lr = mf.get_k(mol, rdm1, omega=omega)
        vk_lr *= (alpha - hyb)
        vk += vk_lr
    return vk


def _ao_val(mol: gto.Mole, mf: Union[dft.rks.RKS, dft.rks_symm.RKS], ao_deriv: int) -> np.ndarray:
    """
    this function returns ao function values on the given grid
    """
    return numint.eval_ao(mol, mf.grids.coords, deriv=ao_deriv)


def _e_elec(h_core: np.ndarray, vj: np.ndarray, vk: np.ndarray, rdm1: np.ndarray) -> float:
    """
    this function returns a mean-field energy contribution from given rdm1
    """
    # contribution from core hamiltonian
    e_core = np.einsum('ij,ji', h_core, rdm1)
    # contribution from effective potential
    e_veff = np.einsum('ij,ji', vj - vk * .5, rdm1) * .5
    return e_core + e_veff


def _e_xc(eps_xc: np.ndarray, weights: np.ndarray, rho: np.ndarray) -> float:
    """
    this function returns a contribution to exchange-correlation energy contribution from given rmd1
    """
    if rho.ndim == 1:
        e_xc = np.einsum('i,i,i->', eps_xc, rho, weights)
    else:
        e_xc = np.einsum('i,i,i->', eps_xc, rho[0], weights)
    return e_xc


