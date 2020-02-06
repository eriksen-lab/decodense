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
from typing import List, Union

import orbitals
import results


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


def e_tot(mol: gto.Mole, mf: Union[scf.hf.RHF, scf.hf_symm.RHF, dft.rks.RKS, dft.rks_symm.RKS], \
            orb_type: str, ao_dip: np.ndarray, mo_coeff: np.ndarray, rep_idx: List[np.ndarray], \
            cube: bool, alpha: float = 1.) -> Union[np.ndarray, np.ndarray]:
    """
    this function returns a sorted orbital-decomposed mean-field energy for a given orbital variant
    """
    # compute total 1-RDM (AO basis)
    rdm1 = np.einsum('ip,jp->ij', mo_coeff, mo_coeff) * 2.

    # core hamiltonian
    h_core = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')

    # fock potential
    vj, vk = mf.get_jk(mol, rdm1)

    if isinstance(mf, (dft.rks.RKS, dft.rks_symm.RKS)):

        # xc-type and ao_deriv
        xc_type = dft.libxc.xc_type(mf.xc)
        if xc_type == 'LDA':
            ao_deriv = 0
        elif xc_type in ['GGA', 'NLC']:
            ao_deriv = 1
        elif xc_type == 'MGGA':
            ao_deriv = 2

        # get range-separated parameter and exact exchange components:
        omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc)
        # scale amount of exact exchange
        vk *= hyb
        # range separated coulomb operator
        if abs(omega) > 1e-10:
            vk_lr = mf.get_k(mol, rdm1, omega=omega)
            vk_lr *= (alpha - hyb)
            vk += vk_lr

        # default mesh grids and weights
        coords = mf.grids.coords
        weights = mf.grids.weights
        ao_value = numint.eval_ao(mol, coords, deriv=ao_deriv)

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
        rdm1_orb = np.einsum('ip,jp->ij', orb, orb) * 2.

        # write cube file
        if cube:
            pyscf_tools.cubegen.density(mol, results.OUT + orb_type + '/rdm1_{:}_tmp.cube'.format(i), rdm1_orb)

        # energy from individual orbitals
        e_orb[i] = _e_elec(h_core, vj, vk, rdm1_orb)

        if isinstance(mf, (dft.rks.RKS, dft.rks_symm.RKS)):

            # orbital-specific rho
            rho_orb = numint.eval_rho(mol, ao_value, rdm1_orb, xctype=xc_type)

            # energy from individual orbitals
            e_orb[i] += _e_xc(eps_xc, weights, rho_orb)

        # dipole from individual orbitals
        dip_orb[i] = np.einsum('xij,ji->x', ao_dip, rdm1_orb).real

    return e_orb, dip_orb


def e_test(mol: gto.Mole, mo_coeff: np.ndarray, rep_idx: List[np.ndarray], mf_dft: dft.RKS) -> float:
    """
    this function returns a sorted orbital-decomposed mean-field energy for a given orbital variant
    """
    # xc-type and ao_deriv
    xc_type = dft.libxc.xc_type(mf_dft.xc)
    if xc_type == 'LDA':
        ao_deriv = 0
    elif xc_type in ['GGA', 'NLC']:
        ao_deriv = 1
    elif xc_type == 'MGGA':
        ao_deriv = 2

    # default mesh grids and weights
    coords = mf_dft.grids.coords
    weights = mf_dft.grids.weights
    ao_value = numint.eval_ao(mol, coords, deriv=ao_deriv)

    # compute total 1-RDM (AO basis)
    rdm1 = np.einsum('ip,jp->ij', mo_coeff, mo_coeff) * 2.

    # rho corresponding to total 1-RDM
    rho = numint.eval_rho(mol, ao_value, rdm1, xctype=xc_type)

    # evaluate eps_xc
    eps_xc = dft.libxc.eval_xc(mf_dft.xc, rho)[0]

    if rho.ndim == 1:
        e_xc = np.einsum('i,i,i->', eps_xc, rho, weights)
    else:
        e_xc = np.einsum('i,i,i->', eps_xc, rho[0], weights)
    print('e_xc full = {:}'.format(e_xc))

    e_xc_sum = 0.

    # loop over orbitals
    for i, j in enumerate(rep_idx):

        # get orbital(s)
        orb = mo_coeff[:, j].reshape(mo_coeff.shape[0], -1)

        # orbital-specific rdm1
        rdm1_orb = np.einsum('ip,jp->ij', orb, orb) * 2.

        # orbital-specific rho
        rho_orb = numint.eval_rho(mol, ao_value, rdm1_orb, xctype=xc_type)

        if rho_orb.ndim == 1:
            e_xc_orb = np.einsum('i,i,i->', eps_xc, rho_orb, weights)
        else:
            e_xc_orb = np.einsum('i,i,i->', eps_xc, rho_orb[0], weights)
        print('e_xc = {:} from orbs = {:}'.format(e_xc_orb, j))
        e_xc_sum += e_xc_orb

    return e_xc_sum


