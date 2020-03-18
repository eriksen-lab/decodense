#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
properties module
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

from .tools import make_rdm1


def prop_tot(mol: gto.Mole, mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
             prop_type: str, mo_coeff: Tuple[np.ndarray, np.ndarray], mo_occ: Tuple[np.ndarray, np.ndarray], \
             rep_idx: List[List[np.ndarray]]) -> np.ndarray:
        """
        this function returns sorted orbital-decomposed mean-field properties for a given orbital type
        """
        # ao dipole integrals with gauge origin at (0.0, 0.0, 0.0)
        if prop_type == 'dipole':
            with mol.with_common_origin([0.0, 0.0, 0.0]):
                ao_dip = mol.intor_symmetric('int1e_r', comp=3)

        # compute total 1-RDM (AO basis)
        rdm1_tot = np.array([make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])])

        # core hamiltonian
        h_core = _h_core(mol)
        # fock potential
        vj, vk = scf.hf.get_jk(mol, rdm1_tot)

        if isinstance(mf, dft.rks.KohnShamDFT):
            # xc-type and ao_deriv
            xc_type, ao_deriv = _xc_ao_deriv(mf)
            # update exchange operator wrt range-separated parameter and exact exchange components
            vk = _vk_dft(mol, mf, rdm1_tot, vk)
            # ao function values on given grid
            ao_value = _ao_val(mol, mf, ao_deriv)
            # rho corresponding to total 1-RDM
            if mol.spin == 0:
                rho = numint.eval_rho(mol, ao_value, rdm1_tot[0] + rdm1_tot[1], xctype=xc_type)
            else:
                rho = (numint.eval_rho(mol, ao_value, rdm1_tot[0], xctype=xc_type), \
                       numint.eval_rho(mol, ao_value, rdm1_tot[1], xctype=xc_type))
            # evaluate eps_xc (xc energy density)
            eps_xc = dft.libxc.eval_xc(mf.xc, rho, spin=mol.spin)[0]

        # init orbital-specific energy or dipole array
        if prop_type == 'energy':
            prop_orb = [np.zeros(len(rep_idx[0]), dtype=np.float64), np.zeros(len(rep_idx[1]), dtype=np.float64)]
        elif prop_type == 'dipole':
            prop_orb = [np.zeros([len(rep_idx[0]), 3], dtype=np.float64), np.zeros([len(rep_idx[1]), 3], dtype=np.float64)]

        # loop over orbitals
        for i in range(2):
            for j, k in enumerate(rep_idx[i]):
                # get orbital(s)
                orb = mo_coeff[i][:, k].reshape(mo_coeff[i].shape[0], -1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mo_occ[i][k])
                # energy or dipole from individual orbitals
                if prop_type == 'energy':
                    prop_orb[i][j] = _e_elec(h_core, vj[0] + vj[1], vk[i], rdm1_orb)
                elif prop_type == 'dipole':
                    prop_orb[i][j] = -np.einsum('xij,ji->x', ao_dip, rdm1_orb).real
                # additional xc energy contribution
                if prop_type == 'energy' and isinstance(mf, dft.rks.KohnShamDFT):
                    # orbital-specific rho
                    rho_orb = numint.eval_rho(mol, ao_value, rdm1_orb, xctype=xc_type)
                    # energy from individual orbitals
                    prop_orb[i][j] += _e_xc(eps_xc, mf.grids.weights, rho_orb)

        return prop_orb


def e_nuc(mol: gto.Mole) -> np.ndarray:
        """
        this function returns the nuclear repulsion energy
        """
        # coordinates and charges of nuclei
        coords = mol.atom_coords()
        charges = mol.atom_charges()
        # internuclear distances (with self-repulsion removed)
        dist = gto.inter_distance(mol)
        dist[np.diag_indices_from(dist)] = 1e200

        return np.einsum('i,ij,j->i', charges, 1. / dist, charges) * .5


def dip_nuc(mol: gto.Mole) -> np.ndarray:
        """
        this function returns the nuclear contribution to the molecular dipole moment
        """
        # coordinates and charges of nuclei
        coords = mol.atom_coords()
        charges = mol.atom_charges()

        return np.einsum('i,ix->ix', charges, coords)


def _h_core(mol: gto.Mole) -> np.ndarray:
        """
        this function returns the core hamiltonian
        """
        return mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')


def _xc_ao_deriv(mf: dft.rks.KohnShamDFT) -> Tuple[str, int]:
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


def _vk_dft(mol: gto.Mole, mf: dft.rks.KohnShamDFT, rdm1: np.ndarray, vk: np.ndarray) -> np.ndarray:
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


def _ao_val(mol: gto.Mole, mf: dft.rks.KohnShamDFT, ao_deriv: int) -> np.ndarray:
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
        e_veff = np.einsum('ij,ji', vj - vk, rdm1) * .5
        return e_core + e_veff


def _e_xc(eps_xc: np.ndarray, weights: np.ndarray, rho: np.ndarray) -> float:
        """
        this function returns a contribution to exchange-correlation energy contribution from given rmd1 (rho)
        """
        if rho.ndim == 1:
            e_xc = np.einsum('i,i,i->', eps_xc, rho, weights)
        else:
            e_xc = np.einsum('i,i,i->', eps_xc, rho[0], weights)
        return e_xc


