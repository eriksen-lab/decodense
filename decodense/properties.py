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

from .tools import make_rdm1, write_cube


def prop_tot(mol: gto.Mole, mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
             mo_coeff: Tuple[np.ndarray, np.ndarray], mo_occ: Tuple[np.ndarray, np.ndarray], \
             ref: str, prop_type: str, part: str, cube: bool, \
             **kwargs: Union[List[np.ndarray], List[List[np.ndarray]]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        this function returns atom-decomposed mean-field properties
        """
        # ao dipole integrals with gauge origin at (0.0, 0.0, 0.0)
        if prop_type == 'dipole':
            with mol.with_common_origin([0.0, 0.0, 0.0]):
                ao_dip = mol.intor_symmetric('int1e_r', comp=3)

        # compute total 1-RDM (AO basis)
        rdm1_tot = np.array([make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])])

        # core hamiltonian
        kin, nuc, sub_nuc = _h_core(mol)
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
            if np.allclose(rdm1_tot[0], rdm1_tot[1]):
                rho = numint.eval_rho(mol, ao_value, rdm1_tot[0] * 2., xctype=xc_type)
            else:
                rho = (numint.eval_rho(mol, ao_value, rdm1_tot[0], xctype=xc_type), \
                       numint.eval_rho(mol, ao_value, rdm1_tot[1], xctype=xc_type))
            # evaluate xc energy density
            eps_xc = dft.libxc.eval_xc(mf.xc, rho, spin=0 if isinstance(rho, np.ndarray) else -1)[0]

        if part == 'atoms':

            # get weights
            weights = kwargs['weights']

            # init atom-specific energy or dipole array
            if prop_type == 'energy':
                prop_atom = np.zeros(mol.natm, dtype=np.float64)
            elif prop_type == 'dipole':
                prop_atom = np.zeros([mol.natm, 3], dtype=np.float64)

            # loop over atoms
            for k in range(mol.natm):
                # get atom-specific rdm1
                rdm1_atom = np.zeros_like(rdm1_tot[0])
                # loop over spins
                for i, spin_mo in enumerate((mol.alpha, mol.beta)):
                    # loop over spin-orbitals
                    for l, j in enumerate(spin_mo):
                        # get orbital(s)
                        orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], -1)
                        # orbital-specific rdm1
                        rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                        # weighted contribution to rdm1_atom
                        rdm1_orb_atom = rdm1_orb * weights[i][l][k]
                        # energy or dipole from individual atoms
                        if prop_type == 'energy':
                            prop_atom[k] += _trace(kin + nuc, rdm1_orb_atom)
                            prop_atom[k] += _trace(vj[0] + vj[1] - vk[i], rdm1_orb_atom, scaling = .5)
                        elif prop_type == 'dipole':
                            prop_atom[k] -= _trace(ao_dip, rdm1_orb_atom)
                        # add to rdm1_atom
                        rdm1_atom += rdm1_orb_atom
                # write rdm1_atom as cube file
                if cube:
                    write_cube(mol, rdm1_atom, 'atom_{:s}_rdm1_{:d}'.format(mol.atom_symbol(k).lower(), k))
                # additional xc energy contribution
                if prop_type == 'energy' and isinstance(mf, dft.rks.KohnShamDFT):
                    # atom-specific rho
                    rho_atom = numint.eval_rho(mol, ao_value, rdm1_atom, xctype=xc_type)
                    # energy from individual atoms
                    prop_atom[k] += _e_xc(eps_xc, mf.grids.weights, rho_atom)

            return prop_atom

        elif part == 'eda':

            # init atom-specific energy or dipole array
            if prop_type == 'energy':
                prop_atom = np.zeros(mol.natm, dtype=np.float64)
            elif prop_type == 'dipole':
                prop_atom = np.zeros([mol.natm, 3], dtype=np.float64)

            # loop over atoms
            for k in range(mol.natm):
                # loop over spins
                for i in range(2):
                    # get AOs on atom k
                    select = np.where([atom[0] == k for atom in mol.ao_labels(fmt=None)])[0]
                    # weighted contribution to rdm1_atom
                    rdm1_atom = rdm1_tot[i][select]
                    # energy or dipole from individual atoms
                    if prop_type == 'energy':
                        prop_atom[k] += _trace(kin[select], rdm1_atom)
                        prop_atom[k] += _trace(nuc[select], rdm1_atom, scaling = .5)
                        prop_atom[k] += _trace(sub_nuc[k], rdm1_tot[i], scaling = .5)
                        prop_atom[k] += _trace((vj[0] + vj[1] - vk[i])[select], rdm1_atom, scaling = .5)
                    elif prop_type == 'dipole':
                        prop_atom[k] -= _trace(ao_dip[:, select], rdm1_atom)

            return prop_atom

        else: # bonds

            # get weights
            rep_idx = kwargs['rep_idx']

            # init orbital-specific energy or dipole array
            if prop_type == 'energy':
                prop_orb = [np.zeros(len(rep_idx[0]), dtype=np.float64), np.zeros(len(rep_idx[1]), dtype=np.float64)]
            elif prop_type == 'dipole':
                prop_orb = [np.zeros([len(rep_idx[0]), 3], dtype=np.float64), np.zeros([len(rep_idx[1]), 3], dtype=np.float64)]

            # loop over spins
            for i in range(2):
                # loop over spin-orbitals
                for j, k in enumerate(rep_idx[i]):
                    # get orbital(s)
                    orb = mo_coeff[i][:, k].reshape(mo_coeff[i].shape[0], -1)
                    # orbital-specific rdm1
                    rdm1_orb = make_rdm1(orb, mo_occ[i][k])
                    # write rdm1_orb as cube file
                    if cube:
                        if mol.spin == 0:
                            write_cube(mol, rdm1_orb * 2., 'rdm1_{:d}'.format(j))
                        else:
                            write_cube(mol, rdm1_orb, 'spin_{:s}_rdm1_{:d}'.format('a' if i == 0 else 'b', j))
                    # energy or dipole from individual orbitals
                    if prop_type == 'energy':
                        prop_orb[i][j] += _trace(kin + nuc, rdm1_orb)
                        prop_orb[i][j] += _trace(vj[0] + vj[1] - vk[i], rdm1_orb, scaling = .5)
                    elif prop_type == 'dipole':
                        prop_orb[i][j] -= _trace(ao_dip, rdm1_orb)
                    # additional xc energy contribution
                    if prop_type == 'energy' and isinstance(mf, dft.rks.KohnShamDFT):
                        # orbital-specific rho
                        rho_orb = numint.eval_rho(mol, ao_value, rdm1_orb, xctype=xc_type)
                        # energy from individual orbitals
                        prop_orb[i][j] += _e_xc(eps_xc, mf.grids.weights, rho_orb)
                # closed-shell system
                if mol.spin == 0:
                    prop_orb[i+1] = prop_orb[i]
                    break

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


def _h_core(mol: gto.Mole) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        this function returns the components of the core hamiltonian
        """
        coords = mol.atom_coords()
        charges = mol.atom_charges()
        kin = mol.intor_symmetric('int1e_kin')
        sub_nuc = np.zeros([mol.natm, mol.nao_nr(), mol.nao_nr()], dtype=np.float64)
        for k in range(mol.natm):
            with mol.with_rinv_origin(coords[k]):
                sub_nuc[k] = mol.intor('int1e_rinv') * -charges[k]
        nuc = np.sum(sub_nuc, axis=0)

        return kin, nuc, sub_nuc


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


def _trace(op: np.ndarray, rdm1: np.ndarray, scaling: float = 1.) -> float:
        """
        this function returns the trace between an operator and an rdm1
        """
        if op.ndim == 2:
            return np.einsum('ij,ij', op, rdm1) * scaling
        else:
            return np.einsum('xij,ij->x', op, rdm1) * scaling


def _e_xc(eps_xc: np.ndarray, grid_weights: np.ndarray, rho: np.ndarray) -> float:
        """
        this function returns a contribution to the exchange-correlation energy from given rmd1 (via rho)
        """
        return np.einsum('i,i,i->', eps_xc, rho if rho.ndim == 1 else rho[0], grid_weights)


