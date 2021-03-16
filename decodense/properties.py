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
from pyscf import gto, scf, dft, lo
from pyscf.dft import numint
from pyscf import tools as pyscf_tools
from typing import List, Tuple, Dict, Union, Any

from .tools import make_rdm1, write_cube
from .decomp import PROP_KEYS


def prop_tot(mol: gto.Mole, mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
             mo_coeff: Tuple[np.ndarray, np.ndarray], mo_occ: Tuple[np.ndarray, np.ndarray], \
             ref: str, pop: str, prop_type: str, part: str, cube: bool, \
             **kwargs: Any) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        this function returns atom-decomposed mean-field properties
        """
        # dft logical
        dft_calc = isinstance(mf, dft.rks.KohnShamDFT)

        # ao dipole integrals with specified gauge origin
        if prop_type == 'dipole':
            with mol.with_common_origin(kwargs['dipole_origin']):
                ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        else:
            ao_dip = None

        # compute total 1-RDM (AO basis)
        rdm1_tot = np.array([make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])])

        # mol object projected into minao basis
        if pop == 'iao':
            pmol = lo.iao.reference_mol(mol)
        else:
            pmol = mol

        # effective atomic charges
        if 'weights' in kwargs:
            weights = kwargs['weights']
            charge_atom = pmol.atom_charges() - np.sum(weights[0] + weights[1], axis=0)
        else:
            charge_atom = 0.

        # nuclear repulsion property
        if prop_type == 'energy':
            prop_nuc_rep = _e_nuc(pmol)
        elif prop_type == 'dipole':
            prop_nuc_rep = _dip_nuc(pmol)

        # core hamiltonian
        kin, nuc, sub_nuc = _h_core(mol)
        # fock potential
        vj, vk = scf.hf.get_jk(mol, rdm1_tot)

        # calculate xc energy density
        if dft_calc:
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
        else:
            xc_type = ''
            ao_value = eps_xc = None

        if part in ['atoms', 'eda']:

            # init atom-specific energy or dipole arrays
            if prop_type == 'energy':
                prop = {prop_key: np.zeros(pmol.natm, dtype=np.float64) for prop_key in PROP_KEYS}
            elif prop_type == 'dipole':
                prop = {prop_key: np.zeros([pmol.natm, 3], dtype=np.float64) for prop_key in PROP_KEYS[-2:]}

            # loop over atoms
            for k in range(pmol.natm):
                if part == 'atoms':
                    res = prop_atom(mol, rdm1_tot, mo_coeff, mo_occ, \
                                    weights, dft_calc, cube, prop_type, \
                                    mf, vj, vk, kin, nuc, sub_nuc, prop_nuc_rep, \
                                    ao_value, eps_xc, xc_type, ao_dip, k)
                elif part == 'eda':
                    res = prop_eda(mol, rdm1_tot, mo_coeff, dft_calc, prop_type, \
                                   mf, vj, vk, kin, nuc, sub_nuc, prop_nuc_rep, \
                                   ao_value, eps_xc, xc_type, ao_dip, k)
                for key, val in res.items():
                    prop[key][k] = val

        else: # bonds

            # get rep_idx
            rep_idx = kwargs['rep_idx']

            # init orbital-specific energy or dipole array
            if prop_type == 'energy':
                prop_coul = prop_exch = prop_kin = prop_rdm_att = prop_xc = prop_nuc_att = None
                prop_el = [np.zeros(len(rep_idx[0]), dtype=np.float64), np.zeros(len(rep_idx[1]), dtype=np.float64)]
                prop_struct = prop_nuc_rep
            elif prop_type == 'dipole':
                prop_coul = prop_exch = prop_kin = prop_rdm_att = prop_xc = prop_nuc_att = None
                prop_el = [np.zeros([len(rep_idx[0]), 3], dtype=np.float64), np.zeros([len(rep_idx[1]), 3], dtype=np.float64)]
                prop_struct = prop_nuc_rep

            # loop over spins
            for i in range(2):
                # loop over spin-orbitals
                for j, k in enumerate(rep_idx[i]):
                    # get orbital(s)
                    orb = mo_coeff[i][:, k].reshape(mo_coeff[i].shape[0], -1)
                    # orbital-specific rdm1
                    rdm1_orb = make_rdm1(orb, mo_occ[i][k])
                    # total energy or dipole moment associated with given spin-orbital
                    if prop_type == 'energy':
                        prop_el[i][j] += _trace(np.sum(vj, axis=0) - vk[i], rdm1_orb, scaling = .5)
                        prop_el[i][j] += _trace(kin + nuc, rdm1_orb)
                    elif prop_type == 'dipole':
                        prop_el[i][j] -= _trace(ao_dip, rdm1_orb)
                    # additional xc energy contribution
                    if prop_type == 'energy' and dft_calc:
                        # orbital-specific rho
                        rho_orb = numint.eval_rho(mol, ao_value, rdm1_orb, xctype=xc_type)
                        # energy from individual orbitals
                        prop_el[i][j] += _e_xc(eps_xc, mf.grids.weights, rho_orb)
                    # write rdm1_orb as cube file
                    if cube:
                        if mol.spin == 0:
                            write_cube(pmol, rdm1_orb * 2., 'rdm1_{:d}'.format(j))
                        else:
                            write_cube(pmol, rdm1_orb, 'spin_{:s}_rdm1_{:d}'.format('a' if i == 0 else 'b', j))
                # closed-shell system
                if ref == 'restricted' and mol.spin == 0:
                    prop_el[i+1] = prop_el[i]
                    break

        return {**prop, 'charge_atom': charge_atom}


def prop_atom(mol: gto.Mole, rdm1_tot: np.ndarray, mo_coeff: np.ndarray, mo_occ: np.ndarray, \
              weights: List[np.ndarray], dft_calc: bool, cube: bool, prop_type: str, \
              mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], vj: np.ndarray, vk: np.ndarray, \
              kin: np.ndarray, nuc: np.ndarray, sub_nuc: np.ndarray, prop_nuc_rep: np.ndarray, \
              ao_value: Union[None, np.ndarray], eps_xc: Union[None, np.ndarray], \
              xc_type: str, ao_dip: Union[None, np.ndarray], orb_idx: int) -> Dict[str, Any]:
        """
        this function returns atom-wise energy/dipole contributions
        """
        # init results
        if prop_type == 'energy':
            res = {prop_key: 0. for prop_key in PROP_KEYS}
        elif prop_type == 'dipole':
            res = {prop_key: np.zeros(3, dtype=np.float64) for prop_key in PROP_KEYS[-2:]}
        # atom-specific rdm1
        rdm1_atom = np.zeros_like(rdm1_tot)
        # loop over spins
        for i, spin_mo in enumerate((mol.alpha, mol.beta)):
            # loop over spin-orbitals
            for m, j in enumerate(spin_mo):
                # get orbital(s)
                orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], -1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                # weighted contribution to rdm1_atom
                rdm1_atom[i] += rdm1_orb * weights[i][m][orb_idx]
            # coulumb & exchange energy associated with given atom
            if prop_type == 'energy':
                res['coul'] += _trace(np.sum(vj, axis=0), rdm1_atom[i], scaling = .5)
                res['exch'] -= _trace(vk[i], rdm1_atom[i], scaling = .5)
        # kinetic & nuclear/rdm attraction energies associated with given atom
        if prop_type == 'energy':
            res['kin'] = _trace(kin, np.sum(rdm1_atom, axis=0))
            res['rdm_att'] = _trace(nuc, np.sum(rdm1_atom, axis=0), scaling = .5)
            res['nuc_att'] = _trace(sub_nuc[orb_idx], np.sum(rdm1_tot, axis=0), scaling = .5)
            # additional xc energy contribution
            if dft_calc:
                # atom-specific rho
                rho_atom = numint.eval_rho(mol, ao_value, np.sum(rdm1_atom, axis=0), xctype=xc_type)
                # energy from individual atoms
                res['xc'] = _e_xc(eps_xc, mf.grids.weights, rho_atom)
        # sum up electronic and structural contributions
        if prop_type == 'energy':
            res['el'] = res['coul'] + res['exch'] + res['kin'] + res['rdm_att'] + res['xc']
            res['struct'] = res['nuc_att'] + prop_nuc_rep[orb_idx]
        elif prop_type == 'dipole':
            res['el'] = -_trace(ao_dip, np.sum(rdm1_atom, axis=0))
            res['struct'] = prop_nuc_rep[orb_idx]
        # write rdm1_atom as cube file
        if cube:
            write_cube(mol, np.sum(rdm1_atom, axis=0), f'atom_{mol.atom_symbol(orb_idx).lower():s}_rdm1_{orb_idx:d}')
        return res


def prop_eda(mol: gto.Mole, rdm1_tot: np.ndarray, mo_coeff: np.ndarray, dft_calc: bool, prop_type: str, \
             mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], vj: np.ndarray, vk: np.ndarray, \
             kin: np.ndarray, nuc: np.ndarray, sub_nuc: np.ndarray, prop_nuc_rep: np.ndarray, \
             ao_value: Union[None, np.ndarray], eps_xc: Union[None, np.ndarray], \
             xc_type: str, ao_dip: Union[None, np.ndarray], orb_idx: int) -> Dict[str, Any]:
        """
        this function returns EDA energy/dipole contributions
        """
        # init results
        if prop_type == 'energy':
            res = {prop_key: 0. for prop_key in PROP_KEYS}
        elif prop_type == 'dipole':
            res = {prop_key: np.zeros(3, dtype=np.float64) for prop_key in PROP_KEYS[-2:]}
        # get AOs on atom k
        select = np.where([atom[0] == orb_idx for atom in mol.ao_labels(fmt=None)])[0]
        # common energy contributions associated with given atom
        if prop_type == 'energy':
            # loop over spins
            for i, spin_mo in enumerate((mol.alpha, mol.beta)):
                res['coul'] += _trace(np.sum(vj, axis=0)[select], rdm1_tot[i][select], scaling = .5)
                res['exch'] -= _trace(vk[i][select], rdm1_tot[i][select], scaling = .5)
            res['kin'] = _trace(kin[select], np.sum(rdm1_tot, axis=0)[select])
            res['rdm_att'] = _trace(nuc[select], np.sum(rdm1_tot, axis=0)[select], scaling = .5)
            res['nuc_att'] = _trace(sub_nuc[orb_idx], np.sum(rdm1_tot, axis=0), scaling = .5)
            # additional xc energy contribution
            if dft_calc:
                # atom-specific rho
                rho_atom = numint.eval_rho(mol, ao_value, np.sum(rdm1_tot, axis=0), xctype=xc_type, idx=select)
                # energy from individual atoms
                res['xc'] = _e_xc(eps_xc, mf.grids.weights, rho_atom)
        # sum up electronic and structural contributions
        if prop_type == 'energy':
            res['el'] = res['coul'] + res['exch'] + res['kin'] + res['rdm_att'] + res['xc']
            res['struct'] = res['nuc_att'] + prop_nuc_rep[orb_idx]
        elif prop_type == 'dipole':
            res['el'] = -_trace(ao_dip[:, select], np.sum(rdm1_tot, axis=0)[select]) # type:ignore
            res['struct'] = prop_nuc_rep[orb_idx]
        return res


def _e_nuc(mol: gto.Mole) -> np.ndarray:
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


def _dip_nuc(mol: gto.Mole) -> np.ndarray:
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
        # kinetic integrals
        kin = mol.intor_symmetric('int1e_kin')
        # coordinates and charges of nuclei
        coords = mol.atom_coords()
        charges = mol.atom_charges()
        # individual atomic potentials
        sub_nuc = np.zeros([mol.natm, mol.nao_nr(), mol.nao_nr()], dtype=np.float64)
        for k in range(mol.natm):
            with mol.with_rinv_origin(coords[k]):
                sub_nuc[k] = mol.intor('int1e_rinv') * -charges[k]
        # total nuclear potential
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


def _trace(op: np.ndarray, rdm1: np.ndarray, scaling: float = 1.) -> Union[float, np.ndarray]:
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


