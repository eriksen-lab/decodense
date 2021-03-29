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
import multiprocessing as mp
from itertools import starmap
from pyscf import gto, scf, dft, lo, lib
from pyscf.dft import numint
from pyscf import tools as pyscf_tools
from typing import List, Tuple, Dict, Union, Any

from .tools import make_rdm1
from .decomp import PROP_KEYS


def prop_tot(mol: gto.Mole, mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
             mo_coeff: Tuple[np.ndarray, np.ndarray], mo_occ: Tuple[np.ndarray, np.ndarray], \
             ref: str, pop: str, prop_type: str, part: str, multiproc: bool, \
             **kwargs: Any) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        this function returns atom-decomposed mean-field properties
        """
        # declare nested kernel functions in global scope
        global prop_atom
        global prop_eda
        global prop_bonds

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
                c0_tot, c1_tot = _make_rho_int(ao_value, rdm1_tot[0] * 2., xc_type)
                rho_tot = _make_rho(c0_tot, c1_tot, ao_value, xc_type)
            else:
                c0_tot, c1_tot = zip(_make_rho_int(ao_value, rdm1_tot[0], xc_type), \
                                     _make_rho_int(ao_value, rdm1_tot[1], xc_type))
                rho_tot = (_make_rho(c0_tot[0], c1_tot[0], ao_value, xc_type), \
                           _make_rho(c0_tot[1], c1_tot[1], ao_value, xc_type))
                c0_tot = np.sum(c0_tot, axis=0)
                if c1_tot[0] is not None:
                    c1_tot = np.sum(c1_tot, axis=0)
            # evaluate xc energy density
            eps_xc = dft.libxc.eval_xc(mf.xc, rho_tot, spin=0 if isinstance(rho_tot, np.ndarray) else -1)[0]
            # grid weights
            grid_weights = mf.grids.weights
        else:
            xc_type = ''
            grid_weights = ao_value = eps_xc = None

        # dimensions
        alpha = mol.alpha
        beta = mol.beta
        if part == 'eda':
            ao_labels = mol.ao_labels(fmt=None)

        def prop_atom(atom_idx: int) -> Dict[str, Any]:
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
                for i, spin_mo in enumerate((alpha, beta)):
                    # loop over spin-orbitals
                    for m, j in enumerate(spin_mo):
                        # get orbital(s)
                        orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], -1)
                        # orbital-specific rdm1
                        rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                        # weighted contribution to rdm1_atom
                        rdm1_atom[i] += rdm1_orb * weights[i][m][atom_idx]
                    # coulumb & exchange energy associated with given atom
                    if prop_type == 'energy':
                        res['coul'] += _trace(np.sum(vj, axis=0), rdm1_atom[i], scaling = .5)
                        res['exch'] -= _trace(vk[i], rdm1_atom[i], scaling = .5)
                # common energy contributions associated with given atom
                if prop_type == 'energy':
                    res['kin'] = _trace(kin, np.sum(rdm1_atom, axis=0))
                    res['rdm_att'] = _trace(nuc, np.sum(rdm1_atom, axis=0), scaling = .5)
                    res['nuc_att'] = _trace(sub_nuc[atom_idx], np.sum(rdm1_tot, axis=0), scaling = .5)
                    # additional xc energy contribution
                    if dft_calc:
                        # atom-specific rho
                        c0_atom, c1_atom = _make_rho_int(ao_value, np.sum(rdm1_atom, axis=0), xc_type)
                        rho_atom = _make_rho(c0_atom, c1_atom, ao_value, xc_type)
                        # energy from individual atoms
                        res['xc'] = _e_xc(eps_xc, grid_weights, rho_atom)
                # sum up electronic and structural contributions
                if prop_type == 'energy':
                    res['el'] = res['coul'] + res['exch'] + res['kin'] + res['rdm_att'] + res['xc']
                    res['struct'] = res['nuc_att'] + prop_nuc_rep[atom_idx]
                elif prop_type == 'dipole':
                    res['el'] = -_trace(ao_dip, np.sum(rdm1_atom, axis=0))
                    res['struct'] = prop_nuc_rep[atom_idx]
                return res

        def prop_eda(atom_idx: int) -> Dict[str, Any]:
                """
                this function returns EDA energy/dipole contributions
                """
                # init results
                if prop_type == 'energy':
                    res = {prop_key: 0. for prop_key in PROP_KEYS}
                elif prop_type == 'dipole':
                    res = {prop_key: np.zeros(3, dtype=np.float64) for prop_key in PROP_KEYS[-2:]}
                # get AOs on atom k
                select = np.where([atom[0] == atom_idx for atom in ao_labels])[0]
                # common energy contributions associated with given atom
                if prop_type == 'energy':
                    # loop over spins
                    for i, _ in enumerate((alpha, beta)):
                        res['coul'] += _trace(np.sum(vj, axis=0)[select], rdm1_tot[i][select], scaling = .5)
                        res['exch'] -= _trace(vk[i][select], rdm1_tot[i][select], scaling = .5)
                    res['kin'] = _trace(kin[select], np.sum(rdm1_tot, axis=0)[select])
                    res['rdm_att'] = _trace(nuc[select], np.sum(rdm1_tot, axis=0)[select], scaling = .5)
                    res['nuc_att'] = _trace(sub_nuc[atom_idx], np.sum(rdm1_tot, axis=0), scaling = .5)
                    # additional xc energy contribution
                    if dft_calc:
                        # atom-specific rho
                        rho_atom = _make_rho(c0_tot[:, select], \
                                             c1_tot if c1_tot is None else c1_tot[:, :, select], \
                                             ao_value[:, :, select], xc_type)
                        # energy from individual atoms
                        res['xc'] = _e_xc(eps_xc, grid_weights, rho_atom)
                # sum up electronic and structural contributions
                if prop_type == 'energy':
                    res['el'] = res['coul'] + res['exch'] + res['kin'] + res['rdm_att'] + res['xc']
                    res['struct'] = res['nuc_att'] + prop_nuc_rep[atom_idx]
                elif prop_type == 'dipole':
                    res['el'] = -_trace(ao_dip[:, select], np.sum(rdm1_tot, axis=0)[select])
                    res['struct'] = prop_nuc_rep[atom_idx]
                return res

        def prop_bonds(spin_idx: int, orb_idx: int) -> Dict[str, Any]:
                """
                this function returns bond-wise energy/dipole contributions
                """
                # init results
                res = {'el': 0. if prop_type == 'energy' else np.zeros(3, dtype=np.float64)}
                # get orbital(s)
                orb = mo_coeff[spin_idx][:, orb_idx].reshape(mo_coeff[spin_idx].shape[0], -1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mo_occ[spin_idx][orb_idx])
                # total energy or dipole moment associated with given spin-orbital
                if prop_type == 'energy':
                    res['el'] = _trace(np.sum(vj, axis=0) - vk[spin_idx], rdm1_orb, scaling = .5)
                    res['el'] += _trace(kin + nuc, rdm1_orb)
                    # additional xc energy contribution
                    if dft_calc:
                        # orbital-specific rho
                        c0_orb, c1_orb = _make_rho_int(ao_value, rdm1_orb, xc_type)
                        rho_orb = _make_rho(c0_orb, c1_orb, ao_value, xc_type)
                        # energy from individual orbitals
                        res['el'] += _e_xc(eps_xc, grid_weights, rho_orb)
                elif prop_type == 'dipole':
                    res['el'] = -_trace(ao_dip, rdm1_orb)
                return res

        # perform decomposition
        if part in ['atoms', 'eda']:
            # init atom-specific energy or dipole arrays
            if prop_type == 'energy':
                prop = {prop_key: np.zeros(pmol.natm, dtype=np.float64) for prop_key in PROP_KEYS}
            elif prop_type == 'dipole':
                prop = {prop_key: np.zeros([pmol.natm, 3], dtype=np.float64) for prop_key in PROP_KEYS[-2:]}
            # domain
            domain = np.arange(pmol.natm)
            # execute kernel
            if multiproc:
                n_threads = min(domain.size, lib.num_threads())
                with mp.Pool(processes=n_threads) as pool:
                    res = pool.map(prop_atom if part == 'atoms' else prop_eda, domain)
            else:
                res = list(map(prop_atom if part == 'atoms' else prop_eda, domain))
            # collect results
            for k, r in enumerate(res):
                for key, val in r.items():
                    prop[key][k] = val
        else: # bonds
            # get rep_idx
            rep_idx = kwargs['rep_idx']
            # init orbital-specific energy or dipole array
            if prop_type == 'energy':
                prop = {'el': [np.zeros(len(rep_idx[0]), dtype=np.float64), np.zeros(len(rep_idx[1]), dtype=np.float64)], \
                        'struct': prop_nuc_rep}
            elif prop_type == 'dipole':
                prop = {'el': [np.zeros([len(rep_idx[0]), 3], dtype=np.float64), np.zeros([len(rep_idx[1]), 3], dtype=np.float64)], \
                        'struct': prop_nuc_rep}
            # domain
            domain = np.array([(i, j) for i, _ in enumerate((mol.alpha, mol.beta)) for j in rep_idx[i]])
            # execute kernel
            if multiproc:
                n_threads = min(domain.size, lib.num_threads())
                with mp.Pool(processes=n_threads) as pool:
                    res = pool.starmap(prop_bonds, domain)
            else:
                res = list(starmap(prop_bonds, domain))
            # collect results
            for k, r in enumerate(res):
                for key, val in r.items():
                    prop[key][0 if k < len(rep_idx[0]) else 1][k % len(rep_idx[0])] = val
        return {**prop, 'charge_atom': charge_atom}


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


def _make_rho_int(ao_value: np.ndarray, \
                  rdm1: np.ndarray, xc_type: str) -> Tuple[np.ndarray, Union[None, np.ndarray]]:
        """
        this function returns the rho intermediates (c0, c1) needed in _make_rho()
        (adpated from: dft/numint.py:eval_rho() in PySCF)
        """
        # determine dimensions based on xctype
        xctype = xc_type.upper()
        if xctype == 'LDA' or xctype == 'HF':
            ngrids, nao = ao_value.shape
        else:
            ngrids, nao = ao_value[0].shape
        # compute rho intermediate based on xctype
        if xctype == 'LDA' or xctype == 'HF':
            c0 = np.dot(ao_value, rdm1)
            c1 = None
        elif xctype in ('GGA', 'NLC'):
            c0 = np.dot(ao_value[0], rdm1)
            c1 = None
        else: # meta-GGA
            c0 = np.dot(ao_value[0], rdm1)
            c1 = np.empty((3, ngrids, nao), dtype=np.float64)
            for i in range(1, 4):
                c1[i-1] = np.dot(ao_value[i], rdm1.T)
        return c0, c1


def _make_rho(c0: np.ndarray, c1: np.ndarray, \
              ao_value: np.ndarray, xc_type: str) -> np.ndarray:
        """
        this function returns rho
        (adpated from: dft/numint.py:eval_rho() in PySCF)
        """
        # determine dimensions based on xctype
        xctype = xc_type.upper()
        if xctype == 'LDA' or xctype == 'HF':
            ngrids = ao_value.shape[0]
        else:
            ngrids = ao_value[0].shape[0]
        # compute rho intermediate based on xctype
        if xctype == 'LDA' or xctype == 'HF':
            rho = np.einsum('pi,pi->p', ao_value, c0)
        elif xctype in ('GGA', 'NLC'):
            rho = np.empty((4, ngrids), dtype=np.float64)
            rho[0] = np.einsum('pi,pi->p', c0, ao_value[0])
            for i in range(1, 4):
                rho[i] = np.einsum('pi,pi->p', c0, ao_value[i]) * 2.
        else: # meta-GGA
            rho = np.empty((6, ngrids), dtype=np.float64)
            rho[0] = np.einsum('pi,pi->p', ao_value[0], c0)
            rho[5] = 0.
            for i in range(1, 4):
                rho[i] = np.einsum('pi,pi->p', c0, ao_value[i]) * 2.
                rho[5] += np.einsum('pi,pi->p', c1[i-1], ao_value[i])
            XX, YY, ZZ = 4, 7, 9
            ao_value_2 = ao_value[XX] + ao_value[YY] + ao_value[ZZ]
            rho[4] = np.einsum('pi,pi->p', c0, ao_value_2)
            rho[4] += rho[5]
            rho[4] *= 2.
            rho[5] *= .5
        return rho


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


