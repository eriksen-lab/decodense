#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
orbitals module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import gto, scf, dft, lo
from typing import List, Tuple, Union

from .tools import make_rdm1


def loc_orbs(mol: gto.Mole, mo_coeff: Tuple[np.ndarray, np.ndarray], s: np.ndarray, variant: str) -> np.ndarray:
        """
        this function returns a set of localized MOs of a specific variant
        """
        # init localizer
        if variant == 'pm':
            for i, nspin in enumerate((mol.nalpha, mol.nbeta)):
                # pipek-mezey procedure
                loc_core = lo.PM(mol, mo_coeff[i][:, :mol.ncore])
                loc_val = lo.PM(mol, mo_coeff[i][:, mol.ncore:nspin])
                # localize core and valence occupied orbitals
                mo_coeff[i][:, :mol.ncore] = loc_core.kernel()
                mo_coeff[i][:, mol.ncore:nspin] = loc_val.kernel()
                # closed-shell system
                if mol.spin == 0:
                    mo_coeff[i+1][:, :nspin] = mo_coeff[i][:, :nspin]
                    break
        elif 'ibo' in variant:
            for i, nspin in enumerate((mol.nalpha, mol.nbeta)):
                # IAOs
                iao_core = lo.iao.iao(mol, mo_coeff[i][:, :mol.ncore])
                iao_val = lo.iao.iao(mol, mo_coeff[i][:, mol.ncore:nspin])
                # orthogonalize IAOs
                iao_core = lo.vec_lowdin(iao_core, s)
                iao_val = lo.vec_lowdin(iao_val, s)
                # IBOs
                mo_coeff[i][:, :mol.ncore] = lo.ibo.ibo(mol, mo_coeff[i][:, :mol.ncore], \
                                                        iaos=iao_core, exponent=int(variant[-1]), verbose=0)
                mo_coeff[i][:, mol.ncore:nspin] = lo.ibo.ibo(mol, mo_coeff[i][:, mol.ncore:nspin], \
                                                             iaos=iao_val, exponent=int(variant[-1]), verbose=0)
                # closed-shell system
                if mol.spin == 0:
                    mo_coeff[i+1][:, :nspin] = mo_coeff[i][:, :nspin]
                    break

        return mo_coeff


def assign_rdm1s(mol: gto.Mole, s: np.ndarray, mo_coeff: Tuple[np.ndarray, np.ndarray], \
                 mo_occ: np.ndarray, pop: str, thres: float) -> Tuple[List[List[np.ndarray]], np.ndarray]:
        """
        this function returns a list of repetitive center indices and an array of unique charge centres
        """
        # init charge_centres array
        cent = [np.zeros([mol.nalpha, 2], dtype=np.int), np.zeros([mol.nbeta, 2], dtype=np.int)]

        for i, nspin in enumerate((mol.nalpha, mol.nbeta)):
            for j in range(nspin):
                # get orbital
                orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], 1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                # charge centres of rdm1_orb
                cent[i][j] = _charge_centres(mol, s, orb, rdm1_orb, pop, thres)
            # closed-shell system
            if mol.spin == 0:
                cent[i+1] = cent[i]
                break

        # unique centres
        cent_unique = np.array([np.unique(cent[i], axis=0) for i in range(2)])
        # repetitive centres
        rep_idx = [[np.where((cent[i] == j).all(axis=1))[0] for j in cent_unique[i]] for i in range(2)]

        return rep_idx, cent_unique


def atom_part(mol: gto.Mole, prop_type: str, prop_old: np.ndarray, cent: np.ndarray) -> np.ndarray:
        """
        this function collects results based on the involved atoms
        """
        # init prop_new
        if prop_type == 'energy':
            prop_new = np.zeros(mol.natm, dtype=np.float64)
        elif prop_type == 'dipole':
            prop_new = np.zeros([mol.natm, 3], dtype=np.float64)

        # collect contributions
        for i in range(2):
            for a, (j, k) in enumerate(cent[i]):
                if j == k:
                    # contribution from core orbital or lone pair
                    prop_new[j] += prop_old[i][a]
                else:
                    # contribution from valence orbital
                    prop_new[j] += prop_old[i][a] / 2.
                    prop_new[k] += prop_old[i][a] / 2.

        return prop_new


def _charge_centres(mol: gto.Mole, s: np.ndarray, orb: np.ndarray, \
                    rdm1: np.ndarray, pop: str, thres: float) -> np.ndarray:
        """
        this function returns a single atom/pair of atoms onto which a given MO is assigned
        """
        if pop == 'mulliken':
            # traditional mulliken charges
            charges = _mulliken_charges(mol, s, rdm1)
        elif pop == 'iao':
            # base mulliken charges on IAOs
            iao = lo.iao.iao(mol, orb)
            iao = lo.vec_lowdin(iao, s)
            orb_iao = np.einsum('ki,kl,lj', iao, s, orb)
            rdm1_iao = np.einsum('ip,jp->ij', orb_iao, orb_iao) * 2.
            pmol = mol.copy()
            pmol.build(False, False, basis='minao')
            # charges
            charges = _mulliken_charges(pmol, np.eye(pmol.nao_nr()), rdm1_iao)

        # get sorted indices
        max_idx = np.argsort(charges)[::-1]

        if np.abs(charges[max_idx[0]]) / np.abs((charges[max_idx[0]] + charges[max_idx[1]])) > thres:
            # core orbital or lone pair
            return np.sort(np.array([max_idx[0], max_idx[0]], dtype=np.int))
        else:
            # valence orbitals
            return np.sort(np.array([max_idx[0], max_idx[1]], dtype=np.int))


def _mulliken_charges(mol: gto.Mole, s: np.ndarray, rdm1: np.ndarray) -> np.ndarray:
        """
        this function returns the mulliken charges on the individual atoms
        """
        # mulliken population matrix
        pop = np.einsum('ij,ji->i', rdm1, s).real
        # init charges
        charges = np.zeros(mol.natm)

        # loop over AOs
        for i, k in enumerate(mol.ao_labels(fmt=None)):
            charges[k[0]] += pop[i]

        return charges


