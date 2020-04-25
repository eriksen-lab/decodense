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
from typing import List, Tuple, Dict, Union, Any

from .tools import make_rdm1


def loc_orbs(mol: gto.Mole, mo_coeff: Tuple[np.ndarray, np.ndarray], s: np.ndarray, variant: str) -> np.ndarray:
        """
        this function returns a set of localized MOs of a specific variant
        """
        # init localizer
        if variant == 'boys':
            for i, nspin in enumerate((mol.nalpha, mol.nbeta)):
                # pipek-mezey procedure
                loc_core = lo.Boys(mol, mo_coeff[i][:, :mol.ncore])
                loc_val = lo.Boys(mol, mo_coeff[i][:, mol.ncore:nspin])
                loc_core.conv_tol = loc_val.conv_tol = 1.e-10
                # localize core and valence occupied orbitals
                mo_coeff[i][:, :mol.ncore] = loc_core.kernel()
                mo_coeff[i][:, mol.ncore:nspin] = loc_val.kernel()
                # closed-shell system
                if mol.spin == 0:
                    mo_coeff[i+1][:, :nspin] = mo_coeff[i][:, :nspin]
                    break
        elif variant == 'pm':
            for i, nspin in enumerate((mol.nalpha, mol.nbeta)):
                # pipek-mezey procedure
                loc_core = lo.PM(mol, mo_coeff[i][:, :mol.ncore])
                loc_val = lo.PM(mol, mo_coeff[i][:, mol.ncore:nspin])
                loc_core.conv_tol = loc_val.conv_tol = 1.e-10
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
                mo_coeff[i][:, :mol.ncore] = lo.ibo.ibo(mol, mo_coeff[i][:, :mol.ncore], iaos=iao_core, \
                                                        grad_tol = 1.e-10, exponent=int(variant[-1]), verbose=0)
                mo_coeff[i][:, mol.ncore:nspin] = lo.ibo.ibo(mol, mo_coeff[i][:, mol.ncore:nspin], iaos=iao_val, \
                                                             grad_tol = 1.e-10, exponent=int(variant[-1]), verbose=0)
                # closed-shell system
                if mol.spin == 0:
                    mo_coeff[i+1][:, :nspin] = mo_coeff[i][:, :nspin]
                    break

        return mo_coeff


def assign_rdm1s(mol: gto.Mole, s: np.ndarray, mo_coeff: Tuple[np.ndarray, np.ndarray], \
                 mo_occ: np.ndarray, pop: str) -> List[np.ndarray]:
        """
        this function returns a list of population weights of each spin-orbital on the individual atoms
        """
        # init charge weights array
        weights = [np.zeros([mol.nalpha, mol.natm], dtype=np.float64), np.zeros([mol.nbeta, mol.natm], dtype=np.float64)]

        if pop == 'iao':
            # mol object projected into minao basis
            pmol = mol.copy()
            pmol.build(False, False, basis='minao')
        elif pop == 'meta-lowdin':
            # transformation coefficients to local ao basis 
            c = lo.orth.restore_ao_character(mol)
            orth_coeff = lo.orth.orth_ao(mol, 'meta_lowdin', pre_orth_ao=c, s=s)
            c_inv = np.einsum('ki,kj->ij', orth_coeff, s)

        for i, nspin in enumerate((mol.nalpha, mol.nbeta)):
            for j in range(nspin):
                # get orbital
                orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], 1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                # charge centre weights of rdm1_orb
                if pop == 'mulliken':
                    weights[i][j] = _charge_weights(mol, s, orb, rdm1_orb, pop)
                elif pop == 'iao':
                    weights[i][j] = _charge_weights(mol, s, orb, rdm1_orb, pop, pmol=pmol)
                elif pop == 'meta-lowdin':
                    weights[i][j] = _charge_weights(mol, s, orb, rdm1_orb, pop, c_inv=c_inv)
            # closed-shell system
            if mol.spin == 0:
                weights[i+1] = weights[i]
                break

        return weights


def partition(mol: gto.Mole, prop_type: str, prop_old: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        this function collects results based on the involved atoms
        """
        # init prop_new
        if prop_type == 'energy':
            prop_new = np.zeros(mol.natm, dtype=np.float64)
        elif prop_type == 'dipole':
            prop_new = np.zeros([mol.natm, 3], dtype=np.float64)

        # sum up contributions
        for i in range(2):
            prop_new += np.einsum('i,ik', prop_old[i], weights[i])
            # closed-shell system
            if mol.spin == 0:
                prop_new *= 2.
                break

        return prop_new


def _charge_weights(mol: gto.Mole, s: np.ndarray, orb: np.ndarray, \
                    rdm1: np.ndarray, pop: str, **kwargs: Union[gto.Mole, np.ndarray]) -> np.ndarray:
        """
        this function returns an array of weights based an atomic charges
        """
        if pop == 'mulliken':
            # traditional mulliken charges
            charges = _mulliken_charges(mol, s, rdm1)
        elif pop == 'iao':
            # base mulliken charges on IAOs (JCTC, 9, 4834 (2013))
            iao = lo.iao.iao(mol, orb)
            iao = lo.vec_lowdin(iao, s)
            orb_iao = np.einsum('ki,kl,lj->ij', iao, s, orb)
            rdm1_iao = np.einsum('ip,jp->ij', orb_iao, orb_iao)
            # charges
            charges = _mulliken_charges(kwargs['pmol'], np.eye(rdm1_iao.shape[0]), rdm1_iao)
        elif pop == 'meta-lowdin':
            # base mulliken charges on meta-Lowdin atomic orbitals (JCTC, 10, 3784 (2014))	
            rdm1_meta = np.einsum('ik,kl,jl->ij', kwargs['c_inv'], rdm1, kwargs['c_inv'])
            # charges
            charges = _mulliken_charges(mol, np.eye(rdm1_meta.shape[0]), rdm1_meta)

        return charges


def _mulliken_charges(mol: gto.Mole, s: np.ndarray, rdm1: np.ndarray) -> np.ndarray:
        """
        this function returns the mulliken charges on the individual atoms
        """
        # mulliken population matrix
        pop = np.einsum('ij,ji->i', rdm1, s)
        # init charges
        charges = np.zeros(mol.natm)

        # loop over AOs
        for i, k in enumerate(mol.ao_labels(fmt=None)):
            charges[k[0]] += pop[i]

        return charges


