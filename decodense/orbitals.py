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
        if variant == 'fb':
            for i, nspin in enumerate((mol.nalpha, mol.nbeta)):
                # foster-boys procedure
                loc = lo.Boys(mol, mo_coeff[i][:, :nspin])
                loc.conv_tol = 1.e-10
                # FB MOs
                mo_coeff[i][:, :nspin] = loc.kernel()
                # closed-shell system
                if mol.spin == 0:
                    mo_coeff[i+1][:, :nspin] = mo_coeff[i][:, :nspin]
                    break
        elif variant == 'pm':
            for i, nspin in enumerate((mol.nalpha, mol.nbeta)):
                # pipek-mezey procedure
                loc = lo.PM(mol, mo_coeff[i][:, :nspin])
                loc.conv_tol = 1.e-10
                # PM MOs
                mo_coeff[i][:, :nspin] = loc.kernel()
                # closed-shell system
                if mol.spin == 0:
                    mo_coeff[i+1][:, :nspin] = mo_coeff[i][:, :nspin]
                    break
        elif 'ibo' in variant:
            for i, nspin in enumerate((mol.nalpha, mol.nbeta)):
                # orthogonalized IAOs
                iao = lo.iao.iao(mol, mo_coeff[i][:, :nspin])
                iao = lo.vec_lowdin(iao, s)
                # IBOs
                mo_coeff[i][:, :nspin] = lo.ibo.ibo(mol, mo_coeff[i][:, :nspin], iaos=iao, \
                                                    grad_tol = 1.e-10, exponent=int(variant[-1]), verbose=0)
                # closed-shell system
                if mol.spin == 0:
                    mo_coeff[i+1][:, :nspin] = mo_coeff[i][:, :nspin]
                    break

        return mo_coeff


def assign_rdm1s(mol: gto.Mole, s: np.ndarray, mo_coeff: Tuple[np.ndarray, np.ndarray], \
                 mo_occ: np.ndarray, pop: str, verbose: int) -> List[np.ndarray]:
        """
        this function returns a list of population weights of each spin-orbital on the individual atoms
        """
        # init charge weights array
        weights = [np.zeros([mol.nalpha, mol.natm], dtype=np.float64), np.zeros([mol.nbeta, mol.natm], dtype=np.float64)]

        if pop == 'iao':
            # mol object projected into minao basis
            pmol = mol.copy()
            pmol.build(False, False, basis='minao')

        if 0 < verbose:
            symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
            print('\n *** partial charge weights: ***')
            print(' spin  ' + 'MO       ' + '      '.join(['{:}'.format(i) for i in symbols]))

        for i, nspin in enumerate((mol.nalpha, mol.nbeta)):

            if pop == 'mulliken':
                mo = mo_coeff[i]
            elif pop == 'iao':
                iao = lo.iao.iao(mol, mo_coeff[i][:, :nspin])
                iao = lo.vec_lowdin(iao, s)
                mo = np.einsum('ki,kl,lj->ij', iao, s, mo_coeff[i][:, :nspin])

            for j in range(nspin):
                # get orbital
                orb = mo[:, j].reshape(mo.shape[0], 1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                # charge centre weights of rdm1_orb
                weights[i][j] = _charges(pmol if pop == 'iao' else mol, \
                                         s if pop == 'mulliken' else np.eye(rdm1_orb.shape[0]), \
                                         rdm1_orb)

                if 0 < verbose:
                    with np.printoptions(suppress=True, linewidth=200, formatter={'float': '{:6.3f}'.format}):
                        print('  {:s}    {:>2d}   {:}'.format('a' if i == 0 else 'b', j, weights[i][j]))

            # closed-shell system
            if mol.spin == 0:
                weights[i+1] = weights[i]
                break

        return weights


def _charges(mol: gto.Mole, s: np.ndarray, rdm1: np.ndarray) -> np.ndarray:
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


