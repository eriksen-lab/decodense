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


def loc_orbs(mol: gto.Mole, mo_coeff: Tuple[np.ndarray, np.ndarray], \
             s: np.ndarray, ref: str, variant: str) -> np.ndarray:
        """
        this function returns a set of localized MOs of a specific variant
        """
        # init localizer
        if variant == 'fb':
            for i, spin_mo in enumerate((mol.alpha, mol.beta)):
                # foster-boys procedure
                loc = lo.Boys(mol, mo_coeff[i][:, spin_mo])
                loc.conv_tol = 1.e-10
                # FB MOs
                mo_coeff[i][:, spin_mo] = loc.kernel()
                # closed-shell reference
                if ref == 'restricted' and mol.spin == 0:
                    mo_coeff[i+1][:, spin_mo] = mo_coeff[i][:, spin_mo]
                    break
        elif variant == 'pm':
            for i, spin_mo in enumerate((mol.alpha, mol.beta)):
                # pipek-mezey procedure
                loc = lo.PM(mol, mo_coeff[i][:, spin_mo])
                loc.conv_tol = 1.e-10
                # PM MOs
                mo_coeff[i][:, spin_mo] = loc.kernel()
                # closed-shell reference
                if ref == 'restricted' and mol.spin == 0:
                    mo_coeff[i+1][:, spin_mo] = mo_coeff[i][:, spin_mo]
                    break
        elif 'ibo' in variant:
            for i, spin_mo in enumerate((mol.alpha, mol.beta)):
                # orthogonalized IAOs
                iao = lo.iao.iao(mol, mo_coeff[i][:, spin_mo])
                iao = lo.vec_lowdin(iao, s)
                # IBOs
                mo_coeff[i][:, spin_mo] = lo.ibo.ibo(mol, mo_coeff[i][:, spin_mo], iaos=iao, \
                                                    grad_tol = 1.e-10, exponent=int(variant[-1]), verbose=0)
                # closed-shell reference
                if ref == 'restricted' and mol.spin == 0:
                    mo_coeff[i+1][:, spin_mo] = mo_coeff[i][:, spin_mo]
                    break

        return mo_coeff


def assign_rdm1s(mol: gto.Mole, s: np.ndarray, mo_coeff: Tuple[np.ndarray, np.ndarray], \
                 mo_occ: np.ndarray, ref: str, pop: str, part: str, verbose: int, \
                 **kwargs: float) -> Tuple[Union[List[np.ndarray], List[List[np.ndarray]]], Union[None, np.ndarray]]:
        """
        this function returns a list of population weights of each spin-orbital on the individual atoms
        """
        # init population weights array
        weights = [np.zeros([mol.alpha.size, mol.natm], dtype=np.float64), np.zeros([mol.beta.size, mol.natm], dtype=np.float64)]

        # init population centres array and get threshold
        if part == 'bonds':
            centres = [np.zeros([mol.alpha.size, 2], dtype=np.int), np.zeros([mol.beta.size, 2], dtype=np.int)]
            thres = kwargs['thres']

        # mol object projected into minao basis
        if pop == 'iao':
            pmol = mol.copy()
            pmol.build(False, False, basis='minao')

        # verbose print
        if 0 < verbose:
            symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
            print('\n *** partial population weights: ***')
            print(' spin  ' + 'MO       ' + '      '.join(['{:}'.format(i) for i in symbols]))

        # loop over spin
        for i, spin_mo in enumerate((mol.alpha, mol.beta)):

            # get mo coefficients and occupation
            if pop == 'mulliken':
                mo = mo_coeff[i][:, spin_mo]
            elif pop == 'iao':
                iao = lo.iao.iao(mol, mo_coeff[i][:, spin_mo])
                iao = lo.vec_lowdin(iao, s)
                mo = np.einsum('ki,kl,lj->ij', iao, s, mo_coeff[i][:, spin_mo])
            mocc = mo_occ[i][spin_mo]

            # loop over spin-orbitals
            for j in range(spin_mo.size):
                # get orbital
                orb = mo[:, j].reshape(mo.shape[0], 1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mocc[j])
                # population weights of rdm1_orb
                weights[i][j] = _population(pmol if pop == 'iao' else mol, \
                                            s if pop == 'mulliken' else np.eye(rdm1_orb.shape[0]), \
                                            rdm1_orb)

                # verbose print
                if 0 < verbose:
                    with np.printoptions(suppress=True, linewidth=200, formatter={'float': '{:6.3f}'.format}):
                        print('  {:s}    {:>2d}   {:}'.format('a' if i == 0 else 'b', j, weights[i][j]))

                if part == 'bonds':
                    # get sorted indices
                    max_idx = np.argsort(weights[i][j])[::-1]
                    # compute population centres
                    if np.abs(weights[i][j][max_idx[0]]) / np.abs((weights[i][j][max_idx[0]] + weights[i][j][max_idx[1]])) > thres:
                        # core orbital or lone pair
                        centres[i][j] = np.sort(np.array([max_idx[0], max_idx[0]], dtype=np.int))
                    else:
                        # valence orbitals
                        centres[i][j] = np.sort(np.array([max_idx[0], max_idx[1]], dtype=np.int))

            # closed-shell reference
            if ref == 'restricted' and mol.spin == 0:
                weights[i+1] = weights[i]
                if part == 'bonds':
                    centres[i+1] = centres[i]
                break

        # unique and repetitive centres
        if part == 'bonds':
            centres_unique = np.array([np.unique(centres[i], axis=0) for i in range(2)])
            rep_idx = [[np.where((centres[i] == j).all(axis=1))[0] for j in centres_unique[i]] for i in range(2)]

        if part in ['atoms', 'eda']:
            return weights, None
        else: # bonds
            return rep_idx, centres_unique


def _population(mol: gto.Mole, s: np.ndarray, rdm1: np.ndarray) -> np.ndarray:
        """
        this function returns the mulliken populations on the individual atoms
        """
        # mulliken population matrix
        pop = np.einsum('ij,ji->i', rdm1, s)
        # init populations
        populations = np.zeros(mol.natm)

        # loop over AOs
        for i, k in enumerate(mol.ao_labels(fmt=None)):
            populations[k[0]] += pop[i]

        return populations


