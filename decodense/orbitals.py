#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
orbitals module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import multiprocessing as mp
import numpy as np
from pyscf import gto, scf, dft, lo, lib
from typing import List, Tuple, Dict, Union, Any

from .tools import dim, make_rdm1, contract

LOC_CONV = 1.e-10


def loc_orbs(mol: gto.Mole, mo_coeff_in: np.ndarray, \
             mo_occ: np.ndarray, variant: str, ndo: bool, \
             loc_lst: Union[None, List[Any]]) -> np.ndarray:
        """
        this function returns a set of localized MOs of a specific variant
        """
        # rhf reference
        if mo_occ[0].size == mo_occ[1].size:
            rhf = np.allclose(mo_coeff_in[0], mo_coeff_in[1]) and np.allclose(mo_occ[0], mo_occ[1])
        else:
            rhf = False

        # ndo assertion
        assert not ndo, 'localization of NDOs is not implemented'

        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # molecular dimensions
        alpha, beta = dim(mo_occ)

        # localization list(s)
        if loc_lst is not None:
            if not rhf:
                assert len(loc_lst) == 2, 'loc_lst must be supplied for both alpha and beta spaces'
            for i, idx_arr in enumerate(loc_lst):
                assert np.sum([len(idx) for idx in idx_arr]) == (alpha, beta)[i].size, 'loc_lst does not cover all occupied orbitals'

        # init mo_coeff_out
        mo_coeff_out = [np.zeros_like(mo_coeff_in[i]) for i in range(2)]

        # loop over spins
        for i, spin_mo in enumerate((alpha, beta)):

            # selective localization
            if loc_lst is None:
                idx_arr = spin_mo.reshape(1, -1)
            else:
                idx_arr = loc_lst[i]

            # localize orbitals
            for idx in idx_arr:
                if variant == 'fb':
                    # foster-boys procedure
                    loc = lo.Boys(mol, mo_coeff_in[i][:, idx])
                    loc.conv_tol = LOC_CONV
                    # FB MOs
                    mo_coeff_out[i][:, idx] = loc.kernel()
                elif variant == 'pm':
                    # pipek-mezey procedure
                    loc = lo.PM(mol, mo_coeff_in[i][:, idx])
                    loc.conv_tol = LOC_CONV
                    # PM MOs
                    mo_coeff_out[i][:, idx] = loc.kernel()
                elif 'ibo' in variant:
                    # orthogonalized IAOs
                    iao = lo.iao.iao(mol, mo_coeff_in[i][:, idx])
                    iao = lo.vec_lowdin(iao, s)
                    # IBOs
                    mo_coeff_out[i][:, idx] = lo.ibo.ibo(mol, mo_coeff_in[i][:, idx], iaos=iao, \
                                                         grad_tol = LOC_CONV, exponent=int(variant[-1]), verbose=0)

            # closed-shell reference
            if rhf:
                mo_coeff_out[i+1][:, spin_mo] = mo_coeff_out[i][:, spin_mo]
                break

        return mo_coeff_out


def assign_rdm1s(mol: gto.Mole, mo_coeff: np.ndarray, \
                 mo_occ: np.ndarray, pop: str, part: str, ndo: bool, \
                 multiproc: bool, verbose: bool, **kwargs: Any) -> List[np.ndarray]:
        """
        this function returns a list of population weights of each spin-orbital on the individual atoms
        """
        # declare nested kernel function in global scope
        global get_weights

        # rhf reference
        if mo_occ[0].size == mo_occ[1].size:
            rhf = np.allclose(mo_coeff[0], mo_coeff[1]) and np.allclose(mo_occ[0], mo_occ[1])
        else:
            rhf = False

        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # molecular dimensions
        alpha, beta = dim(mo_occ)

        # max number of occupied spin-orbs
        n_spin = max(alpha.size, beta.size)

        # mol object projected into minao basis
        if pop == 'iao':
            # ndo assertion
            assert not ndo, 'IAO-based populations for NDOs is not implemented'
            pmol = lo.iao.reference_mol(mol)
        else:
            pmol = mol

        # number of atoms
        natm = pmol.natm

        # AO labels
        ao_labels = pmol.ao_labels(fmt=None)

        # overlap matrix
        if pop == 'mulliken':
            ovlp = s
        else:
            ovlp = np.eye(pmol.nao_nr())

        def get_weights(orb_idx: int):
            """
            this function computes the full set of population weights
            """
            # get orbital
            orb = mo[:, orb_idx].reshape(mo.shape[0], 1)
            # orbital-specific rdm1
            rdm1_orb = make_rdm1(orb, mocc[orb_idx])
            # population weights of rdm1_orb
            return _population(natm, ao_labels, ovlp, rdm1_orb)

        # init population weights array
        weights = [np.zeros([n_spin, pmol.natm], dtype=np.float64), np.zeros([n_spin, pmol.natm], dtype=np.float64)]

        # loop over spin
        for i, spin_mo in enumerate((alpha, beta)):

            # get mo coefficients and occupation
            if pop == 'mulliken':
                mo = mo_coeff[i][:, spin_mo]
            elif pop == 'iao':
                iao = lo.iao.iao(mol, mo_coeff[i][:, spin_mo])
                iao = lo.vec_lowdin(iao, s)
                mo = contract('ki,kl,lj->ij', iao, s, mo_coeff[i][:, spin_mo])
            mocc = mo_occ[i][spin_mo]

            # domain
            domain = np.arange(spin_mo.size)
            # execute kernel
            if multiproc:
                n_threads = min(domain.size, lib.num_threads())
                with mp.Pool(processes=n_threads) as pool:
                    weights[i] = pool.map(get_weights, domain) # type:ignore
            else:
                weights[i] = list(map(get_weights, domain)) # type:ignore

            # closed-shell reference
            if rhf:
                weights[i+1] = weights[i]
                break

        # verbose print
        if 0 < verbose:
            symbols = [pmol.atom_pure_symbol(i) for i in range(pmol.natm)]
            print('\n *** partial population weights: ***')
            print(' spin  ' + 'MO       ' + '      '.join(['{:}'.format(i) for i in symbols]))
            for i, spin_mo in enumerate((alpha, beta)):
                for j in domain:
                    with np.printoptions(suppress=True, linewidth=200, formatter={'float': '{:6.3f}'.format}):
                        print('  {:s}    {:>2d}   {:}'.format('a' if i == 0 else 'b', spin_mo[j], weights[i][j]))

        return weights


def _population(natm: int, ao_labels: np.ndarray, ovlp: np.ndarray, rdm1: np.ndarray) -> np.ndarray:
        """
        this function returns the mulliken populations on the individual atoms
        """
        # mulliken population array
        pop = contract('ij,ji->i', rdm1, ovlp)
        # init populations
        populations = np.zeros(natm)

        # loop over AOs
        for i, k in enumerate(ao_labels):
            populations[k[0]] += pop[i]

        return populations


