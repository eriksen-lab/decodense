#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tools module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import copy
import numpy as np
try:
    import opt_einsum as oe
    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False
from subprocess import Popen, PIPE
from pyscf import gto, scf, dft
from pyscf import tools as pyscf_tools
from typing import Tuple, List, Dict, Union

MAX_CYCLE = 100
NDO_THRES = 1.e-12

class Logger(object):
        """
        this class pipes all write statements to both stdout and output_file
        """
        def __init__(self, output_file, both=True) -> None:
            """
            init Logger
            """
            self.terminal = sys.stdout
            self.log = open(output_file, 'a')
            self.both = both

        def write(self, message) -> None:
            """
            define write
            """
            self.log.write(message)
            if self.both:
                self.terminal.write(message)

        def flush(self) -> None:
            """
            define flush
            """
            pass


def git_version() -> str:
        """
        this function returns the git revision as a string
        """
        def _minimal_ext_cmd(cmd):
            env = {}
            for k in ['SYSTEMROOT', 'PATH', 'HOME']:
                v = os.environ.get(k)
                if v is not None:
                    env[k] = v
            # LANGUAGE is used on win32
            env['LANGUAGE'] = 'C'
            env['LANG'] = 'C'
            env['LC_ALL'] = 'C'
            out = Popen(cmd, stdout=PIPE, env=env, \
                        cwd=os.path.dirname(__file__)).communicate()[0]
            return out

        try:
            out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
            GIT_REVISION = out.strip().decode('ascii')
        except OSError:
            GIT_REVISION = "Unknown"

        return GIT_REVISION


def dim(mol: gto.Mole, mo_occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        determine molecular dimensions
        """
        return np.where(np.abs(mo_occ[0]) > 0.)[0], np.where(np.abs(mo_occ[1]) > 0.)[0]


def mf_info(mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], \
            mo_coeff_in: np.ndarray = None, \
            mo_occ_in: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        retrieve mf information (mo coefficients & occupations)
        """
        # mo coefficients
        if mo_coeff_in is None:
            if np.asarray(mf.mo_coeff).ndim == 2:
                mo_coeff_out = np.asarray((mf.mo_coeff,) * 2)
            else:
                mo_coeff_out = mf.mo_coeff
        else:
            if np.asarray(mo_coeff_in).ndim == 2:
                mo_coeff_out = np.asarray((mo_coeff_in,) * 2)
            else:
                mo_coeff_out = mo_coeff_in
        # mo occupations
        if mo_occ_in is None:
            if np.asarray(mf.mo_occ).ndim == 1:
                mo_occ_out = np.asarray((np.zeros(mf.mo_occ.size, dtype=np.float64),) * 2)
                mo_occ_out[0][np.where(0. < mf.mo_occ)] += 1.
                mo_occ_out[1][np.where(1. < mf.mo_occ)] += 1.
            else:
                mo_occ_out = mf.mo_occ
        else:
            if np.asarray(mo_occ_in).ndim == 1:
                mo_occ_out = np.asarray((np.zeros(mo_occ_in.size, dtype=np.float64),) * 2)
                mo_occ_out[0][np.where(0. < mo_occ_in)] += 1.
                mo_occ_out[1][np.where(1. < mo_occ_in)] += 1.
            else:
                mo_occ_out = mo_occ_in

        return mo_coeff_out, mo_occ_out


def make_rdm1(mo: np.ndarray, occup: np.ndarray) -> np.ndarray:
        """
        this function returns an 1-RDM (in ao basis) corresponding to given mo(s)
        """
        return contract('ip,jp->ij', occup * mo, mo)


def make_ndo(mol: gto.Mole, mo_coeff: np.ndarray, \
             rdm1_delta: np.ndarray, thres: float = NDO_THRES) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns ndo coefficients and occupations corresponding
        to given mo coefficients and rdm1_delta
        """
        # assertions
        assert mo_coeff.ndim == 3, '`make_ndo` functions expects alpha/beta mo coefficients'
        assert rdm1_delta.ndim == 3, '`make_ndo` functions expects alpha/beta delta rdm1'
        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')
        # ao to mo transformation of dm_delta
        rdm1_delta_mo = contract('xpi,pq,xqr,rs,xsj->xij', mo_coeff, s, rdm1_delta, s, mo_coeff)
        # diagonalize dm_delta_mo
        occ_ndo, u = np.linalg.eigh(rdm1_delta_mo)
        # transform to ndo basis
        mo_ndo = contract('xip,xpj->xij', mo_coeff, u)

        # retain only significant ndos
        return np.array([mo_ndo[i][:, np.where(np.abs(occ_ndo[i]) >= thres)[0]] for i in range(2)]), \
               np.array([occ_ndo[i][np.where(np.abs(occ_ndo[i]) >= thres)[0]] for i in range(2)])


def write_rdm1(mol: gto.Mole, part: str, \
               mo_coeff: np.ndarray, mo_occ: np.ndarray, fmt: str, \
               weights: List[np.ndarray] = None, \
               rep_idx: List[List[np.ndarray]] = None, \
               identifier: str = '') -> None:
        """
        this function writes a 1-RDM as a numpy or cube (default) file
        """
        # assertion
        assert fmt in ['cube', 'numpy'], 'fmt arg to write_rdm1() must be `cube` or `numpy`'
        # molecular dimensions
        alpha, beta = dim(mol, mo_occ)
        # compute total 1-RDM (AO basis)
        rdm1_tot = np.array([make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])])
        # write rdm1s for given partitioning
        if part == 'atoms':
            # assertion
            assert weights is not None, 'missing `weights` arg in write_rdm1() function'
            # loop over atoms
            for a in range(mol.natm):
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
                        rdm1_atom[i] += rdm1_orb * weights[i][m][a]
                if fmt == 'cube':
                    # write rdm1_atom as cube file
                    pyscf_tools.cubegen.density(mol, f'atom_{mol.atom_symbol(a).upper():s}{a:d}_rdm1{identifier:}.cube', \
                                                np.sum(rdm1_atom, axis=0))
                else:
                    # write rdm1_atom as numpy file
                    np.save(f'atom_{mol.atom_symbol(a).upper():s}{a:d}_rdm1{identifier:}.npy', np.sum(rdm1_atom, axis=0))
        elif part == 'bonds':
            # assertion
            assert rep_idx is not None, 'missing `rep_idx` arg in write_rdm1() function'
            # loop over spins
            for i, _ in enumerate((alpha, beta)):
                # loop over repeating indices
                for k, j in enumerate(rep_idx[i]):
                    # get orbital(s)
                    orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], -1)
                    # orbital-specific rdm1
                    rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                    if fmt == 'cube':
                        # write rdm1_orb as cube file
                        pyscf_tools.cubegen.density(mol, f'spin_{"a" if i == 0 else "b":s}_rdm1_{k:d}{identifier:}.cube', rdm1_orb)
                    else:
                        # write rdm1_orb as numpy file
                        np.save(f'spin_{"a" if i == 0 else "b":s}_rdm1_{k:d}{identifier:}.npy', rdm1_orb)
        else:
            raise RuntimeError('invalid choice of partitioning in write_rdm1() function.')


def res_add(res_a, res_b):
        """
        this function adds two result dictionaries
        """
        return {key: res_a[key] + res_b[key] for key in res_a.keys()}


def res_sub(res_a, res_b):
        """
        this function subtracts two result dictionaries
        """
        return {key: res_a[key] - res_b[key] for key in res_a.keys()}


def contract(eqn, *tensors):
        """
        interface to optimized einsum operation
        """
        if OE_AVAILABLE:
            return oe.contract(eqn, *tensors)
        else:
            return np.einsum(eqn, *tensors, optimize=True)


