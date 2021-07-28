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
        return np.where(mo_occ[0] > 0.)[0], np.where(mo_occ[1] > 0.)[0]


def format_mf(mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT], spin: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        format mf information (mo coefficients & occupations)
        """
        if isinstance(mf, (scf.hf.RHF, scf.rohf.ROHF, dft.rks.RKS, dft.roks.ROKS)):
            mo_coeff = np.asarray((mf.mo_coeff,) * 2)
            mo_occ = np.asarray((np.zeros(mf.mo_occ.size, dtype=np.float64),) * 2)
            if isinstance(mf, (scf.hf.RHF, dft.rks.RKS)):
                mo_occ[0][np.where(0. < mf.mo_occ)] += 1.
                mo_occ[1][np.where(1. < mf.mo_occ)] += 1.
            elif isinstance(mf, (scf.hf.ROHF, dft.rks.ROKS)):
                if spin != 0:
                    mo_occ[0][np.where(0. < mf.mo_occ)] += 1.
                    mo_occ[1][np.where(1. < mf.mo_occ)] += 1.
                else:
                    mo_occ[0][np.where(0. < mf.mo_occ)] += 1.
                    mo_occ[0][np.where(mf.mo_occ == 1.)[0]] += 1.
                    mo_occ[1][np.where(0. < mf.mo_occ)] += 1.
                    mo_occ[1][np.where(mf.mo_occ == 1.)[1]] += 1.
        else:
            mo_coeff = mf.mo_coeff
            mo_occ = mf.mo_occ
        return mo_coeff, mo_occ


def make_rdm1(mo: np.ndarray, occup: np.ndarray) -> np.ndarray:
        """
        this function returns an 1-RDM (in ao basis) corresponding to given mo(s)
        """
        return contract('ip,jp->ij', occup * mo, mo)


def write_cube(mol: gto.Mole, part: str, mo_coeff: np.ndarray, mo_occ: np.ndarray, \
               weights: List[np.ndarray] = None, rep_idx: List[List[np.ndarray]] = None) -> None:
        """
        this function writes a 1-RDM as a cube file
        """
        # compute total 1-RDM (AO basis)
        rdm1_tot = np.array([make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])])
        # write cube files for given partitioning
        if part == 'atoms':
            # assertion
            assert weights is not None, 'missing `weights` arg in write_cube() function'
            # loop over atoms
            for a in range(mol.natm):
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
                        rdm1_atom[i] += rdm1_orb * weights[i][m][a]
                # write rdm1_atom as cube file
                pyscf_tools.cubegen.density(mol, f'atom_{mol.atom_symbol(a).upper():s}_rdm1_{a:d}.cube', \
                                            np.sum(rdm1_atom, axis=0))
        elif part == 'bonds':
            # assertion
            assert rep_idx is not None, 'missing `rep_idx` arg in write_cube() function'
            # loop over spins
            for i, _ in enumerate((mol.alpha, mol.beta)):
                # loop over repeating indices
                for k, j in enumerate(rep_idx[i]):
                    # get orbital(s)
                    orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], -1)
                    # orbital-specific rdm1
                    rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                    # write rdm1_orb as cube file
                    pyscf_tools.cubegen.density(mol, f'spin_{"a" if i == 0 else "b":s}_rdm1_{k:d}.cube', \
                                                rdm1_orb)
        else:
            raise RuntimeError('invalid choice of partitioning in write_cube() function.')


def contract(eqn, *tensors):
        """
        interface to optimized einsum operation
        """
        if OE_AVAILABLE:
            return oe.contract(eqn, *tensors)
        else:
            return np.einsum(eqn, *tensors, optimize=True)


