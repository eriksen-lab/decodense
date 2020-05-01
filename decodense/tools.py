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
import numpy as np
from subprocess import Popen, PIPE
from pyscf import gto, scf, dft
from pyscf import tools as pyscf_tools
from typing import Tuple, Dict, Union


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


def mf_calc(mol: gto.Mole, xc: str, ref: str, irrep_nelec: Dict['str', int], \
            conv_tol: float, verbose: int) -> Tuple[Union[scf.hf.SCF, dft.rks.KohnShamDFT], np.ndarray, np.ndarray]:
        """
        this function returns the results of a mean-field (hf or ks-dft) calculation
        """
        if xc == '':
            # hf calc
            if mol.spin == 0:
                mf = scf.RHF(mol)
            else:
                if ref == 'restricted':
                    mf = scf.ROHF(mol)
                elif ref == 'unrestricted':
                    mf = scf.UHF(mol)
        else:
            # dft calc
            if mol.spin == 0:
                mf = dft.RKS(mol)
            else:
                if ref == 'restricted':
                    mf = dft.ROKS(mol)
                elif ref == 'unrestricted':
                    mf = dft.UKS(mol)
            mf.xc = xc
        mf.irrep_nelec = irrep_nelec
        mf.conv_tol = conv_tol
        mf.verbose = verbose
        mf.kernel()
        assert mf.converged, 'mean-field calculation not converged'

        # restricted references
        if ref == 'restricted':
            mo_coeff = np.asarray((mf.mo_coeff,) * 2)
            mo_occ = np.asarray((np.zeros(mf.mo_occ.size, dtype=np.float64),) * 2)
            mo_occ[0][np.where(0. < mf.mo_occ)] += 1.
            mo_occ[1][np.where(1. < mf.mo_occ)] += 1.
        else:
            mo_coeff = mf.mo_coeff
            mo_occ = mf.mo_occ

        return mf, mo_coeff, mo_occ


def dim(mol: gto.Mole, mo_occ: np.ndarray) -> Tuple[int, int]:
        """
        determine molecular dimensions
        """
        nalpha = np.where(mo_occ[0] > 0.)[0].size
        nbeta = np.where(mo_occ[1] > 0.)[0].size

        return nalpha, nbeta


def make_rdm1(mo: np.ndarray, occup: np.ndarray) -> np.ndarray:
        """
        this function returns an 1-RDM (in ao basis) corresponding to given mo(s)
        """
        return np.einsum('ip,jp->ij', occup * mo, mo)


def write_cube(mol: gto.Mole, rdm1: np.ndarray, name: str) -> None:
        """
        this function writes an 1-RDM1 as a cube file
        """
        pyscf_tools.cubegen.density(mol, '{:}.cube'.format(name), rdm1)


