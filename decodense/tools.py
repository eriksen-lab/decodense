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
from typing import Tuple, Union


class Logger(object):
        """
        this class pipes all write statements to both stdout and output_file
        """
        def __init__(self, output_file, both=True):
            """
            init Logger
            """
            self.terminal = sys.stdout
            self.log = open(output_file, 'a')
            self.both = both

        def write(self, message):
            """
            define write
            """
            self.log.write(message)
            if self.both:
                self.terminal.write(message)

        def flush(self):
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


def time_str(time: float) -> str:
        """
        this function returns time as a HH:MM:SS string
        """
        # hours, minutes, and seconds
        hours = time // 3600.
        minutes = (time - (time // 3600) * 3600.) // 60.
        seconds = time - hours * 3600. - minutes * 60.

        # init time string
        string: str = ''
        form: Tuple[float, ...] = ()

        # write time string
        if hours > 0:
            string += '{:.0f}h '
            form += (hours,)
        if minutes > 0:
            string += '{:.0f}m '
            form += (minutes,)
        string += '{:.2f}s'
        form += (seconds,)

        return string.format(*form)


def _ncore(mol: gto.Mole) -> int:
        """
        this function returns number of core orbitals
        """
        # init ncore
        ncore = 0
        # loop over atoms
        for i in range(mol.natm):
            if mol.atom_charge(i) > 2:
                ncore += 1
            if mol.atom_charge(i) > 12:
                ncore += 4
            if mol.atom_charge(i) > 20:
                ncore += 4
            if mol.atom_charge(i) > 30:
                ncore += 6
        return ncore


def dim(mol: gto.Mole, mo_occ: np.ndarray) -> Tuple[int, int, int]:
        """
        determine molecular dimensions
        """
        ncore = _ncore(mol)
        nalpha = np.where(mo_occ[0] > 0.)[0].size
        nbeta = np.where(mo_occ[1] > 0.)[0].size
        return ncore, nalpha, nbeta


def make_rdm1(mo: np.ndarray, occup: np.ndarray) -> np.ndarray:
        """
        this function returns an 1-RDM (in ao basis) corresponding to given mo(s)
        """
        return np.einsum('ip,jp->ij', occup * mo, mo)


