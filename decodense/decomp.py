#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
decomp module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from pyscf import gto
from typing import List, Dict


# component keys
COMP_KEYS = ['coul', 'exch', 'kin', 'solvent', 'nuc_att_glob', 'nuc_att_loc', 'nuc_att', 'xc', 'el', 'struct']

class DecompCls(object):
        """
        this class contains all decomp attributes
        """
        def __init__(self, loc: str = '', pop: str = 'mulliken', \
                     part = 'atoms', thres = .75, multiproc: bool = False, \
                     prop: str = 'energy', cube: bool = False, verbose: int = 0) -> None:
                """
                init molecule attributes
                """
                # set system defaults
                self.loc = loc
                self.pop = pop
                self.part = part
                self.thres = thres
                self.multiproc = multiproc
                self.prop = prop
                self.cube = cube
                self.verbose = verbose
                # set internal defaults
                self.res: Dict[str, np.ndarray] = {comp_key: None for comp_key in COMP_KEYS}
                self.charge_atom: np.ndarray = None
                self.dist: np.ndarray = None
                self.weights: np.ndarray = None
                self.centres: np.ndarray = None


def sanity_check(mol: gto.Mole, decomp: DecompCls) -> None:
        """
        this function performs sanity checks of decomp attributes
        """
        # localization procedure
        assert decomp.loc in ['', 'fb', 'pm', 'ibo-2', 'ibo-4'], \
            'invalid localization procedure. valid choices: none (default), `fb`, `pm`, `ibo-2`, and `ibo-4`'
        # population scheme
        assert decomp.pop in ['mulliken', 'iao'], \
            'invalid population scheme. valid choices: `mulliken` (default) or `iao`'
        # partitioning
        assert decomp.part in ['atoms', 'eda', 'bonds'], \
            'invalid partitioning. valid choices: `atoms` (default), `eda`, or `bonds`'
        # bond partitioning threshold
        assert isinstance(decomp.thres, float), \
            'invalid bond partitioning threshold. valid choices: 0. < `thres` < 1. (default: .75)'
        assert 0. < decomp.thres < 1., \
            'invalid bond partitioning threshold. valid choices: 0. < `thres` < 1. (default: .75)'
        # multiprocessing
        assert isinstance(decomp.multiproc, bool), \
            'invalid multiprocessing argument. must be a bool'
        # property
        assert decomp.prop in ['energy', 'dipole'], \
            'invalid property. valid choices: `energy` (default) and `dipole`'
        # cube
        assert isinstance(decomp.cube, bool), \
            'invalid cube argument. must be a bool'
        # verbosity
        assert isinstance(decomp.verbose, int), \
            'invalid verbosity. valid choices: 0 <= `verbose` (default: 0)'
        assert 0 <= decomp.verbose, \
            'invalid verbosity. valid choices: 0 <= `verbose` (default: 0)'


