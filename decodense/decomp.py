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


# property keys
PROP_KEYS = ['coul', 'exch', 'kin', 'rdm_att', 'xc', 'nuc_att', 'el', 'struct']

class DecompCls(object):
        """
        this class contains all decomp attributes
        """
        def __init__(self, basis: str = 'sto3g', loc: str = '', pop: str = 'mulliken', \
                     xc: str = '', part = 'atoms', irrep_nelec: Dict[str, int] = {}, \
                     ref: str = 'restricted', conv_tol: float = 1.e-10, thres = .75, \
                     mom: List[Dict[int, int]] = [], grid_level: int = 3, \
                     multiproc: bool = False, prop: str = 'energy', \
                     cube: bool = False, verbose: int = 0) -> None:
                """
                init molecule attributes
                """
                # set system defaults
                self.basis = basis
                self.loc = loc
                self.pop = pop
                self.xc = xc
                self.part = part
                self.irrep_nelec = irrep_nelec
                self.ref = ref
                self.conv_tol = conv_tol
                self.thres = thres
                self.mom = mom
                self.grid_level = grid_level
                self.multiproc = multiproc
                self.prop = prop
                self.cube = cube
                self.verbose = verbose
                # set internal defaults
                self.res: Dict[str, np.ndarray] = {'kin': None, 'coul': None, 'exch': None, 'xc': None, \
                                                   'nuc_att': None, 'nuc_rep': None, 'el': None, 'tot': None}
                self.charge_atom: np.ndarray = None
                self.dist: np.ndarray = None
                self.weights: np.ndarray = None
                self.centres: np.ndarray = None


def sanity_check(mol: gto.Mole, decomp: DecompCls) -> None:
        """
        this function performs sanity checks of decomp attributes
        """
        # basis
        assert decomp.basis == mol.basis, \
            'invalid basis. basis set (default: `sto3g`) in decomp and mol objects must match'
        # localization procedure
        assert decomp.loc in ['', 'fb', 'pm', 'ibo-2', 'ibo-4'], \
            'invalid localization procedure. valid choices: none (default), `fb`, `pm`, `ibo-2`, and `ibo-4`'
        # population scheme
        assert decomp.pop in ['mulliken', 'iao'], \
            'invalid population scheme. valid choices: `mulliken` (default) or `iao`'
        # partitioning
        assert decomp.part in ['atoms', 'eda', 'bonds'], \
            'invalid partitioning. valid choices: `atoms` (default), `eda`, or `bonds`'
        # irrep_nelec
        assert decomp.irrep_nelec is False or all([isinstance(i, int) for i in decomp.irrep_nelec.values()]), \
            'invalid irrep_nelec dict. valid choices: empty (default) or dict of str and ints'
        # reference
        assert decomp.ref in ['restricted', 'unrestricted'], \
            'invalid reference. valid choices: `restricted` (default) and `unrestricted`'
        # mf convergence tolerance
        assert isinstance(decomp.conv_tol, float), \
            'invalid convergence threshold. valid choices: 0. < `conv_tol` (default: 1.e-10)'
        assert 0. < decomp.conv_tol, \
            'invalid convergence threshold. valid choices: 0. < `conv_tol` (default: 1.e-10)'
        # bond partitioning threshold
        assert isinstance(decomp.thres, float), \
            'invalid bond partitioning threshold. valid choices: 0. < `thres` < 1. (default: .75)'
        assert 0. < decomp.thres < 1., \
            'invalid bond partitioning threshold. valid choices: 0. < `thres` < 1. (default: .75)'
        # mom
        assert isinstance(decomp.mom, list), \
            'invalid mom argument. must be a list of dictionaries'
        if 0 < len(decomp.mom):
            assert decomp.ref == 'unrestricted', \
                'invalid mom argument. only implemented for unrestricted references'
        assert all([isinstance(i, int) for j in decomp.mom for i in j.keys()]), \
            'invalid mom argument. dictionaries keys (MO indices) must be ints'
        assert all([isinstance(i, float) and i in [0., 1., 2.] for j in decomp.mom for i in j.values()]), \
            'invalid mom argument. dictionaries values (occupations) must be floats with a value of 0., 1., or 2.'
        assert len(decomp.mom) <= 2, \
            'invalid mom argument. must be a list of at max two dictionaries'
        # ks-dft grid level
        assert isinstance(decomp.grid_level, int), \
            'invalid ks-dft grid level. valid choices: 0 < `grid_level` (default: 3)'
        assert 0 < decomp.grid_level, \
            'invalid ks-dft grid level. valid choices: 0 < `grid_level` (default: 3)'
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


