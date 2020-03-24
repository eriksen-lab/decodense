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
from typing import Dict


class DecompCls(object):
        """
        this class contains all decomp attributes
        """
        def __init__(self) -> None:
                """
                init molecule attributes
                """
                # set system defaults
                self.loc: str = 'ibo-2'
                self.pop: str = 'iao'
                self.xc: str = ''
                self.irrep_nelec: Dict['str', int] = {}
                self.ref: str = 'restricted'
                self.orbs: str = 'localized'
                self.prop: str = 'energy'
                self.part: str = 'atoms'
                self.thres: float = .98
                self.verbose: int = 0
                # set calculation defaults
                self.ss: float = 0.
                self.s: float = 0.
                self.time: float = 0.
                self.prop_el: np.ndarray = None
                self.prop_nuc: np.ndarray = None
                self.prop_tot: np.ndarray = None
                self.cent: np.ndarray = None


def sanity_check(mol: gto.Mole, decomp: DecompCls) -> None:
        """
        this function performs sanity checks of decomp attributes
        """
        # irrep_nelec
        assert decomp.irrep_nelec is False or all([isinstance(i, int) for i in decomp.irrep_nelec.values()]), \
            'invalid irrep_nelec dict. valid choices: empty (default) or dict of str and ints'
        # reference
        assert decomp.ref in ['restricted', 'unrestricted'], \
            'invalid reference. valid choices: `restricted` (default) and `unrestricted`'
        if decomp.ref == 'unrestricted':
            assert mol.spin != 0, \
                'invalid reference. unrestricted references are only meaningful for non-singlet states'
        # orbitals
        assert decomp.orbs in ['canonical', 'localized'], \
            'invalid orbitals. valid choices: `canonical` and `localized` (default)'
        # property
        assert decomp.prop in ['energy', 'dipole'], \
            'invalid property. valid choices: `energy` (default) and `dipole`'
        # partitioning
        assert decomp.part in ['atoms', 'bonds'], \
            'invalid partitioning. valid choices: `atoms` (default) and `bonds`'
        # threshold
        assert isinstance(decomp.thres, float), \
            'invalid threshold. valid choices: 0. < `thres` < 1. (default: .98)'
        assert 0. < decomp.thres < 1., \
            'invalid threshold. valid choices: 0. < `thres` < 1. (default: .98)'
        # localization procedure
        assert decomp.loc in ['pm', 'ibo-2', 'ibo-4'], \
            'invalid localization procedure. valid choices: `pm` (default), `ibo-2`, and `ibo-4`'
        # population scheme
        assert decomp.pop in ['mulliken', 'iao'], \
            'invalid population scheme. valid choices: `mulliken` (default) and `iao`'
        # verbosity
        assert isinstance(decomp.verbose, int), \
            'invalid verbosity. valid choices: 0 <= `verbose` (default: 0)'
        assert 0 <= decomp.verbose, \
            'invalid verbosity. valid choices: 0 <= `verbose` (default: 0)'


