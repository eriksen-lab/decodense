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
from typing import List, Dict, Union, Any


# component keys
COMP_KEYS = ['coul', 'exch', 'kin', 'solvent', 'nuc_att_glob', 'nuc_att_loc', 'nuc_att', 'xc', 'el', 'struct']

class DecompCls(object):
        """
        this class contains all decomp attributes
        """
        __slots__ = ('mo_basis', 'pop', 'mo_init', 'part', 'ndo', \
                     'gauge_origin', 'prop', 'write', 'verbose', \
                     'res', 'charge_atom', 'dist', 'weights', 'centres')

        def __init__(self, mo_basis: str = 'can', pop: str = 'mulliken', mo_init: str = 'can', \
                     part = 'atoms', ndo: bool = False, \
                     gauge_origin: Union[List[Any], np.ndarray] = np.zeros(3), \
                     prop: str = 'energy', write: str = '', verbose: int = 0) -> None:
                """
                init molecule attributes
                """
                # set system defaults
                self.mo_basis = mo_basis
                self.pop = pop
                self.mo_init = mo_init
                self.part = part
                self.ndo = ndo
                self.gauge_origin = gauge_origin
                self.prop = prop
                self.write = write
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
        # MO basis
        assert decomp.mo_basis in ['can', 'fb', 'pm'], \
            'invalid MO basis. valid choices: `can` (default), `fb`, or `pm`'
        # population scheme
        assert decomp.pop in ['mulliken', 'lowdin', 'meta_lowdin', 'becke', 'iao'], \
            'invalid population scheme. valid choices: `mulliken` (default), `lowdin`, `meta_lowdin`, `becke`, or `iao`'
        # MO start guess (for localization)
        assert decomp.mo_init in ['can', 'ibo'], \
            'invalid MO start guess. valid choices: `can` (default) or `ibo`'
        # partitioning
        assert decomp.part in ['atoms', 'eda', 'orbitals'], \
            'invalid partitioning. valid choices: `atoms` (default), `eda`, or `orbitals`'
        # NDO decomposition
        assert isinstance(decomp.ndo, bool), \
            'invalid NDO argument. must be a bool'
        # gauge origin
        assert isinstance(decomp.gauge_origin, (list, np.ndarray)), \
            'invalid gauge origin. must be a list or numpy array of ints/floats'
        # property
        assert decomp.prop in ['energy', 'dipole'], \
            'invalid property. valid choices: `energy` (default) and `dipole`'
        # write
        assert isinstance(decomp.write, str), \
            'invalid write format argument. must be a str'
        assert decomp.write in ['', 'cube', 'numpy'], \
            'invalid write format. valid choices: `cube` and `numpy`'
        # verbosity
        assert isinstance(decomp.verbose, int), \
            'invalid verbosity. valid choices: 0 <= `verbose` (default: 0)'
        assert 0 <= decomp.verbose, \
            'invalid verbosity. valid choices: 0 <= `verbose` (default: 0)'


