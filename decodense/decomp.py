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

import warnings
import numpy as np
from pyscf import gto, scf, dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from typing import List, Dict, Union, Any


# component keys
COMP_KEYS = ['coul', 'exch', 'kin', 'solvent', 'nuc_att_glob', 'nuc_att_loc', 'nuc_att', 'xc', 'el', 'struct', 'nuc_att_vloc1_glob', 'nuc_att_vloc1_loc', 'nuc_att_vloc2_glob', 'nuc_att_vloc2_loc', 'nuc_att_vnlc_glob', 'nuc_att_vnlc_loc']

class DecompCls(object):
        """
        this class contains all decomp attributes
        """
        def __init__(self, loc: str = '', pop: str = 'mulliken', \
                     part = 'atoms', ndo: bool = False, multiproc: bool = False, \
                     gauge_origin: Union[List[Any], np.ndarray] = np.zeros(3), \
                     prop: str = 'energy', write: str = '', verbose: int = 0) -> None:
                """
                init molecule attributes
                """
                # set system defaults
                self.loc = loc
                self.pop = pop
                self.part = part
                self.ndo = ndo
                self.multiproc = multiproc
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

# TODO why mol can be None
def sanity_check(mol: Union[None, gto.Mole, pbc_gto.Cell], \
                 mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF], \
                 decomp: DecompCls) -> None:
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
        assert decomp.part in ['atoms', 'eda', 'orbitals'], \
            'invalid partitioning. valid choices: `atoms` (default), `eda`, or `orbitals`'
        # ndo decomposition
        assert isinstance(decomp.ndo, bool), \
            'invalid ndo argument. must be a bool'
        # multiprocessing
        assert isinstance(decomp.multiproc, bool), \
            'invalid multiprocessing argument. must be a bool'
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
        # pbc implementation
        if isinstance(mol, pbc_gto.Cell):
            assert np.shape(mf.kpt) == (3,), \
                'PBC module is in development, only gamma-point methods implemented.'
            assert gamma_point(mf.kpt), \
                'PBC module is in development, only gamma-point methods implemented.'
            assert mol.dimension == 3 or mol.dimension == 1, \
               'PBC module is in development, current implementation treats 1D- and 3D-cells only.' 
            assert decomp.prop == 'energy' and decomp.part in ['atoms', 'eda'], \
                'PBC module is in development. Only gamma-point RHF calculation of energy for 1D- and 3D-periodic systems can be decomposed into atomwise contributions.'

