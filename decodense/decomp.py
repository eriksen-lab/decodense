#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
decomp module
"""

__author__ = 'Janus Juul Eriksen, Technical University of Denmark, DK'
__maintainer__ = 'Janus Juul Eriksen'
__email__ = 'janus@kemi.dtu.dk'
__status__ = 'Development'

import numpy as np
from pyscf import gto, scf, dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from typing import List, Dict, Union, Any


# component keys
class CompKeys:
        coul = 'Coul.'
        exch = 'Exch.'
        kin = 'Kin.'
        solvent = 'Solv.'
        nuc_att_glob = 'E_ne (1)'
        nuc_att_loc = 'E_ne (2)'
        nuc_att = 'E_ne'
        xc = 'XC'
        struct = 'Struct.'
        nuc_att_vloc_glob = 'E_ne_loc (1)'
        nuc_att_vloc_loc = 'E_ne_loc (2)'
        nuc_att_vnlc_glob = 'E_ne_nl (1)'
        nuc_att_vnlc_loc = 'E_ne_nl (2)'
        el = 'Elect.'
        tot = 'Total'
        charge_atom = 'Charge'
        atoms = 'Atom'
        orbitals = 'Orbital'
        mo_occ = 'Occup.'
        orbsym = 'Symm.'

class DecompCls(object):
        """
        this class contains all decomp attributes
        """
        __slots__ = ('minao', 'mo_basis', 'pop_method', 'mo_init', 'loc_exp', 'part', \
                     'ndo', 'gauge_origin', 'prop', 'write', 'verbose', 'unit', \
                     'res', 'charge_atom', 'dist', 'weights', 'centres')

        def __init__(self, minao: str = 'MINAO', mo_basis: str = 'can', pop_method: str = 'mulliken', \
                     mo_init: str = 'can', loc_exp: int = 2, \
                     part = 'atoms', ndo: bool = False, \
                     gauge_origin: Union[List[Any], np.ndarray] = np.zeros(3), \
                     prop: str = 'energy', write: str = '', verbose: int = 0, unit: str = 'au') -> None:
                """
                init molecule attributes
                """
                # set system defaults
                self.minao = minao
                self.mo_basis = mo_basis
                self.pop_method = pop_method
                self.mo_init = mo_init
                self.loc_exp = loc_exp
                self.part = part
                self.ndo = ndo
                self.gauge_origin = gauge_origin
                self.prop = prop
                self.write = write
                self.verbose = verbose
                self.unit = unit
                # set internal defaults
                self.res: Dict[str, np.ndarray] = {}
                self.charge_atom: np.ndarray = None
                self.dist: np.ndarray = None
                self.weights: np.ndarray = None
                self.centres: np.ndarray = None

def sanity_check(mol: Union[gto.Mole, pbc_gto.Cell], \
                 mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF], \
                 decomp: DecompCls) -> None:
        """
        this function performs sanity checks of decomp attributes
        """
        # Reference basis for IAOs
        assert decomp.minao in ['MINAO', 'ANO'], \
            'invalid minao basis. valid choices: `MINAO` (default) or `ANO`'
        # MO basis
        assert decomp.mo_basis in ['can', 'fb', 'pm'], \
            'invalid MO basis. valid choices: `can` (default), `fb`, or `pm`'
        # population scheme
        assert decomp.pop_method in ['mulliken', 'lowdin', 'meta_lowdin', 'becke', 'iao'], \
            'invalid population scheme. valid choices: `mulliken` (default), `lowdin`, `meta_lowdin`, `becke`, or `iao`'
        # MO start guess (for localization)
        assert decomp.mo_init in ['can', 'cholesky', 'ibo'], \
            'invalid MO start guess. valid choices: `can` (default), `cholesky`, or `ibo`'
        # localization exponent
        assert decomp.loc_exp in [2, 4], \
            'invalid MO start guess. valid choices: 2 (default) or 4'
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
        # cell object
        if isinstance(mol, pbc_gto.Cell):
            assert np.shape(mf.kpt) == (3,), \
                'PBC module is in development, only gamma-point methods implemented.'
            assert gamma_point(mf.kpt), \
                'PBC module is in development, only gamma-point methods implemented.'
            assert mol.dimension == 3 or mol.dimension == 1, \
               'PBC module is in development, current implementation treats 1D- and 3D-cells only.' 
            assert decomp.prop == 'energy' and decomp.part in ['atoms', 'eda'], \
                'PBC module is in development. Only gamma-point calculation of energy for 1D- and 3D-periodic systems can be decomposed into atomwise contributions.'
        # unit
        assert isinstance(decomp.unit, str), \
            'invalid unit. valid choices: `au` (default), `kcal_mol`, `ev`, `kj_mol`, or `debye`'

