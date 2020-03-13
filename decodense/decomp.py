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


class DecompCls(object):
        """
        this class contains all decomp attributes
        """
        def __init__(self) -> None:
                """
                init molecule attributes
                """
                # set system defaults
                self.basis: str = 'sto-3g'
                self.spin: int = 0
                self.loc: str = 'pm'
                self.pop: str = 'mulliken'
                self.xc: str = ''
                self.time: float = 0.
                # set calculation defaults
                self.orbs: str = 'localized'
                self.prop: str = 'energy'
                self.part: str = 'atoms'
                self.thres: float = .98
                self.verbose: bool = False


def sanity_check(decomp: DecompCls) -> None:
        """
        this function performs sanity checks of decomp attributes
        """
        # singlet check
        assert decomp.spin == 0, \
            'invalid spin. decodense is currently only implemented for singlet ground states'
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
        assert 0. < decomp.thres < 1., \
            'invalid threshold. valid choices: 0. < `thres` < 1. (default: .98)'
        # localization procedure
        assert decomp.loc in ['pm', 'ibo-2', 'ibo-4'], \
            'invalid localization procedure. valid choices: `pm` (default), `ibo-2`, and `ibo-4`'
        # population scheme
        assert decomp.pop in ['mulliken', 'iao'], \
            'invalid population scheme. valid choices: `mulliken` (default) and `iao`'
        # verbosity
        assert isinstance(decomp.verbose, bool), \
            'invalid verbosity. valid choices: True or False'


