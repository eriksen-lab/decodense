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
                # set calculation defaults
                self.prop: str = 'energy'
                self.thres: float = .98


def sanity_check(decomp: DecompCls) -> None:
        """
        this function performs sanity checks of decomp attributes
        """
        # singlet check
        assert decomp.spin == 0, \
            'invalid spin. decodense is currently only implemented for singlet ground states'
        # property
        assert decomp.prop in ['energy', 'dipole'], \
            'invalid property. valid choices: `energy` and `dipole`'
        # localization procedure
        assert decomp.loc in ['pm', 'ibo-2', 'ibo-4'], \
            'invalid localization procedure. valid choices: `pm`, `ibo-2`, and `ibo-4`'
        # population scheme
        assert decomp.pop in ['mulliken', 'iao'], \
            'invalid population scheme. valid choices: `mulliken` and `iao`'


if __name__ == "__main__":
    import doctest
    doctest.testmod()


