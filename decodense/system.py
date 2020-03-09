#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
system module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__license__ = 'MIT'
__version__ = '0.8'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'


class DecompCls(object):
        """
        this class contains all system attributes
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
                self.dft: bool = False
                self.xc: str = ''
                # set calculation defaults
                self.prop: str = 'energy'
                self.thres: float = .98


def sanity_check(decomp: DecompCls) -> None:
        """
        this function performs sanity checks of decomp attributes
        """
        # singlet check
        assert decomp.spin == 0, 'decomposition scheme only implemented for singlet states'
        # dft check
        if decomp.dft:
            assert decomp.xc != '', 'invalid choice of xc functional'


if __name__ == "__main__":
    import doctest
    doctest.testmod()


