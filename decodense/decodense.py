#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main mf_decomp program
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import warnings
import numpy as np
from pyscf import lib, gto, scf, dft
from pyscf.pbc import gto as cgto
from pyscf.pbc import scf as kscf
from mpi4py import MPI
from typing import Dict, Tuple, List, Union, Any

from .decomp import DecompCls, sanity_check
from .orbitals import loc_orbs, assign_rdm1s
from .properties import prop_tot
from .tools import dim, make_rdm1, format_mf, write_rdm1


def main(mol: Union[None, gto.Mole, cgto.Cell], decomp: DecompCls, \
         mf: Union[None, scf.hf.SCF, dft.rks.KohnShamDFT, kscf.RHF], \
         loc_lst: Union[None, List[Any]] = None, \
         dipole_origin: Union[List[float], np.ndarray] = [0.] * 3) -> Dict[str, Any]:
        """
        main decodense program
        """
        # sanity check
        sanity_check(mol, decomp)
    
        # init time
        time = MPI.Wtime()

        if isinstance(mol, cgto.Cell) and (decomp.prop == 'energy'):
            if decomp.part == 'atoms':
               # decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, decomp.pop, \
               #                       decomp.prop, decomp.part, decomp.multiproc, \
               #                       weights = weights, dipole_origin = dipole_origin)
                decomp.res['time'] = MPI.Wtime() - time

                warnings.warn('PBC module is in development, but atomwise nuclear-nuclear repulsion' + \
                         ' term of energy at gamma point has been computed', Warning)
                return decomp.res
            else:
                sys.exit('PBC module is in development, can only compute atomwise ' + \
                         'nuclear-nuclear repulsion term of energy at gamma point')
        elif isinstance(mol, cgto.Cell) and (decomp.prop != 'energy'):
            sys.exit('PBC module is in development, the only valid choice for property: energy')

        # mf calculation
        mo_coeff, mo_occ = format_mf(mf, mol.spin)

        # overlap matrix
        s = mol.intor_symmetric('int1e_ovlp')

        # compute localized molecular orbitals
        if decomp.loc != '':
            mo_coeff = loc_orbs(mol, mo_coeff, mo_occ, s, decomp.loc, loc_lst)

        # inter-atomic distance array
        dist = gto.mole.inter_distance(mol) * lib.param.BOHR

        # decompose property
        if decomp.part in ['atoms', 'eda']:
            weights = assign_rdm1s(mol, s, mo_coeff, mo_occ, decomp.pop, decomp.part, \
                                   multiproc = decomp.multiproc, verbose = decomp.verbose)[0]
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, decomp.pop, \
                                  decomp.prop, decomp.part, decomp.multiproc, \
                                  weights = weights, dipole_origin = dipole_origin)
        elif decomp.part == 'bonds':
            rep_idx, centres = assign_rdm1s(mol, s, mo_coeff, mo_occ, decomp.pop, decomp.part, \
                                            multiproc = decomp.multiproc, verbose = decomp.verbose, \
                                            thres = decomp.thres)
            decomp.res = prop_tot(mol, mf, mo_coeff, mo_occ, decomp.pop, \
                                  decomp.prop, decomp.part, decomp.multiproc, \
                                  rep_idx = rep_idx, dipole_origin = dipole_origin)

        # write rdm1s
        if decomp.write != '':
            write_rdm1(mol, decomp.part, mo_coeff, mo_occ, decomp.write, \
                       weights if decomp.part == 'atoms' else None, \
                       rep_idx if decomp.part == 'bonds' else None)

        # determine spin
        alpha, beta = dim(mol, mo_occ)
        decomp.res['ss'], decomp.res['s'] = scf.uhf.spin_square((mo_coeff[0][:, alpha], \
                                                                 mo_coeff[1][:, beta]), s)

        # collect time
        decomp.res['time'] = MPI.Wtime() - time

        # collect centres & dist
        if decomp.part == 'bonds':
            decomp.res['centres'] = centres
            decomp.res['dist'] = dist

        return decomp.res


