#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
results module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import os
import numpy as np
import math
from pyscf import gto, scf, dft
from typing import Tuple, List, Union, Any

from .decomp import DecompCls
from .tools import git_version

# output folder and files
OUT = os.getcwd()+'/output'
RES_FILE = OUT+'/mf_decomp.results'


def sort(mol: gto.Mole, res_can_old: np.ndarray, res_loc_old: np.ndarray, \
         centres_old: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    this function returns sorted results for unique bonds only
    """
    # sort results wrt canonical contributions
    if res_can_old.ndim == 1:
        sorter_can = np.argsort(np.abs(res_can_old))[::-1]
    else:
        sorter_can = np.argsort(np.fromiter(map(np.linalg.norm, res_can_old), \
                                            dtype=np.float64, count=res_can_old.shape[0]))[::-1]
    res_can_new = res_can_old[sorter_can]
    # sort results wrt localized contributions
    if res_loc_old.ndim == 1:
        sorter_loc = np.argsort(np.abs(res_loc_old))[::-1]
    else:
        sorter_loc = np.argsort(np.fromiter(map(np.linalg.norm, res_loc_old), \
                                            dtype=np.float64, count=res_loc_old.shape[0]))[::-1]
    res_loc_new = res_loc_old[sorter_loc]
    # sort localized centres
    centres_new = centres_old[sorter_loc]

    return res_can_new, res_loc_new, centres_new


def info(mol: gto.Mole, decomp: DecompCls) -> str:
    """
    this function prints the results header and basic info
    """
    # init string & form
    string: str = ''
    form: Tuple[Any, ...] = ()

    # print geometry
    string += '\n\n   ------------------------------------\n'
    string += '{:^43}\n'
    string += '   ------------------------------------\n'
    form += ('geometry',)

    molecule = gto.tostring(mol).split('\n')
    for i in range(len(molecule)):
        atom = molecule[i].split()
        for j in range(1, 4):
            atom[j] = float(atom[j])
        string += '   {:<3s} {:>10.5f} {:>10.5f} {:>10.5f}\n'
        form += (*atom,)
    string += '   ------------------------------------\n'

    # system info
    string += '\n\n system info:\n'
    string += ' ------------\n'
    string += ' point group        = {:}\n'
    string += ' basis set          = {:}\n'
    string += '\n localization       = {:}\n'
    string += ' assignment         = {:}\n'
    string += ' threshold          = {:}\n'
    form += (mol.groupname, decomp.basis, decomp.loc, decomp.pop, decomp.thres,)
    if decomp.xc != '':
        string += ' xc functional      = {:}\n'
        form += (decomp.xc,)
    string += '\n electrons          = {:}\n'
    string += ' occupied orbitals  = {:}\n'
    string += ' virtual orbitals   = {:}\n'
    string += ' total orbitals     = {:}\n'
    form += (mol.nelectron, mol.nocc, mol.nvirt, mol.norb,)

    # git version
    string += '\n git version: {:}\n\n'
    form += (git_version(),)

    return string.format(*form)


def table(mol: gto.Mole, decomp: DecompCls, prop_can: np.ndarray, prop_loc: np.ndarray, \
          mf: Union[scf.hf.RHF, scf.hf_symm.RHF, dft.rks.RKS, dft.rks_symm.RKS], \
          centres: np.ndarray, dist: np.ndarray) -> str:
    """
    this function prints the energy results
    """
    # nuclear repulsion energy and dipole moment
    if decomp.prop == 'energy':
        prop_nuc = mol.energy_nuc()
        prop_ref = mf.e_tot
    elif decomp.prop == 'dipole':
        prop_nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
        prop_ref = scf.hf.dip_moment(mol, mf.make_rdm1(), unit='au', verbose=0)

    # init string & form
    string: str = ''
    form: Tuple[Any, ...] = ()

    if decomp.prop == 'energy':

        string += '------------------------------------------------------------------------\n'
        string += '{:^70}\n'
        string += '------------------------------------------------------------------------\n'
        string += '  MO  |   canonical   |   localized   |     atom(s)   |   bond length\n'
        string += '------------------------------------------------------------------------\n'
        string += '------------------------------------------------------------------------\n'
        form += ('ground-state energy',)

        for i in range(mol.nocc):
            if i < prop_loc.size:
                core = centres[i, 0] == centres[i, 1]
                string += '  {:>2d}  | {:>10.3f}    | {:>10.3f}    |{:^15s}| {:>10s}\n'
                form += (i, prop_can[i], prop_loc[i], \
                         mol.atom_symbol(centres[i, 0]) if core else '{:s} & {:s}'. \
                         format(mol.atom_symbol(centres[i, 0]), mol.atom_symbol(centres[i, 1])), \
                         '' if core else '{:>.3f}'. \
                         format(dist[centres[i, 0], centres[i, 1]]),)
            else:
                string += '  {:>2d}  | {:>10.3f}    |\n'
                form += (i, prop_can[i],)

        string += '------------------------------------------------------------------------\n'
        string += '------------------------------------------------------------------------\n'
        string += '  sum | {:>10.3f}    | {:>10.3f}    |\n'
        form += (np.sum(prop_can), np.sum(prop_loc),)

        string += '---------------------------------------\n'
        string += '  nuc | {:>+10.3f}    | {:>+10.3f}    |\n'
        form += (prop_nuc, prop_nuc,)

        string += '---------------------------------------\n'
        string += '---------------------------------------\n'
        string += '  tot | {:>12.5f}  | {:>12.5f}  |\n'
        form += (np.sum(prop_can) + prop_nuc, np.sum(prop_loc) + prop_nuc,)

        string += '---------------------------------------\n\n'
        string += ' *** reference energy = {:.5f}\n\n'
        form += (prop_ref,)

    elif decomp.prop == 'dipole':

        string += '----------------------------------------------------------------------------------------------------------------\n'
        string += '{:^100}\n'
        string += '----------------------------------------------------------------------------------------------------------------\n'
        string += '  MO  |             canonical             |            localized              |     atom(s)   |   bond length\n'
        string += '----------------------------------------------------------------------------------------------------------------\n'
        string += '      |     x     /     y     /     z     |     x     /     y     /     z     |\n'
        string += '----------------------------------------------------------------------------------------------------------------\n'
        string += '----------------------------------------------------------------------------------------------------------------\n'
        form += ('ground-state dipole moment',)

        for i in range(mol.nocc):
            if i < prop_loc.shape[0]:
                core = centres[i, 0] == centres[i, 1]
                string += '  {:>2d}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |{:^15s}| {:>10s}\n'
                form += (i, *prop_can[i] + 1.0e-10, *prop_loc[i] + 1.0e-10, \
                            mol.atom_symbol(centres[i, 0]) if core else '{:s} & {:s}'. \
                            format(mol.atom_symbol(centres[i, 0]), mol.atom_symbol(centres[i, 1])), \
                            '' if core else '{:>.3f}'. \
                            format(dist[centres[i, 0], centres[i, 1]]),)
            else:
                string += '  {:>2d}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
                form += (i, *prop_can[i] + 1.0e-10,)

        string += '----------------------------------------------------------------------------------------------------------------\n'
        string += '----------------------------------------------------------------------------------------------------------------\n'

        string += '  sum | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
        form += (*np.fromiter(map(math.fsum, prop_can.T), dtype=prop_can.dtype, count=prop_can.shape[1]) + 1.0e-10, \
                    *np.fromiter(map(math.fsum, prop_loc.T), dtype=prop_loc.dtype, count=prop_loc.shape[1]) + 1.0e-10,)

        string += '-------------------------------------------------------------------------------\n'
        string += '  nuc | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
        form += (*prop_nuc + 1.0e-10, *prop_nuc + 1.0e-10,)

        string += '-------------------------------------------------------------------------------\n'
        string += '-------------------------------------------------------------------------------\n'

        string += '  tot | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
        form += (*(prop_nuc + np.fromiter(map(math.fsum, prop_can.T), dtype=prop_can.dtype, count=prop_can.shape[1])) + 1.0e-10, \
                    *(prop_nuc + np.fromiter(map(math.fsum, prop_loc.T), dtype=prop_loc.dtype, count=prop_loc.shape[1])) + 1.0e-10,)

        string += '-------------------------------------------------------------------------------\n\n'
        string += ' *** reference dipole moment = {:>8.3f}  / {:>8.3f}  / {:>8.3f}\n\n'
        form += (*prop_ref + 1.0e-10,)

    return string.format(*form)

