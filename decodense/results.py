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
from pyscf import gto
from typing import Tuple, List, Any

import system
import tools

# output folder and files
OUT = os.getcwd()+'/output'
RES_FILE = OUT+'/mf_decomp.results'


def sort(mol: gto.Mole, res_can_old: np.ndarray, res_loc_old: np.ndarray, \
         centres_old: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    this function returns sorted results for unique bonds only
    """
    # sort results wrt canonical contributions
    sorter_can = np.argsort(res_can_old)
    res_can_new = res_can_old[sorter_can]
    # sort results wrt localized contributions
    sorter_loc = np.argsort(res_loc_old)
    res_loc_new = res_loc_old[sorter_loc]
    # sort localized centres
    centres_new = centres_old[sorter_loc]

    return res_can_new, res_loc_new, centres_new


def main(mol: gto.Mole, decomp: system.DecompCls) -> str:
    """
    this function prints the results header
    """
    # init string & form
    string: str = ''
    form: Tuple[Any, ...] = ()

    # print geometry
    string += '\n\n   ------------------------------------\n'
    string += '{:^43}\n'
    string += '   ------------------------------------\n'
    form += ('geometry',)

    molecule = mol.atom.split('\n')
    for i in range(len(molecule)-1):
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
    if decomp.dft:
        string += ' xc functional      = {:}\n'
        form += (decomp.xc,)
    string += '\n electrons          = {:}\n'
    string += ' occupied orbitals  = {:}\n'
    string += ' virtual orbitals   = {:}\n'
    string += ' total orbitals     = {:}\n'
    form += (mol.nelectron, mol.nocc, mol.nvirt, mol.norb,)

    # git version
    string += '\n git version: {:}\n\n'
    form += (tools.git_version(),)

    return string.format(*form)


def energy(mol: gto.Mole, e_can: np.ndarray, e_loc: np.ndarray, \
            e_nuc: np.ndarray, e_ref: np.ndarray, centres: np.ndarray, rr: np.ndarray) -> str:
    """
    this function prints the energy results
    """
    # init string & form
    string: str = ''
    form: Tuple[Any, ...] = ()

    string += '------------------------------------------------------------------------\n'
    string += '{:^70}\n'
    string += '------------------------------------------------------------------------\n'
    string += '  MO  |   canonical   |   localized   |     atom(s)   |   bond length\n'
    string += '------------------------------------------------------------------------\n'
    string += '------------------------------------------------------------------------\n'
    form += ('ground-state energy',)

    for i in range(mol.nocc):
        if i < e_loc.size:
            core = centres[i, 0] == centres[i, 1]
            string += '  {:>2d}  | {:>10.3f}    | {:>10.3f}    |{:^15s}| {:>10s}\n'
            form += (i, e_can[i], e_loc[i], \
                        mol.atom_symbol(centres[i, 0]) if core else '{:s} & {:s}'. \
                        format(mol.atom_symbol(centres[i, 0]), mol.atom_symbol(centres[i, 1])), \
                        '' if core else '{:>.3f}'. \
                        format(rr[centres[i, 0], centres[i, 1]]),)
        else:
            string += '  {:>2d}  | {:>10.3f}    |\n'
            form += (i, e_can[i],)

    string += '------------------------------------------------------------------------\n'
    string += '------------------------------------------------------------------------\n'
    string += '  sum | {:>10.3f}    | {:>10.3f}    |\n'
    form += (np.sum(e_can), np.sum(e_loc),)

    string += '---------------------------------------\n'
    string += '  nuc | {:>+10.3f}    | {:>+10.3f}    |\n'
    form += (e_nuc, e_nuc,)

    string += '---------------------------------------\n'
    string += '---------------------------------------\n'
    string += '  tot | {:>12.5f}  | {:>12.5f}  |\n'
    form += (np.sum(e_can) + e_nuc, np.sum(e_loc) + e_nuc,)

    string += '---------------------------------------\n\n'
    string += ' *** reference energy = {:.5f}\n\n'
    form += (e_ref,)

    return string.format(*form)


def dipole(mol: gto.Mole, dip_can: np.ndarray, dip_loc: np.ndarray, \
            dip_nuc: np.ndarray, dip_ref: np.ndarray, centres: np.ndarray, rr: np.ndarray) -> str:
    """
    this function prints the dipole results
    """
    # init string & form
    string: str = ''
    form: Tuple[Any, ...] = ()

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
        if i < dip_loc.shape[0]:
            core = centres[i, 0] == centres[i, 1]
            string += '  {:>2d}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |{:^15s}| {:>10s}\n'
            form += (i, *dip_can[i] + 1.0e-10, *dip_loc[i] + 1.0e-10, \
                        mol.atom_symbol(centres[i, 0]) if core else '{:s} & {:s}'. \
                        format(mol.atom_symbol(centres[i, 0]), mol.atom_symbol(centres[i, 1])), \
                        '' if core else '{:>.3f}'. \
                        format(rr[centres[i, 0], centres[i, 1]]),)
        else:
            string += '  {:>2d}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
            form += (i, *dip_can[i] + 1.0e-10,)

    string += '----------------------------------------------------------------------------------------------------------------\n'
    string += '----------------------------------------------------------------------------------------------------------------\n'

    string += '  sum | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
    form += (*np.fromiter(map(math.fsum, dip_can.T), dtype=dip_can.dtype, count=dip_can.shape[1]) + 1.0e-10, \
                *np.fromiter(map(math.fsum, dip_loc.T), dtype=dip_loc.dtype, count=dip_loc.shape[1]) + 1.0e-10,)

    string += '-------------------------------------------------------------------------------\n'
    string += '  nuc | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
    form += (*dip_nuc + 1.0e-10, *dip_nuc + 1.0e-10,)

    string += '-------------------------------------------------------------------------------\n'
    string += '-------------------------------------------------------------------------------\n'

    string += '  tot | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
    form += (*(dip_nuc - np.fromiter(map(math.fsum, dip_can.T), dtype=dip_can.dtype, count=dip_can.shape[1])) + 1.0e-10, \
                *(dip_nuc - np.fromiter(map(math.fsum, dip_loc.T), dtype=dip_loc.dtype, count=dip_loc.shape[1])) + 1.0e-10,)

    string += '-------------------------------------------------------------------------------\n\n'
    string += ' *** reference dipole moment = {:>8.3f}  / {:>8.3f}  / {:>8.3f}\n\n'
    form += (*dip_ref + 1.0e-10,)

    return string.format(*form)

