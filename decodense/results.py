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
from pyscf import gto, lib
from typing import Tuple, List, Union, Any

from .decomp import DecompCls
from .tools import git_version, time_str


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
        string += ' point group        =  {:}\n'
        string += ' basis set          =  {:}\n'
        string += '\n localization       =  {:}\n'
        string += ' assignment         =  {:}\n'
        string += ' partitioning       =  {:}\n'
        string += ' orbital basis      =  {:}\n'
        string += ' threshold          =  {:}\n'
        form += (mol.groupname, decomp.basis, decomp.loc, decomp.pop, decomp.part, decomp.orbs, decomp.thres,)
        if decomp.xc != '':
            string += ' xc functional      =  {:}\n'
            form += (decomp.xc,)
        string += '\n electrons          =  {:}\n'
        string += ' spin multiplicity  =  {:}\n'
        string += ' alpha electrons    =  {:}\n'
        string += ' beta electrons     =  {:}\n'
        string += ' basis functions    =  {:}\n'
        form += (mol.nelectron, mol.spin, mol.nalpha, mol.nbeta, mol.nao_nr(),)

        # calculation info
        string += '\n total time         =  {:}\n'
        if decomp.prop == 'energy':
            string += ' reference result   = {:.5f}\n'
            form += (time_str(decomp.time), decomp.prop_ref)
        elif decomp.prop == 'dipole':
            string += ' reference result   = {:.3f}  / {:.3f}  / {:.3f}\n'
            form += (time_str(decomp.time), *decomp.prop_ref)

        # git version
        string += '\n git version: {:}\n\n'
        form += (git_version(),)

        return string.format(*form)


def table_atoms(mol: gto.Mole, decomp: DecompCls) -> str:
        """
        this function prints the results based on an atom-based partitioning
        """
        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        if decomp.prop == 'energy':

            string += '----------------------------------------------------\n'
            string += '{:^52}\n'
            string += '{:^52}\n'
            string += '----------------------------------------------------\n'
            string += '----------------------------------------------------\n'
            string += ' atom |  electronic  |    nuclear   |     total\n'
            string += '----------------------------------------------------\n'
            string += '----------------------------------------------------\n'
            form += ('ground-state energy', decomp.orbs + ' MOs',)

            for i in range(mol.natm):
                string += ' {:<5s}|{:>12.5f}  |{:>+12.5f}  |{:>+12.5f}\n'
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), \
                                           decomp.prop_el[i], decomp.prop_nuc[i], \
                                           decomp.prop_el[i] + decomp.prop_nuc[i],)

            string += '----------------------------------------------------\n'
            string += '----------------------------------------------------\n'
            string += ' sum  |{:>12.5f}  |{:>+12.5f}  |{:>+12.5f}\n'
            string += '----------------------------------------------------\n\n'
            form += (np.sum(decomp.prop_el), np.sum(decomp.prop_nuc), \
                     np.sum(decomp.prop_el + decomp.prop_nuc),)

        elif decomp.prop == 'dipole':

            string += '-------------------------------------------------------------------------------------------------------------------\n'
            string += '{:^83}\n'
            string += '{:^83}\n'
            string += '-------------------------------------------------------------------------------------------------------------------\n'
            string += '      |             electronic            |               nuclear             |               total\n'
            string += ' atom -------------------------------------------------------------------------------------------------------------\n'
            string += '      |     x     /     y     /     z     |     x     /     y     /     z     |     x     /     y     /     z\n'
            string += '-------------------------------------------------------------------------------------------------------------------\n'
            string += '-------------------------------------------------------------------------------------------------------------------\n'
            form += ('ground-state dipole moment', decomp.orbs + ' MOs',)

            for i in range(mol.natm):
                string += ' {:<5s}| {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}\n'
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), \
                                           *decomp.prop_el[i] + 1.e-10, *decomp.prop_nuc[i] + 1.e-10, \
                                           *(decomp.prop_el[i] + decomp.prop_nuc[i]) + 1.e-10)

            string += '-------------------------------------------------------------------------------------------------------------------\n'
            string += '-------------------------------------------------------------------------------------------------------------------\n'

            string += ' sum  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}\n'
            string += '-------------------------------------------------------------------------------------------------------------------\n\n'
            form += (*np.fromiter(map(math.fsum, decomp.prop_el.T), dtype=np.float64, count=3) + 1.e-10, \
                     *np.fromiter(map(math.fsum, decomp.prop_nuc.T), dtype=np.float64, count=3) + 1.e-10, \
                     *np.fromiter(map(math.fsum, decomp.prop_el.T + decomp.prop_nuc.T), dtype=np.float64, count=3) + 1.e-10,)

        return string.format(*form)


def table_bonds(mol: gto.Mole, decomp: DecompCls, cent: np.ndarray) -> str:
        """
        this function prints the results based on a bond-based partitioning
        """
        # inter-atomic distance array
        dist = gto.mole.inter_distance(mol) * lib.param.BOHR

        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        if decomp.prop == 'energy':

            string += '--------------------------------------------------------\n'
            string += '{:^55}\n'
            string += '{:^55}\n'
            string += '--------------------------------------------------------\n'
            string += '  MO  |  electronic  |    atom(s)    |   bond length\n'
            string += '--------------------------------------------------------\n'
            string += '--------------------------------------------------------\n'
            form += ('ground-state energy', decomp.orbs,)

            for i in range(decomp.prop_el.size):
                core = cent[i, 0] == cent[i, 1]
                string += '  {:>2d}  |{:>12.5f}   |    {:<11s}| {:>10s}\n'
                form += (i, decomp.prop_el[i], \
                         '{:s}{:d}'.format(mol.atom_symbol(cent[i, 0]), cent[i, 0]) if core \
                         else '{:s}{:d}-{:s}{:d}'.format(mol.atom_symbol(cent[i, 0]), cent[i, 0], \
                                                           mol.atom_symbol(cent[i, 1]), cent[i, 1]), \
                         '' if core else '{:>.3f}'.format(dist[cent[i, 0], cent[i, 1]]),)

            string += '--------------------------------------------------------\n'
            string += '--------------------------------------------------------\n'
            string += ' sum  |{:>12.5f}   |\n'
            form += (np.sum(decomp.prop_el),)

            string += '-----------------------\n'
            string += ' nuc  |{:>+12.5f}   |\n'
            form += (np.sum(decomp.prop_nuc),)

            string += '-----------------------\n'
            string += '-----------------------\n'
            string += ' tot  |{:>12.5f}   |\n'
            string += '-----------------------\n\n'
            form += (np.sum(decomp.prop_el) + np.sum(decomp.prop_nuc),)

        elif decomp.prop == 'dipole':

            string += '----------------------------------------------------------------------------\n'
            string += '{:^70}\n'
            string += '{:^70}\n'
            string += '----------------------------------------------------------------------------\n'
            string += '  MO  |             electronic            |    atom(s)    |   bond length\n'
            string += '----------------------------------------------------------------------------\n'
            string += '      |     x     /     y     /     z     |\n'
            string += '----------------------------------------------------------------------------\n'
            string += '----------------------------------------------------------------------------\n'
            form += ('ground-state dipole moment', decomp.orbs,)

            for i in range(decomp.prop_el.shape[0]):
                core = cent[i, 0] == cent[i, 1]
                string += '  {:>2d}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |    {:<11s}| {:>10s}\n'
                form += (i, *decomp.prop_el[i] + 1.e-10, \
                         '{:s}{:d}'.format(mol.atom_symbol(cent[i, 0]), cent[i, 0]) if core \
                         else '{:s}{:d}-{:s}{:d}'.format(mol.atom_symbol(cent[i, 0]), cent[i, 0], \
                                                           mol.atom_symbol(cent[i, 1]), cent[i, 1]), \
                         '' if core else '{:>.3f}'. \
                         format(dist[cent[i, 0], cent[i, 1]]),)

            string += '----------------------------------------------------------------------------\n'
            string += '----------------------------------------------------------------------------\n'

            string += ' sum  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
            form += (*np.fromiter(map(math.fsum, decomp.prop_el.T), dtype=np.float64, count=3) + 1.e-10,)

            string += '----------------------------------------------------------------------------\n'
            string += ' nuc  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
            form += (*np.fromiter(map(math.fsum, decomp.prop_nuc.T), dtype=np.float64, count=3) + 1.e-10,)

            string += '----------------------------------------------------------------------------\n'
            string += '----------------------------------------------------------------------------\n'

            string += ' tot  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
            string += '----------------------------------------------------------------------------\n\n'
            form += (*(np.fromiter(map(math.fsum, decomp.prop_el.T), dtype=np.float64, count=3) + \
                     np.fromiter(map(math.fsum, decomp.prop_nuc.T), dtype=np.float64, count=3)) + 1.e-10,)

        return string.format(*form)


