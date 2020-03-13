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
        string += ' point group        = {:}\n'
        string += ' basis set          = {:}\n'
        string += '\n localization       = {:}\n'
        string += ' assignment         = {:}\n'
        string += ' partitioning       = {:}\n'
        string += ' threshold          = {:}\n'
        form += (mol.groupname, decomp.basis, decomp.loc, decomp.pop, decomp.part, decomp.thres,)
        if decomp.xc != '':
            string += ' xc functional      = {:}\n'
            form += (decomp.xc,)
        string += '\n electrons          = {:}\n'
        string += ' occupied orbitals  = {:}\n'
        string += ' virtual orbitals   = {:}\n'
        string += ' total orbitals     = {:}\n'
        form += (mol.nelectron, mol.nocc, mol.nvirt, mol.norb,)

        # calculation info
        string += '\n total time         = {:}\n'
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


def table_atoms(mol: gto.Mole, decomp: DecompCls, \
                prop: np.ndarray, orb_type: str) -> str:
        """
        this function prints the results based on an `atoms` partitioning
        """
        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        if decomp.prop == 'energy':

            string += '-----------------------\n'
            string += '{:^23}\n'
            string += '-----------------------\n'
            string += ' atom |   {:9}\n'
            string += '-----------------------\n'
            string += '-----------------------\n'
            form += ('ground-state energy', orb_type,)

            for i in range(prop.size):
                string += ' {:<5s}|{:>12.5f}\n'
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), prop[i],)

            string += '-----------------------\n'
            string += '-----------------------\n'
            string += ' tot  |{:>12.5f}\n'
            string += '-----------------------\n\n'
            form += (np.sum(prop),)

        elif decomp.prop == 'dipole':

            string += '-------------------------------------------\n'
            string += '{:^43}\n'
            string += '-------------------------------------------\n'
            string += ' atom |             {:9}\n'
            string += '-------------------------------------------\n'
            string += '      |     x     /     y     /     z     |\n'
            string += '-------------------------------------------\n'
            string += '-------------------------------------------\n'
            form += ('ground-state dipole moment', orb_type,)

            for i in range(prop.shape[0]):
                string += ' {:<5s}| {:>8.3f}  / {:>8.3f}  / {:>8.3f}\n'
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), *prop[i] + 1.0e-10,)

            string += '-------------------------------------------\n'
            string += '-------------------------------------------\n'

            string += ' tot  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}\n'
            string += '-------------------------------------------\n\n'
            form += (*np.fromiter(map(math.fsum, prop.T), dtype=prop.dtype, count=prop.shape[1]) + 1.0e-10,)

        return string.format(*form)


def table_bonds(mol: gto.Mole, decomp: DecompCls, \
                prop: np.ndarray, centres: np.ndarray, orb_type: str) -> str:
        """
        this function prints the results based on a `bonds` partitioning
        """
        # inter-atomic distance array
        dist = gto.mole.inter_distance(mol) * lib.param.BOHR

        # nuclear repulsion energy and dipole moment
        if decomp.prop == 'energy':
            prop_nuc = mol.energy_nuc()
        elif decomp.prop == 'dipole':
            prop_nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())

        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        if decomp.prop == 'energy':

            string += '--------------------------------------------------------\n'
            string += '{:^55}\n'
            string += '--------------------------------------------------------\n'
            string += '  MO  |   {:9}   |    atom(s)    |   bond length\n'
            string += '--------------------------------------------------------\n'
            string += '--------------------------------------------------------\n'
            form += ('ground-state energy', orb_type,)

            for i in range(prop.size):
                core = centres[i, 0] == centres[i, 1]
                string += '  {:>2d}  |{:>12.5f}   |    {:<11s}| {:>10s}\n'
                form += (i, prop[i], \
                         '{:s}{:d}'.format(mol.atom_symbol(centres[i, 0]), centres[i, 0]) if core \
                         else '{:s}{:d}-{:s}{:d}'.format(mol.atom_symbol(centres[i, 0]), centres[i, 0], \
                                                           mol.atom_symbol(centres[i, 1]), centres[i, 1]), \
                         '' if core else '{:>.3f}'.format(dist[centres[i, 0], centres[i, 1]]),)

            string += '--------------------------------------------------------\n'
            string += '--------------------------------------------------------\n'
            string += ' sum  |{:>12.5f}   |\n'
            form += (np.sum(prop),)

            string += '-----------------------\n'
            string += ' nuc  |{:>+12.5f}   |\n'
            form += (prop_nuc,)

            string += '-----------------------\n'
            string += '-----------------------\n'
            string += ' tot  |{:>12.5f}   |\n'
            string += '-----------------------\n\n'
            form += (np.sum(prop) + prop_nuc,)

        elif decomp.prop == 'dipole':

            string += '----------------------------------------------------------------------------\n'
            string += '{:^70}\n'
            string += '----------------------------------------------------------------------------\n'
            string += '  MO  |             {:9}             |    atom(s)    |   bond length\n'
            string += '----------------------------------------------------------------------------\n'
            string += '      |     x     /     y     /     z     |\n'
            string += '----------------------------------------------------------------------------\n'
            string += '----------------------------------------------------------------------------\n'
            form += ('ground-state dipole moment', orb_type,)

            for i in range(prop.shape[0]):
                core = centres[i, 0] == centres[i, 1]
                string += '  {:>2d}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |{:^15s}| {:>10s}\n'
                form += (i, *prop[i] + 1.0e-10, \
                            mol.atom_symbol(centres[i, 0]) if core else '{:s} & {:s}'. \
                            format(mol.atom_symbol(centres[i, 0]), mol.atom_symbol(centres[i, 1])), \
                            '' if core else '{:>.3f}'. \
                            format(dist[centres[i, 0], centres[i, 1]]),)

            string += '----------------------------------------------------------------------------\n'
            string += '----------------------------------------------------------------------------\n'

            string += ' sum  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
            form += (*np.fromiter(map(math.fsum, prop.T), dtype=prop.dtype, count=prop.shape[1]) + 1.0e-10,)

            string += '----------------------------------------------------------------------------\n'
            string += ' nuc  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
            form += (*prop_nuc + 1.0e-10,)

            string += '----------------------------------------------------------------------------\n'
            string += '----------------------------------------------------------------------------\n'

            string += ' tot  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |\n'
            string += '----------------------------------------------------------------------------\n\n'
            form += (*(prop_nuc + np.fromiter(map(math.fsum, prop.T), dtype=prop.dtype, count=prop.shape[1])) + 1.0e-10,)

        return string.format(*form)


