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
from typing import Dict, Tuple, List, Union, Any

from .decomp import DecompCls
from .tools import git_version
from .data import AU_TO_DEBYE


def info(decomp: DecompCls, mol: Union[None, gto.Mole] = None, time: Union[None, float] = None) -> str:
        """
        this function prints basic info
        """
        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        # print geometry
        if mol is not None:
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
        string += ' basis set          =  {:}\n'
        string += ' partitioning       =  {:}\n'
        string += ' assignment         =  {:}\n'
        form += (decomp.basis, decomp.part, decomp.pop,)
        string += ' localization       =  {:}\n'
        form += (_loc(decomp.loc),)
        string += ' xc functional      =  {:}\n'
        form += (_xc(decomp.xc),)
        if mol is not None:
            string += '\n reference funct.   =  {:}\n'
            string += ' point group        =  {:}\n'
            string += ' electrons          =  {:d}\n'
            string += ' alpha electrons    =  {:d}\n'
            string += ' beta electrons     =  {:d}\n'
            string += ' spin: <S^2>        =  {:.3f}\n'
            string += ' spin: 2*S + 1      =  {:.3f}\n'
            string += ' basis functions    =  {:d}\n'
            form += (_ref(mol, decomp.ref, decomp.xc), mol.groupname, mol.nelectron, mol.alpha.size, mol.beta.size, \
                     decomp.ss + 1.e-6, decomp.s + 1.e-6, mol.nao_nr(),)

        # timing and git version
        if time is not None:
            string += '\n total time         =  {:}\n'
            string += '\n git version: {:}\n\n'
            form += (_time(time), git_version(),)
        else:
            string += '\n git version: {:}\n\n'
            form += (git_version(),)

        return string.format(*form)


def results(mol: gto.Mole, **kwargs: np.ndarray) -> str:
        """
        this function prints the results based on either an atom- or bond-based partitioning
        """
        if kwargs['part'] in ['atoms', 'eda']:
            return atoms(mol, **kwargs)
        else:
            return bonds(mol, **kwargs)


def atoms(mol: gto.Mole, **kwargs: np.ndarray) -> str:
        """
        atom-based partitioning
        """
        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        # property of interest
        prop = kwargs['prop']
        # effective atomic charges
        charge_atom = kwargs['charge_atom']

        # ground state energy
        if prop == 'energy':

            # electronic, nuclear, and total contributions to property
            prop_el = kwargs['prop_el']
            prop_nuc = kwargs['prop_nuc']
            prop_tot = prop_el + prop_nuc
            assert prop_el.ndim == prop_nuc.ndim == 1, 'wrong choice of property'

            string += '--------------------------------------------------------------------\n'
            string += '{:^70}\n'
            string += '--------------------------------------------------------------------\n'
            string += '--------------------------------------------------------------------\n'
            string += ' atom |  electronic  |    nuclear   |     total    |  eff. charge\n'
            string += '--------------------------------------------------------------------\n'
            string += '--------------------------------------------------------------------\n'
            form += ('ground-state energy',)

            for i in range(mol.natm):
                string += ' {:<5s}|{:>+12.5f}  |{:>+12.5f}  |{:>+12.5f}  |{:>+11.3f}\n'
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), prop_el[i], prop_nuc[i], prop_tot[i], charge_atom[i])

            string += '--------------------------------------------------------------------\n'
            string += '--------------------------------------------------------------------\n'
            string += ' sum  |{:>+12.5f}  |{:>+12.5f}  |{:>+12.5f}  |{:>+11.3f}\n'
            string += '--------------------------------------------------------------------\n\n'
            form += (np.sum(prop_el), np.sum(prop_nuc), np.sum(prop_tot), np.sum(charge_atom))

            # atomization energy
            if 'prop_atom' in kwargs:

                prop_atom = kwargs['prop_atom']
                assert prop_atom.size == prop_tot.size, 'mismatch between lengths of input arrays'

                string += '--------------------------------------\n'
                string += '{:^40}\n'
                string += '--------------------------------------\n'
                string += '--------------------------------------\n'
                string += ' atom |      total   |  eff. charge\n'
                string += '--------------------------------------\n'
                string += '--------------------------------------\n'
                form += ('atomization energy',)

                for i in range(mol.natm):
                    string += ' {:<5s}|{:>+12.5f}  |{:>+11.3f}\n'
                    form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), prop_tot[i] - prop_atom[i], charge_atom[i])

                string += '--------------------------------------\n'
                string += '--------------------------------------\n'
                string += ' sum  |{:>+12.5f}  |{:>+11.3f}\n'
                string += '--------------------------------------\n\n'
                form += (np.sum(prop_tot) - np.sum(prop_atom), np.sum(charge_atom))

        # dipole moment
        elif prop == 'dipole':

            # electronic, nuclear, and total contributions to property
            prop_el = kwargs['prop_el']
            prop_nuc = kwargs['prop_nuc']
            prop_tot = prop_el + prop_nuc
            assert prop_el.ndim == prop_nuc.ndim == prop_tot.ndim == 2, 'wrong choice of property'

            # dipole unit
            if 'unit' in kwargs:
                unit = kwargs['unit'].lower()
            else:
                unit = 'au'
            assert unit in ['au', 'debye'], 'illegal unit for dipole moments. valid options are: `au` (default) or `debye`'
            scaling = 1. if unit == 'au' else AU_TO_DEBYE

            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'
            string += '{:^125}\n'
            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'
            string += '      |             electronic            |               nuclear             |                total              |\n'
            string += ' atom -------------------------------------------------------------------------------------------------------------  eff. charge\n'
            string += '      |     x     /     y     /     z     |     x     /     y     /     z     |     x     /     y     /     z     |\n'
            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'
            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'
            form += ('ground-state dipole moment (unit: {:})'.format(unit),)

            for i in range(mol.natm):
                string += ' {:<5s}| {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |{:>+11.3f}\n'
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), *prop_el[i] + 1.e-10, *prop_nuc[i] + 1.e-10, \
                                           *prop_tot[i] * scaling + 1.e-10, charge_atom[i])

            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'
            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'

            string += ' sum  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |{:>+11.3f}\n'
            string += '-----------------------------------------------------------------------------------------------------------------------------------\n\n'
            form += (*np.fromiter(map(math.fsum, prop_el.T), dtype=np.float64, count=3) * scaling + 1.e-10, \
                     *np.fromiter(map(math.fsum, prop_nuc.T), dtype=np.float64, count=3) * scaling + 1.e-10, \
                     *np.fromiter(map(math.fsum, prop_tot.T), dtype=np.float64, count=3) * scaling + 1.e-10, \
                     np.sum(charge_atom))

        return string.format(*form)


def bonds(mol: gto.Mole, **kwargs: np.ndarray) -> str:
        """
        bond-based partitioning
        """
        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        # property of interest
        prop = kwargs['prop']
        # bond centres
        centres = kwargs['centres']
        # bond lengths
        dist = kwargs['dist']

        if prop == 'energy':

            # electronic and nuclear contributions to property
            prop_el = kwargs['prop_el']
            prop_nuc = kwargs['prop_nuc']
            assert prop_el[0].ndim == 1, 'wrong choice of property'

            string += '--------------------------------------------------------\n'
            string += '{:^55}\n'
            string += '--------------------------------------------------------\n'
            string += '  MO  |   electronic  |    atom(s)    |   bond length\n'
            string += '--------------------------------------------------------\n'
            form += ('ground-state energy',)

            for i in range(2):
                string += '--------------------------------------------------------\n'
                string += '{:^55}\n'
                string += '--------------------------------------------------------\n'
                form += ('alpha-spin',) if i == 0 else ('beta-spin',)
                for j in range(prop_el[i].size):
                    core = centres[i][j, 0] == centres[i][j, 1]
                    string += '  {:>2d}  |{:>+12.5f}   |    {:<11s}|  {:>9s}\n'
                    form += (j, prop_el[i][j], \
                             '{:s}{:d}'.format(mol.atom_symbol(centres[0][j, 0]), centres[i][j, 0]) if core \
                             else '{:s}{:d}-{:s}{:d}'.format(mol.atom_symbol(centres[i][j, 0]), centres[i][j, 0], \
                                                               mol.atom_symbol(centres[i][j, 1]), centres[i][j, 1]), \
                             '' if core else '{:>.3f}'.format(dist[centres[i][j, 0], centres[i][j, 1]]),)

            string += '--------------------------------------------------------\n'
            string += '--------------------------------------------------------\n'
            string += ' sum  |{:>+12.5f}   |\n'
            form += (np.sum(prop_el[0]) + np.sum(prop_el[1]),)

            string += '-----------------------\n'
            string += ' nuc  |{:>+12.5f}   |\n'
            form += (np.sum(prop_nuc),)

            string += '-----------------------\n'
            string += '-----------------------\n'
            string += ' tot  |{:>12.5f}   |\n'
            string += '-----------------------\n\n'
            form += (np.sum(prop_el[0]) + np.sum(prop_el[1]) + np.sum(prop_nuc),)

        elif prop == 'dipole':

            # electronic and nuclear contributions to property
            prop_el = kwargs['prop_el']
            prop_nuc = kwargs['prop_nuc']
            assert prop_el[0].ndim == 2, 'wrong choice of property'

            string += '----------------------------------------------------------------------------\n'
            string += '{:^75}\n'
            string += '----------------------------------------------------------------------------\n'
            string += '  MO  |             electronic            |    atom(s)    |   bond length\n'
            string += '----------------------------------------------------------------------------\n'
            string += '      |     x     /     y     /     z     |\n'
            string += '----------------------------------------------------------------------------\n'
            string += '----------------------------------------------------------------------------\n'
            form += ('ground-state dipole moment',)

            for i in range(2):
                string += '----------------------------------------------------------------------------\n'
                string += '{:^75}\n'
                string += '----------------------------------------------------------------------------\n'
                form += ('alpha-spin',) if i == 0 else ('beta-spin',)
                for j in range(prop_el[i].shape[0]):
                    core = centres[i][j, 0] == centres[i][j, 1]
                    string += '  {:>2d}  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |    {:<11s}|  {:>9s}\n'
                    form += (j, *prop_el[i][j] + 1.e-10, \
                             '{:s}{:d}'.format(mol.atom_symbol(centres[i][j, 0]), centres[i][j, 0]) if core \
                             else '{:s}{:d}-{:s}{:d}'.format(mol.atom_symbol(centres[i][j, 0]), centres[i][j, 0], \
                                                               mol.atom_symbol(centres[i][j, 1]), centres[i][j, 1]), \
                             '' if core else '{:>.3f}'. \
                             format(dist[centres[i][j, 0], centres[i][j, 1]]),)

            string += '----------------------------------------------------------------------------\n'
            string += '----------------------------------------------------------------------------\n'

            string += ' sum  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |\n'
            form += (*(np.fromiter(map(math.fsum, prop_el[0].T), dtype=np.float64, count=3) + \
                     np.fromiter(map(math.fsum, prop_el[1].T), dtype=np.float64, count=3)) + 1.e-10,)

            string += '----------------------------------------------------------------------------\n'
            string += ' nuc  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |\n'
            form += (*np.fromiter(map(math.fsum, prop_nuc.T), dtype=np.float64, count=3) + 1.e-10,)

            string += '----------------------------------------------------------------------------\n'
            string += '----------------------------------------------------------------------------\n'

            string += ' tot  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |\n'
            string += '----------------------------------------------------------------------------\n\n'
            form += (*(np.fromiter(map(math.fsum, prop_el[0].T), dtype=np.float64, count=3) + \
                     np.fromiter(map(math.fsum, prop_el[1].T), dtype=np.float64, count=3) + \
                     np.fromiter(map(math.fsum, prop_nuc.T), dtype=np.float64, count=3)) + 1.e-10,)

        return string.format(*form)


def _ref(mol: gto.Mole, ref: str, xc: str) -> str:
        """
        this functions returns the correct (formatted) reference function
        """
        if ref == 'restricted':
            if mol.spin == 0:
                return 'RHF' if xc == '' else 'RKS'
            else:
                return 'ROHF' if xc == '' else 'ROKS'
        else:
            return 'UHF' if xc == '' else 'UKS'


def _loc(loc: str) -> str:
        """
        this functions returns the correct (formatted) localization
        """
        if loc == '':
            return 'none'
        else:
            return loc


def _xc(xc: str) -> str:
        """
        this functions returns the correct (formatted) xc functional
        """
        if xc == '':
            return 'none'
        else:
            return xc


def _time(time: float) -> str:
        """
        this function returns time as a HH:MM:SS string
        """
        # hours, minutes, and seconds
        hours = time // 3600.
        minutes = (time - (time // 3600) * 3600.) // 60.
        seconds = time - hours * 3600. - minutes * 60.

        # init time string
        string: str = ''
        form: Tuple[float, ...] = ()

        # write time string
        if hours > 0:
            string += '{:.0f}h '
            form += (hours,)
        if minutes > 0:
            string += '{:.0f}m '
            form += (minutes,)
        string += '{:.2f}s'
        form += (seconds,)

        return string.format(*form)


