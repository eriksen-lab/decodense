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
from .data import AU_TO_KCAL_MOL, AU_TO_EV, AU_TO_KJ_MOL, AU_TO_DEBYE


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
        form += (_format(decomp.loc),)
        string += ' xc functional      =  {:}\n'
        form += (_format(decomp.xc),)
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


def results(mol: gto.Mole, header: str, **kwargs: np.ndarray) -> str:
        """
        this function prints the results based on either an atom- or bond-based partitioning
        """
        if kwargs['part'] in ['atoms', 'eda']:
            return atoms(mol, header, **kwargs)
        else:
            return bonds(mol, header, **kwargs)


def atoms(mol: gto.Mole, header: str, **kwargs: np.ndarray) -> str:
        """
        atom-based partitioning
        """
        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        # electronic, nuclear, and total contributions to property
        prop_el = kwargs['prop_el']
        prop_nuc = kwargs['prop_nuc']
        prop_tot = prop_el + prop_nuc
        # effective atomic charges
        charge_atom = kwargs['charge_atom']

        # scalar property
        if prop_el.ndim == prop_nuc.ndim == 1:

            if 'unit' in kwargs:
                unit = kwargs['unit'].lower()
            else:
                unit = 'au'
            assert unit in ['au', 'kcal_mol', 'ev', 'kj_mol'], 'illegal unit for energies. ' \
                                                               'valid options are: ' \
                                                               '`au` (default), `kcal_mol`, ' \
                                                               '`ev`, and `kj_mol`.'
            scaling = 1.
            if unit == 'kcal_mol':
                scaling = AU_TO_KCAL_MOL
            elif unit == 'ev':
                scaling = AU_TO_EV
            elif unit == 'kj_mol':
                scaling = AU_TO_KJ_MOL

            string += '-' * 69 + '\n'
            string += '{:^69}\n'
            string += '-' * 69 + '\n'
            string += '-' * 69 + '\n'
            string += ' atom |  electronic  |    nuclear   |     total    |  part. charge\n'
            string += '-' * 69 + '\n'
            string += '-' * 69 + '\n'
            form += ('{:} (unit: {:})'.format(header, unit),)

            for i in range(mol.natm):
                string += ' {:<5s}|{:>+12.5f}  |{:>+12.5f}  |{:>+12.5f}  |{:>+11.3f}\n'
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), \
                                           prop_el[i] * scaling, \
                                           prop_nuc[i] * scaling, \
                                           prop_tot[i] * scaling, \
                                           mol.atom_charge(i) - charge_atom[i],)

            string += '-' * 69 + '\n'
            string += '-' * 69 + '\n'
            string += ' sum  |{:>+12.5f}  |{:>+12.5f}  |{:>+12.5f}  |{:>11.3f}\n'
            string += '-' * 69 + '\n\n'
            form += (np.sum(prop_el) * scaling, \
                     np.sum(prop_nuc) * scaling, \
                     np.sum(prop_tot) * scaling, \
                     0.)

        # tensor property
        elif prop_el.ndim == prop_nuc.ndim == 2:

            if 'unit' in kwargs:
                unit = kwargs['unit'].lower()
            else:
                unit = 'au'
            assert unit in ['au', 'debye'], 'illegal unit for dipole moments. ' \
                                            'valid options are: `au` (default) or `debye`.'
            scaling = 1.
            if unit == 'debye':
                scaling = AU_TO_DEBYE

            string += '-' * 131 + '\n'
            string += '{:^131}\n'
            string += '-' * 131 + '\n'
            string += f'      |{"electronic":^35}|{"nuclear":^35}|{"total":^35}|\n'
            string += ' atom ' + '-' * 109 + '  part. charge\n'
            string += '      |' \
                      f'{"x":^11}/{"y":^11}/{"z":^11}|' \
                      f'{"x":^11}/{"y":^11}/{"z":^11}|' \
                      f'{"x":^11}/{"y":^11}/{"z":^11}|\n'
            string += '-' * 131 + '\n'
            string += '-' * 131 + '\n'
            form += ('{:} (unit: {:})'.format(header, unit),)

            for i in range(mol.natm):
                string += ' {:<5s}|' \
                          ' {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |' \
                          ' {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |' \
                          ' {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |' \
                          '{:>+11.3f}\n'
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), \
                                           *prop_el[i] * scaling + 1.e-10, \
                                           *prop_nuc[i] * scaling + 1.e-10, \
                                           *prop_tot[i] * scaling + 1.e-10, \
                                           mol.atom_charge(i) - charge_atom[i],)

            string += '-' * 131 + '\n'
            string += '-' * 131 + '\n'

            string += ' sum  |' \
                      ' {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |' \
                      ' {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |' \
                      ' {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |' \
                      '{:>11.3f}\n'
            string += '-' * 131 + '\n\n'
            form += (*np.fromiter(map(math.fsum, prop_el.T), dtype=np.float64, count=3) * scaling + 1.e-10, \
                     *np.fromiter(map(math.fsum, prop_nuc.T), dtype=np.float64, count=3) * scaling + 1.e-10, \
                     *np.fromiter(map(math.fsum, prop_tot.T), dtype=np.float64, count=3) * scaling + 1.e-10, \
                     0.)

        return string.format(*form)


def bonds(mol: gto.Mole, header: str, **kwargs: np.ndarray) -> str:
        """
        bond-based partitioning
        """
        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        # bond centres
        centres = kwargs['centres']
        # bond lengths
        dist = kwargs['dist']
        # electronic and nuclear contributions to property
        prop_el = kwargs['prop_el']
        prop_nuc = kwargs['prop_nuc']

        # scalar property
        if prop_el[0].ndim == prop_nuc.ndim == 1:

            if 'unit' in kwargs:
                unit = kwargs['unit'].lower()
            else:
                unit = 'au'
            assert unit in ['au', 'kcal_mol', 'ev', 'kj_mol'], 'illegal unit for energies. ' \
                                                               'valid options are: ' \
                                                               '`au` (default), `kcal_mol`, ' \
                                                               '`ev`, and `kj_mol`.'
            scaling = 1.
            if unit == 'kcal_mol':
                scaling = AU_TO_KCAL_MOL
            elif unit == 'ev':
                scaling = AU_TO_EV
            elif unit == 'kj_mol':
                scaling = AU_TO_KJ_MOL

            string += '-' * 56 + '\n'
            string += '{:^56}\n'
            string += '-' * 56 + '\n'
            string += '  MO  |   electronic  |    atom(s)    |   bond length\n'
            string += '-' * 56 + '\n'
            form += ('{:} (unit: {:})'.format(header, unit),)

            for i in range(2):
                string += '-' * 56 + '\n'
                string += '{:^56}\n'
                string += '-' * 56 + '\n'
                form += ('alpha-spin',) if i == 0 else ('beta-spin',)
                for j in range(centres[i].shape[0]):
                    core = centres[i][j, 0] == centres[i][j, 1]
                    string += '  {:>2d}  |{:>+12.5f}   |    {:<11s}|  {:>9s}\n'
                    form += (j, prop_el[i][j] * scaling, \
                             '{:s}{:d}'.format(mol.atom_symbol(centres[0][j, 0]), centres[i][j, 0]) if core \
                             else '{:s}{:d}-{:s}{:d}'.format(mol.atom_symbol(centres[i][j, 0]), centres[i][j, 0], \
                                                             mol.atom_symbol(centres[i][j, 1]), centres[i][j, 1]), \
                             '' if core else '{:>.3f}'.format(dist[centres[i][j, 0], centres[i][j, 1]]),)

            string += '-' * 56 + '\n'
            string += '-' * 56 + '\n'
            string += ' sum  |{:>+12.5f}   |\n'
            form += ((np.sum(prop_el[0]) + np.sum(prop_el[1])) * scaling,)

            string += '-' * 23 + '\n'
            string += ' nuc  |{:>+12.5f}   |\n'
            form += (np.sum(prop_nuc) * scaling,)

            string += '-' * 23 + '\n'
            string += '-' * 23 + '\n'
            string += ' tot  |{:>12.5f}   |\n'
            string += '-' * 23 + '\n\n'
            form += ((np.sum(prop_el[0]) + np.sum(prop_el[1]) + np.sum(prop_nuc)) * scaling,)

        # tensor property
        elif prop_el[0].ndim == prop_nuc.ndim == 2:

            if 'unit' in kwargs:
                unit = kwargs['unit'].lower()
            else:
                unit = 'au'
            assert unit in ['au', 'debye'], 'illegal unit for dipole moments. ' \
                                            'valid options are: `au` (default) or `debye`.'
            scaling = 1.
            if unit == 'debye':
                scaling = AU_TO_DEBYE

            string += '-' * 76 + '\n'
            string += '{:^76}\n'
            string += '-' * 76 + '\n'
            string += f'  MO  |{"electronic":^35}|{"atom(s)":^15}|   bond length\n'
            string += '-' * 76 + '\n'
            string += f'      |{"x":^11}/{"y":^11}/{"z":^11}|\n'
            string += '-' * 76 + '\n'
            form += ('{:} (unit: {:})'.format(header, unit),)

            for i in range(2):
                string += '-' * 76 + '\n'
                string += '{:^76}\n'
                string += '-' * 76 + '\n'
                form += ('alpha-spin',) if i == 0 else ('beta-spin',)
                for j in range(centres[i].shape[0]):
                    core = centres[i][j, 0] == centres[i][j, 1]
                    string += '  {:>2d}  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |    {:<11s}|  {:>9s}\n'
                    form += (j, *prop_el[i][j] * scaling + 1.e-10, \
                             '{:s}{:d}'.format(mol.atom_symbol(centres[i][j, 0]), centres[i][j, 0]) if core \
                             else '{:s}{:d}-{:s}{:d}'.format(mol.atom_symbol(centres[i][j, 0]), centres[i][j, 0], \
                                                             mol.atom_symbol(centres[i][j, 1]), centres[i][j, 1]), \
                             '' if core else '{:>.3f}'.format(dist[centres[i][j, 0], centres[i][j, 1]]),)

            string += '-' * 76 + '\n'
            string += '-' * 76 + '\n'

            string += ' sum  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |\n'
            form += (*(np.fromiter(map(math.fsum, prop_el[0].T), dtype=np.float64, count=3) + \
                       np.fromiter(map(math.fsum, prop_el[1].T), dtype=np.float64, count=3)) * scaling + 1.e-10,)

            string += '-' * 76 + '\n'
            string += ' nuc  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |\n'
            form += (*np.fromiter(map(math.fsum, prop_nuc.T), dtype=np.float64, count=3) * scaling + 1.e-10,)

            string += '-' * 76 + '\n'
            string += '-' * 76 + '\n'

            string += ' tot  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |\n'
            string += '-' * 76 + '\n\n'
            form += (*(np.fromiter(map(math.fsum, prop_el[0].T), dtype=np.float64, count=3) + \
                       np.fromiter(map(math.fsum, prop_el[1].T), dtype=np.float64, count=3) + \
                       np.fromiter(map(math.fsum, prop_nuc.T), dtype=np.float64, count=3)) * scaling + 1.e-10,)

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


def _format(opt: str) -> str:
        """
        this functions returns the correct formatting
        """
        if opt == '':
            return 'none'
        else:
            return opt


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


