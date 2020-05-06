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
from typing import Dict, Tuple, List, Union, Any

from .decomp import DecompCls
from .tools import git_version


def collect_res(decomp: DecompCls, mol: gto.Mole) -> Dict[str, Any]:
        res: Dict[str, Any] = {'prop_el': decomp.prop_el, 'prop_nuc': decomp.prop_nuc, \
                               'pop': decomp.pop, 'loc': _loc(decomp), 'xc': _xc(decomp), \
                               'ref': _ref(decomp, mol), 'time': decomp.time, 'sym': mol.groupname}
        if decomp.pop_atom is not None:
            res['pop_atom'] = decomp.pop_atom
        if decomp.centres is not None:
            res['centres'] = decomp.centres
        return res


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
        string += ' assignment         =  {:}\n'
        form += (decomp.basis, decomp.pop,)
        string += ' localization       =  {:}\n'
        form += (_loc(decomp),)
        string += ' xc functional      =  {:}\n'
        form += (_xc(decomp),)
        if mol is not None:
            string += '\n reference funct.   =  {:}\n'
            string += ' point group        =  {:}\n'
            string += ' electrons          =  {:d}\n'
            string += ' alpha electrons    =  {:d}\n'
            string += ' beta electrons     =  {:d}\n'
            string += ' spin: <S^2>        =  {:.3f}\n'
            string += ' spin: 2*S + 1      =  {:.3f}\n'
            string += ' basis functions    =  {:d}\n'
            form += (_ref(decomp, mol), mol.groupname, mol.nelectron, mol.alpha.size, mol.beta.size, \
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


def results(decomp: DecompCls, mol: gto.Mole, prop: str, **kwargs: np.ndarray) -> str:
        """
        this function prints the results based on either an atom- or bond-based partitioning
        """
        if decomp.part == 'atoms':
            return atoms(decomp, mol, prop, **kwargs)
        else: # bonds
            if prop == 'atomization':
                raise NotImplementedError('atomization energies are not implemented for bond-wise partitioning')
            return bonds(decomp, mol, prop, **kwargs)


def atoms(decomp: DecompCls, mol: gto.Mole, prop: str, **kwargs: np.ndarray) -> str:
        """
        atom-based partitioning
        """
        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        if prop == 'energy':

            prop_el = kwargs['prop_el']
            prop_nuc = kwargs['prop_nuc']
            prop_tot = prop_el + prop_nuc
            assert prop_el.ndim == prop_nuc.ndim == 1, 'wrong choice of property'
            pop_atom = kwargs['pop_atom']

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
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), prop_el[i], prop_nuc[i], prop_tot[i], -1. * pop_atom[i])

            string += '--------------------------------------------------------------------\n'
            string += '--------------------------------------------------------------------\n'
            string += ' sum  |{:>+12.5f}  |{:>+12.5f}  |{:>+12.5f}  |{:>+11.3f}\n'
            string += '--------------------------------------------------------------------\n\n'
            form += (np.sum(prop_el), np.sum(prop_nuc), np.sum(prop_tot), -1. * np.sum(pop_atom))

        if prop == 'atomization':

            prop_tot = kwargs['prop_el'] + kwargs['prop_nuc']
            prop_atom = kwargs['prop_atom']
            assert prop_tot.ndim == prop_atom.ndim == 1, 'wrong choice of property'
            assert prop_tot.size == prop_atom.size, 'mismatch between lengths of input arrays'
            pop_atom = kwargs['pop_atom']

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
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), prop_tot[i] - prop_atom[i], -1. * pop_atom[i])

            string += '--------------------------------------\n'
            string += '--------------------------------------\n'
            string += ' sum  |{:>+12.5f}  |{:>+11.3f}\n'
            string += '--------------------------------------\n\n'
            form += (np.sum(prop_tot) - np.sum(prop_atom), -1. * np.sum(pop_atom))

        elif prop == 'dipole':

            prop_el = kwargs['prop_el']
            prop_nuc = kwargs['prop_nuc']
            prop_tot = prop_el + prop_nuc
            assert prop_el.ndim == prop_nuc.ndim == prop_tot.ndim == 2, 'wrong choice of property'
            pop_atom = kwargs['pop_atom']

            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'
            string += '{:^125}\n'
            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'
            string += '      |             electronic            |               nuclear             |                total              |\n'
            string += ' atom -------------------------------------------------------------------------------------------------------------  eff. charge\n'
            string += '      |     x     /     y     /     z     |     x     /     y     /     z     |     x     /     y     /     z     |\n'
            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'
            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'
            form += ('ground-state dipole moment',)

            for i in range(mol.natm):
                string += ' {:<5s}| {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |{:>+11.3f}\n'
                form += ('{:s}{:d}'.format(mol.atom_symbol(i), i), *prop_el[i] + 1.e-10, *prop_nuc[i] + 1.e-10, *prop_tot[i] + 1.e-10, -1. * pop_atom[i])

            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'
            string += '-----------------------------------------------------------------------------------------------------------------------------------\n'

            string += ' sum  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  | {:>+8.3f}  / {:>+8.3f}  / {:>+8.3f}  |{:>+11.3f}\n'
            string += '-----------------------------------------------------------------------------------------------------------------------------------\n\n'
            form += (*np.fromiter(map(math.fsum, prop_el.T), dtype=np.float64, count=3) + 1.e-10, \
                     *np.fromiter(map(math.fsum, prop_nuc.T), dtype=np.float64, count=3) + 1.e-10, \
                     *np.fromiter(map(math.fsum, prop_tot.T), dtype=np.float64, count=3) + 1.e-10, \
                     -1. * np.sum(pop_atom))

        return string.format(*form)


def bonds(decomp: DecompCls, mol: gto.Mole, prop: str, **kwargs: np.ndarray) -> str:
        """
        bond-based partitioning
        """
        # init string & form
        string: str = ''
        form: Tuple[Any, ...] = ()

        # inter-atomic distance array
        dist = gto.mole.inter_distance(mol) * lib.param.BOHR

        if prop == 'energy':

            prop_el = kwargs['prop_el']
            prop_nuc = kwargs['prop_nuc']
            assert prop_el[0].ndim == 1, 'wrong choice of property'
            centres = kwargs['centres']

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

            prop_el = kwargs['prop_el']
            prop_nuc = kwargs['prop_nuc']
            assert prop_el[0].ndim == 2, 'wrong choice of property'
            centres = kwargs['centres']

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


def _ref(decomp: DecompCls, mol: gto.Mole) -> str:
        """
        this functions returns the correct (formatted) reference function
        """
        if decomp.ref == 'restricted':
            if mol.spin == 0:
                ref = 'RHF' if decomp.xc == '' else 'RKS'
            else:
                ref = 'ROHF' if decomp.xc == '' else 'ROKS'
        else:
            ref = 'UHF' if decomp.xc == '' else 'UKS'
        return ref


def _loc(decomp: DecompCls) -> str:
        """
        this functions returns the correct (formatted) localization
        """
        if decomp.loc == '':
            return 'none'
        else:
            return decomp.loc


def _xc(decomp: DecompCls) -> str:
        """
        this functions returns the correct (formatted) xc functional
        """
        if decomp.xc == '':
            return 'none'
        else:
            return decomp.xc


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


