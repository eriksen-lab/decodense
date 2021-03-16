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
from pyscf import gto, lo
from typing import Dict, Tuple, List, Union, Any

from .decomp import PROP_KEYS, DecompCls
from .tools import git_version
from .data import AU_TO_KCAL_MOL, AU_TO_EV, AU_TO_KJ_MOL, AU_TO_DEBYE

TOLERANCE = 1.e-10


def info(decomp: DecompCls, mol: Union[None, gto.Mole] = None, **kwargs: float) -> str:
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
        string += ' property           =  {:}\n'
        string += ' basis set          =  {:}\n'
        string += ' partitioning       =  {:}\n'
        string += ' assignment         =  {:}\n'
        form += (decomp.prop, decomp.basis, decomp.part, decomp.pop,)
        string += ' localization       =  {:}\n'
        form += (_format(decomp.loc),)
        string += ' xc functional      =  {:}\n'
        form += (_format(decomp.xc),)
        string += ' conv. tolerance    =  {:}\n'
        form += (_format(decomp.conv_tol),)
        if decomp.xc != '':
            string += ' ks-dft grid level  =  {:}\n'
            form += (_format(decomp.grid_level),)
        if mol is not None:
            string += '\n reference funct.   =  {:}\n'
            string += ' point group        =  {:}\n'
            string += ' electrons          =  {:d}\n'
            string += ' alpha electrons    =  {:d}\n'
            string += ' beta electrons     =  {:d}\n'
            string += ' basis functions    =  {:d}\n'
            form += (_ref(mol, decomp.ref, decomp.xc), mol.groupname, mol.nelectron, \
                     mol.alpha.size, mol.beta.size, mol.nao_nr(),)
            if 'ss' in kwargs:
                string += ' spin: <S^2>        =  {:.3f}\n'
                form += (kwargs['ss'] + 1.e-6,)
            if 's' in kwargs:
                string += ' spin: 2*S + 1      =  {:.3f}\n'
                form += (kwargs['s'] + 1.e-6,)

        # timing and git version
        if 'time' in kwargs:
            string += '\n total time         =  {:}\n'
            string += '\n git version: {:}\n\n'
            form += (_time(kwargs['time']), git_version(),)
        else:
            string += '\n git version: {:}\n\n'
            form += (git_version(),)

        return string.format(*form)


def results(mol: gto.Mole, header: str, **kwargs: np.ndarray) -> str:
        """
        this function prints the results based on either an atom- or bond-based partitioning
        """
        pmol = lo.iao.reference_mol(mol)
        if 'centres' in kwargs:
            return bonds(pmol, header, **kwargs)
        else:
            return atoms(pmol, header, **kwargs)


def atoms(mol: gto.Mole, header: str, **kwargs: np.ndarray) -> str:
        """
        atom-based partitioning
        """
        # init string
        string: str = ''

        # electronic, structural, and total contributions to property
        prop = {prop_key: kwargs[prop_key] for prop_key in PROP_KEYS[-2:]}
        prop['tot'] = prop['el'] + prop['struct']
        # effective atomic charges
        charge_atom = kwargs['charge_atom']

        # scalar property
        if prop['el'].ndim == prop['struct'].ndim == 1:

            # remaining energetic contributions
            for prop_key in PROP_KEYS[:-2]:
                prop[prop_key] = kwargs[prop_key]

            # formatting
            length = 149
            divider = '-' * length + '\n'

            # units
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

            # headers
            string += divider
            string += f'{f"{header} (unit: {unit})":^{length}}\n'
            string += divider
            string += divider
            string += f'{"atom":^6}|{"coulomb":^14}|{"exchange":^14}|{"kinetic":^14}|{"nuc. attr.":^14}|{"xc":^14}||'
            string += f'{"electronic":^14}||{"structural":^14}|||{"total":^14}|||{"part. charge":^16}\n'
            string += divider
            string += divider

            # individual contributions
            for i in range(mol.natm):
                string += f' {f"{mol.atom_symbol(i)}{i}":<5s}|' \
                          f'{prop["coul"][i] * scaling:>+12.5f}  |' \
                          f'{prop["exch"][i] * scaling:>+12.5f}  |' \
                          f'{prop["kin"][i] * scaling:>+12.5f}  |' \
                          f'{prop["rdm_att"][i] * scaling:>+12.5f}  |' \
                          f'{prop["xc"][i] * scaling:>+12.5f}  ||' \
                          f'{prop["el"][i] * scaling:>+12.5f}  ||' \
                          f'{prop["struct"][i] * scaling:>+12.5f}  |||' \
                          f'{prop["tot"][i] * scaling:>+12.5f}  |||' \
                          f'{charge_atom[i]:>+11.3f}\n'
            string += divider
            string += divider

            # total contributions
            string += f'{"sum":^6}|' \
                      f'{np.sum(prop["coul"]) * scaling:>+12.5f}  |' \
                      f'{np.sum(prop["exch"]) * scaling:>+12.5f}  |' \
                      f'{np.sum(prop["kin"]) * scaling:>+12.5f}  |' \
                      f'{np.sum(prop["rdm_att"]) * scaling:>+12.5f}  |' \
                      f'{np.sum(prop["xc"]) * scaling:>+12.5f}  ||' \
                      f'{np.sum(prop["el"]) * scaling:>+12.5f}  ||' \
                      f'{np.sum(prop["struct"]) * scaling:>+12.5f}  |||' \
                      f'{np.sum(prop["tot"]) * scaling:>+12.5f}  |||' \
                      f'{0.:>11.3f}\n'
            string += divider + '\n'

        # tensor property
        elif prop['el'].ndim == prop['struct'].ndim == 2:

            # formatting
            length = 131
            divider = '-' * length + '\n'
            length_2 = 35
            divider_2 = '-' * length_2

            # units
            if 'unit' in kwargs:
                unit = kwargs['unit'].lower()
            else:
                unit = 'au'
            assert unit in ['au', 'debye'], 'illegal unit for dipole moments. ' \
                                            'valid options are: `au` (default) or `debye`.'
            scaling = 1.
            if unit == 'debye':
                scaling = AU_TO_DEBYE

            # headers
            string += divider
            string += f'{f"{header} (unit: {unit})":^{length}}\n'
            string += divider
            string += f'      |{"electronic":^35}|{"structural":^35}|{"total":^35}|\n'
            string += f'{"atom":^6}|' + divider_2 + '|' + divider_2 + '|' + divider_2 + f'|{"part. charge":^16}\n'
            string += '      |' \
                      f'{"x":^11}/{"y":^11}/{"z":^11}|' \
                      f'{"x":^11}/{"y":^11}/{"z":^11}|' \
                      f'{"x":^11}/{"y":^11}/{"z":^11}|\n'
            string += divider
            string += divider

            # individual contributions
            for i in range(mol.natm):
                string += f' {f"{mol.atom_symbol(i)}{i}":<5s}|' \
                          f' {prop["el"][i][0] * scaling + TOLERANCE:>+8.3f}  /' \
                          f' {prop["el"][i][1] * scaling + TOLERANCE:>+8.3f}  /' \
                          f' {prop["el"][i][2] * scaling + TOLERANCE:>+8.3f}  |' \
                          f' {prop["struct"][i][0] * scaling + TOLERANCE:>+8.3f}  /' \
                          f' {prop["struct"][i][1] * scaling + TOLERANCE:>+8.3f}  /' \
                          f' {prop["struct"][i][2] * scaling + TOLERANCE:>+8.3f}  |' \
                          f' {prop["tot"][i][0] * scaling + TOLERANCE:>+8.3f}  /' \
                          f' {prop["tot"][i][1] * scaling + TOLERANCE:>+8.3f}  /' \
                          f' {prop["tot"][i][2] * scaling + TOLERANCE:>+8.3f}  |' \
                          f'{charge_atom[i]:>+11.3f}\n'
            string += divider
            string += divider

            # total contributions
            sum_el = np.fromiter(map(math.fsum, prop['el'].T), dtype=np.float64, count=3)
            sum_struct = np.fromiter(map(math.fsum, prop['struct'].T), dtype=np.float64, count=3)
            sum_tot = np.fromiter(map(math.fsum, prop['tot'].T), dtype=np.float64, count=3)
            string += f'{"sum":^6}|' \
                      f' {sum_el[0] * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {sum_el[1] * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {sum_el[2] * scaling + TOLERANCE:>+8.3f}  |' \
                      f' {sum_struct[0] * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {sum_struct[1] * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {sum_struct[2] * scaling + TOLERANCE:>+8.3f}  |' \
                      f' {sum_tot[0] * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {sum_tot[1] * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {sum_tot[2] * scaling + TOLERANCE:>+8.3f}  |' \
                      f'{0.:>11.3f}\n'
            string += divider + '\n'

        return string


def bonds(mol: gto.Mole, header: str, **kwargs: np.ndarray) -> str:
        """
        bond-based partitioning
        """
        # init string
        string: str = ''

        # bond centres
        centres = kwargs['centres']
        # bond lengths
        dist = kwargs['dist']
        # electronic and structlear contributions to property
        prop_el = kwargs['el']
        prop_struct = kwargs['struct']

        # scalar property
        if prop_el[0].ndim == prop_struct.ndim == 1:

            length = 56
            divider = '-' * length + '\n'
            length_2 = 23
            divider_2 = '-' * length_2 + '\n'

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

            string += divider
            string += f'{f"{header} (unit: {unit})":^{length}}\n'
            string += divider
            string += f'{"MO":^6}|{"electronic":^15}|{"atom(s)":^15}|{"bond length/Ang":^17}\n'
            string += divider

            for i, spin in enumerate(('alpha-spin', 'beta-spin')):
                string += divider
                string += f'{spin:^{length}}\n'
                string += divider
                for j in range(centres[i].shape[0]):
                    atom = f'{mol.atom_symbol(centres[i][j, 0]):s}{centres[i][j, 0]:d}'
                    core = centres[i][j, 0] == centres[i][j, 1]
                    if core:
                        string += f'  {j:>2d}  |' \
                                  f'{prop_el[i][j] * scaling:>+12.5f}   |' \
                                  f'    {atom:<11s}|\n'
                    else:
                        atom += f'-{mol.atom_symbol(centres[i][j, 1]):s}{centres[i][j, 1]:d}'
                        rr = f'{dist[centres[i][j, 0], centres[i][j, 1]]:.3f}'
                        string += f'  {j:>2d}  |' \
                                  f'{prop_el[i][j] * scaling:>+12.5f}   |' \
                                  f'    {atom:<11s}|' \
                                  f'  {rr:>9s}\n'

            string += divider
            string += divider
            string += f'{"sum":^6}|{(np.sum(prop_el[0]) + np.sum(prop_el[1])) * scaling:>+12.5f}   |\n'

            string += divider_2
            string += f'{"struct":^6}|{np.sum(prop_struct) * scaling:>+12.5f}   |\n'

            string += divider_2
            string += divider_2
            string += f'{"tot":^6}|{(np.sum(prop_el[0]) + np.sum(prop_el[1]) + np.sum(prop_struct)) * scaling:>12.5f}   |\n'
            string += divider_2 + '\n'

        # tensor property
        elif prop_el[0].ndim == prop_struct.ndim == 2:

            length = 76
            divider = '-' * length + '\n'
            length_2 = 35
            divider_2 = '-' * length_2

            if 'unit' in kwargs:
                unit = kwargs['unit'].lower()
            else:
                unit = 'au'
            assert unit in ['au', 'debye'], 'illegal unit for dipole moments. ' \
                                            'valid options are: `au` (default) or `debye`.'
            scaling = 1.
            if unit == 'debye':
                scaling = AU_TO_DEBYE

            string += divider
            string += f'{f"{header} (unit: {unit})":^{length}}\n'
            string += divider
            string += f'{"":^6}|{"electronic":^35}|{"":^15}|\n'
            string += f'{"MO":^6}|' + divider_2 + f'|{"atom(s)":^15}|' + f'{"bond length/Ang":^17}\n'
            string += f'{"":^6}|{"x":^11}/{"y":^11}/{"z":^11}|{"":^15}|\n'
            string += divider

            for i, spin in enumerate(('alpha-spin', 'beta-spin')):
                string += divider
                string += f'{spin:^{length}}\n'
                string += divider
                for j in range(centres[i].shape[0]):
                    atom = f'{mol.atom_symbol(centres[i][j, 0]):s}{centres[i][j, 0]:d}'
                    core = centres[i][j, 0] == centres[i][j, 1]
                    if core:
                        string += f'  {j:>2d}  |' \
                                  f' {prop_el[i][j][0] * scaling + TOLERANCE:>+8.3f}  /' \
                                  f' {prop_el[i][j][1] * scaling + TOLERANCE:>+8.3f}  /' \
                                  f' {prop_el[i][j][2] * scaling + TOLERANCE:>+8.3f}  |' \
                                  f'    {atom:<11s}|\n'
                    else:
                        atom += f'-{mol.atom_symbol(centres[i][j, 1]):s}{centres[i][j, 1]:d}'
                        rr = f'{dist[centres[i][j, 0], centres[i][j, 1]]:.3f}'
                        string += f'  {j:>2d}  |' \
                                  f' {prop_el[i][j][0] * scaling + TOLERANCE:>+8.3f}  /' \
                                  f' {prop_el[i][j][1] * scaling + TOLERANCE:>+8.3f}  /' \
                                  f' {prop_el[i][j][2] * scaling + TOLERANCE:>+8.3f}  |' \
                                  f'    {atom:<11s}|' \
                                  f'  {rr:>9s}\n'

            string += divider
            string += divider

            sum_el = (np.fromiter(map(math.fsum, prop_el[0].T), dtype=np.float64, count=3) + \
                      np.fromiter(map(math.fsum, prop_el[1].T), dtype=np.float64, count=3))
            sum_struct = np.fromiter(map(math.fsum, prop_struct.T), dtype=np.float64, count=3)

            string += f'{"sum":^6}|' \
                      f' {sum_el[0] * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {sum_el[1] * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {sum_el[2] * scaling + TOLERANCE:>+8.3f}  |\n'

            string += divider
            string += f'{"struct":^6}|' \
                      f' {sum_struct[0] * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {sum_struct[1] * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {sum_struct[2] * scaling + TOLERANCE:>+8.3f}  |\n'

            string += divider
            string += divider

            string += f'{"tot":^6}|' \
                      f' {(sum_el[0] + sum_struct[0]) * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {(sum_el[1] + sum_struct[1]) * scaling + TOLERANCE:>+8.3f}  /' \
                      f' {(sum_el[2] + sum_struct[2]) * scaling + TOLERANCE:>+8.3f}  |\n'
            string += divider + '\n'

        return string


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


def _format(opt: Any) -> str:
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


