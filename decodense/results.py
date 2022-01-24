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

from .decomp import COMP_KEYS, DecompCls
from .tools import git_version, dim
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
        string += ' partitioning       =  {:}\n'
        string += ' assignment         =  {:}\n'
        string += ' localization       =  {:}\n'
        form += (decomp.prop, decomp.part, decomp.pop, _format(decomp.loc),)
        if mol is not None:
            string += '\n point group        =  {:}\n'
            string += ' electrons          =  {:d}\n'
            string += ' basis functions    =  {:d}\n'
            form += (mol.groupname, mol.nelectron, mol.nao_nr(),)
            if 'ss' in kwargs:
                string += ' spin: <S^2>        =  {:.3f}\n'
                form += (kwargs['ss'] + 1.e-6,)
            if 's' in kwargs:
                string += ' spin: 2*S + 1      =  {:.3f}\n'
                form += (kwargs['s'] + 1.e-6,)

        # git version
        string += '\n git version: {:}\n\n'
        form += (git_version(),)

        return string.format(*form)


def results(mol: gto.Mole, header: str, **kwargs: np.ndarray) -> str:
        """
        this function prints the results based on either an atom- or bond-based partitioning
        """
        pmol = lo.iao.reference_mol(mol)
        if 'charge_atom' in kwargs:
            return atoms(pmol, header, **kwargs)
        else:
            return orbs(pmol, header, **kwargs)


def atoms(mol: gto.Mole, header: str, **kwargs: np.ndarray) -> str:
        """
        atom-based partitioning
        """
        # init string
        string: str = ''

        # electronic, structural, and total contributions to property
        prop = {comp_key: kwargs[comp_key] for comp_key in COMP_KEYS[-2:]}
        prop['tot'] = prop['el'] + prop['struct']
        # effective atomic charges
        charge_atom = kwargs['charge_atom']

        # scalar property
        if prop['el'].ndim == prop['struct'].ndim == 1:

            # remaining energetic contributions
            for comp_key in COMP_KEYS[:-2]:
                prop[comp_key] = kwargs[comp_key]

            # formatting
            length = 189
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
            string += '\n\n' + divider
            string += f'{f"{header} (unit: {unit})":^{length}}\n'
            string += divider
            string += divider
            string += f'{"atom":^6}|{"coulomb":^15}|{"exchange":^15}|{"kinetic":^15}|'
            string += f'{"nuc. att. (G)":^15}|{"nuc. att. (L)":^15}|{"solvent":^15}|{"xc":^15}||'
            string += f'{"electronic":^15}||{"structural":^15}|||{"total":^15}|||{"part. charge":^16}\n'
            string += divider
            string += divider

            # individual contributions
            for i in range(mol.natm):
                string += f' {f"{mol.atom_symbol(i)}{i}":<5s}|' \
                          f'{prop["coul"][i] * scaling:>13.5f}  |' \
                          f'{prop["exch"][i] * scaling:>13.5f}  |' \
                          f'{prop["kin"][i] * scaling:>13.5f}  |' \
                          f'{prop["nuc_att_glob"][i] * scaling:>13.5f}  |' \
                          f'{prop["nuc_att_loc"][i] * scaling:>13.5f}  |' \
                          f'{prop["solvent"][i] * scaling:>13.5f}  |' \
                          f'{prop["xc"][i] * scaling:>13.5f}  ||' \
                          f'{prop["el"][i] * scaling:>13.5f}  ||' \
                          f'{prop["struct"][i] * scaling:>13.5f}  |||' \
                          f'{prop["tot"][i] * scaling:>13.5f}  |||' \
                          f'{charge_atom[i]:>11.3f}\n'
            string += divider
            string += divider

            # total contributions
            string += f'{"sum":^6}|' \
                      f'{np.sum(prop["coul"]) * scaling:>13.5f}  |' \
                      f'{np.sum(prop["exch"]) * scaling:>13.5f}  |' \
                      f'{np.sum(prop["kin"]) * scaling:>13.5f}  |' \
                      f'{np.sum(prop["nuc_att_glob"]) * scaling:>13.5f}  |' \
                      f'{np.sum(prop["nuc_att_loc"]) * scaling:>13.5f}  |' \
                      f'{np.sum(prop["solvent"]) * scaling:>13.5f}  |' \
                      f'{np.sum(prop["xc"]) * scaling:>13.5f}  ||' \
                      f'{np.sum(prop["el"]) * scaling:>13.5f}  ||' \
                      f'{np.sum(prop["struct"]) * scaling:>13.5f}  |||' \
                      f'{np.sum(prop["tot"]) * scaling:>13.5f}  |||' \
                      f'{np.sum(charge_atom) + TOLERANCE:>11.3f}\n'
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
            string += '\n\n' + divider
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
                          f' {prop["el"][i][0] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["el"][i][1] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["el"][i][2] * scaling + TOLERANCE:>8.3f}  |' \
                          f' {prop["struct"][i][0] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["struct"][i][1] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["struct"][i][2] * scaling + TOLERANCE:>8.3f}  |' \
                          f' {prop["tot"][i][0] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["tot"][i][1] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["tot"][i][2] * scaling + TOLERANCE:>8.3f}  |' \
                          f'{charge_atom[i]:>+11.3f}\n'
            string += divider
            string += divider

            # total contributions
            sum_el = np.fromiter(map(math.fsum, prop['el'].T), dtype=np.float64, count=3)
            sum_struct = np.fromiter(map(math.fsum, prop['struct'].T), dtype=np.float64, count=3)
            sum_tot = np.fromiter(map(math.fsum, prop['tot'].T), dtype=np.float64, count=3)
            string += f'{"sum":^6}|' \
                      f' {sum_el[0] * scaling + TOLERANCE:>8.3f}  /' \
                      f' {sum_el[1] * scaling + TOLERANCE:>8.3f}  /' \
                      f' {sum_el[2] * scaling + TOLERANCE:>8.3f}  |' \
                      f' {sum_struct[0] * scaling + TOLERANCE:>8.3f}  /' \
                      f' {sum_struct[1] * scaling + TOLERANCE:>8.3f}  /' \
                      f' {sum_struct[2] * scaling + TOLERANCE:>8.3f}  |' \
                      f' {sum_tot[0] * scaling + TOLERANCE:>8.3f}  /' \
                      f' {sum_tot[1] * scaling + TOLERANCE:>8.3f}  /' \
                      f' {sum_tot[2] * scaling + TOLERANCE:>8.3f}  |' \
                      f'{0.:>11.3f}\n'
            string += divider + '\n'

        return string


def orbs(mol: gto.Mole, header: str, **kwargs: np.ndarray) -> str:
        """
        orbital-based partitioning
        """
        # init string
        string: str = ''

        # mo occupations
        mo_occ = kwargs['mo_occ']
        # orbital symmetries
        orbsym = kwargs['orbsym']
        # index
        if kwargs['ndo']:
            sort_idx = [np.argsort(mo_occ[i]) for i in range(2)]
            mo_idx = [np.array([[sort_idx[i][j], sort_idx[i][-(j+1)]] for j in range(sort_idx[i].size // 2)]).ravel() for i in range(2)]
        else:
            mo_idx = dim(mo_occ)
        # electronic, structural, and total contributions to property
        prop = {comp_key: kwargs[comp_key] for comp_key in COMP_KEYS[-2:]}

        # scalar property
        if prop['el'][0].ndim == 1:

            # remaining energetic contributions
            for comp_key in COMP_KEYS[:-2]:
                prop[comp_key] = kwargs[comp_key]

            # formatting
            length = 155
            divider = '-' * length + '\n'
            length_2 = 27
            divider_2 = '-' * length_2 + '\n'

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
            string += '\n\n' + divider
            string += f'{f"{header} (unit: {unit})":^{length}}\n'
            string += divider
            string += f'{"MO":^10}|{"coulomb":^15}|{"exchange":^15}|{"kinetic":^15}|{"nuc. attr.":^15}|'
            string += f'{"solvent":^15}|{"xc":^15}||{"electronic":^15}||{"occupation":^15}||{"symmetry":^15}\n'
            string += divider

            # individual contributions
            for i, spin in enumerate(('alpha-spin', 'beta-spin')):
                string += divider
                string += f'{spin:^{length}}\n'
                string += divider
                for j in mo_idx[i]:
                    string += f'   {j:>2d}     |' \
                              f'{prop["coul"][i][j] * scaling:>12.5f}   |' \
                              f'{prop["exch"][i][j] * scaling:>12.5f}   |' \
                              f'{prop["kin"][i][j] * scaling:>12.5f}   |' \
                              f'{prop["nuc_att"][i][j] * scaling:>12.5f}   |' \
                              f'{prop["solvent"][i][j] * scaling:>12.5f}   |' \
                              f'{prop["xc"][i][j] * scaling:>12.5f}   ||' \
                              f'{prop["el"][i][j] * scaling:>12.5f}   ||' \
                              f'{mo_occ[i][j]:>12.2e}   ||' \
                              f'{orbsym[i][j]:^15}\n'

                # summed contributions
                string += divider
                string += f'{"sum":^10}|' \
                          f'{np.sum(prop["coul"][i]) * scaling:>12.5f}   |' \
                          f'{np.sum(prop["exch"][i]) * scaling:>12.5f}   |' \
                          f'{np.sum(prop["kin"][i]) * scaling:>12.5f}   |' \
                          f'{np.sum(prop["nuc_att"][i]) * scaling:>12.5f}   |' \
                          f'{np.sum(prop["solvent"][i]) * scaling:>12.5f}   |' \
                          f'{np.sum(prop["xc"][i]) * scaling:>12.5f}   ||' \
                          f'{np.sum(prop["el"][i]) * scaling:>12.5f}   ||' \
                          f'{round(np.sum(mo_occ[i]) + TOLERANCE, 6):>12.2f}   ||\n'
            string += divider
            string += divider + '\n'

            # total contributions
            e_el = (np.sum(prop['el'][0]) + np.sum(prop['el'][1])) * scaling
            e_struct = prop['struct'] * scaling
            string += divider_2
            string += f'{"total sum":^{length_2-1}}|\n'
            string += divider_2
            string += f'{"electronic":^10}|{e_el:>+12.5f}   |\n'
            string += divider_2
            string += f'{"struct":^10}|{e_struct:>+12.5f}   |\n'
            string += divider_2
            string += divider_2
            string += f'{"total":^10}|{e_el + e_struct:>12.5f}   |\n'
            string += divider_2 + '\n'

#        # tensor property
#        elif prop['el'][0].ndim == prop['struct'][0].ndim == 2:
#
#            # formatting
#            length = 76
#            divider = '-' * length + '\n'
#            length_2 = 35
#            divider_2 = '-' * length_2
#
#            # units
#            if 'unit' in kwargs:
#                unit = kwargs['unit'].lower()
#            else:
#                unit = 'au'
#            assert unit in ['au', 'debye'], 'illegal unit for dipole moments. ' \
#                                            'valid options are: `au` (default) or `debye`.'
#            scaling = 1.
#            if unit == 'debye':
#                scaling = AU_TO_DEBYE
#
#            # headers
#            string += divider
#            string += f'{f"{header} (unit: {unit})":^{length}}\n'
#            string += divider
#            string += f'{"":^6}|{"electronic":^35}|{"":^15}|\n'
#            string += f'{"MO":^6}|' + divider_2 + f'|{"atom(s)":^15}|' + f'{"bond length/Ang":^17}\n'
#            string += f'{"":^6}|{"x":^11}/{"y":^11}/{"z":^11}|{"":^15}|\n'
#            string += divider
#
#            # individual contributions
#            for i, spin in enumerate(('alpha-spin', 'beta-spin')):
#                string += divider
#                string += f'{spin:^{length}}\n'
#                string += divider
#                for j in range(centres[i].shape[0]):
#                    atom = f'{mol.atom_symbol(centres[i][j, 0]):s}{centres[i][j, 0]:d}'
#                    core = centres[i][j, 0] == centres[i][j, 1]
#                    if core:
#                        string += f'  {j:>2d}  |' \
#                                  f' {prop["el"][i][j][0] * scaling + TOLERANCE:>+8.3f}  /' \
#                                  f' {prop["el"][i][j][1] * scaling + TOLERANCE:>+8.3f}  /' \
#                                  f' {prop["el"][i][j][2] * scaling + TOLERANCE:>+8.3f}  |' \
#                                  f'    {atom:<11s}|\n'
#                    else:
#                        atom += f'-{mol.atom_symbol(centres[i][j, 1]):s}{centres[i][j, 1]:d}'
#                        rr = f'{dist[centres[i][j, 0], centres[i][j, 1]]:.3f}'
#                        string += f'  {j:>2d}  |' \
#                                  f' {prop["el"][i][j][0] * scaling + TOLERANCE:>+8.3f}  /' \
#                                  f' {prop["el"][i][j][1] * scaling + TOLERANCE:>+8.3f}  /' \
#                                  f' {prop["el"][i][j][2] * scaling + TOLERANCE:>+8.3f}  |' \
#                                  f'    {atom:<11s}|' \
#                                  f'  {rr:>9s}\n'
#
#            # total contributions
#            string += divider
#            string += divider
#            sum_el = (np.fromiter(map(math.fsum, prop['el'][0].T), dtype=np.float64, count=3) + \
#                      np.fromiter(map(math.fsum, prop['el'][1].T), dtype=np.float64, count=3))
#            sum_struct = np.fromiter(map(math.fsum, prop['struct'].T), dtype=np.float64, count=3)
#            string += f'{"sum":^6}|' \
#                      f' {sum_el[0] * scaling + TOLERANCE:>+8.3f}  /' \
#                      f' {sum_el[1] * scaling + TOLERANCE:>+8.3f}  /' \
#                      f' {sum_el[2] * scaling + TOLERANCE:>+8.3f}  |\n'
#            string += divider
#            string += f'{"struct":^6}|' \
#                      f' {sum_struct[0] * scaling + TOLERANCE:>+8.3f}  /' \
#                      f' {sum_struct[1] * scaling + TOLERANCE:>+8.3f}  /' \
#                      f' {sum_struct[2] * scaling + TOLERANCE:>+8.3f}  |\n'
#            string += divider
#            string += divider
#            string += f'{"tot":^6}|' \
#                      f' {(sum_el[0] + sum_struct[0]) * scaling + TOLERANCE:>+8.3f}  /' \
#                      f' {(sum_el[1] + sum_struct[1]) * scaling + TOLERANCE:>+8.3f}  /' \
#                      f' {(sum_el[2] + sum_struct[2]) * scaling + TOLERANCE:>+8.3f}  |\n'
#            string += divider + '\n'

        return string


def _format(opt: Any) -> str:
        """
        this functions returns the correct formatting
        """
        if opt == '':
            return 'none'
        else:
            return opt

