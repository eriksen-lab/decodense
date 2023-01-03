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
import pandas as pd
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
        string += ' MO basis           =  {:}\n'
        string += ' population scheme  =  {:}\n'
        string += ' MO start guess     =  {:}\n'
        form += (decomp.prop, decomp.part, decomp.mo_basis, decomp.pop, decomp.mo_init,)
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


def results(mol: gto.Mole, header: str = '', **kwargs: np.ndarray) -> str:
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

        # property type
        scalar_prop = kwargs['el'].ndim == 1
        # property contributions
        if scalar_prop:
            prop = {comp_key: kwargs[comp_key] for comp_key in COMP_KEYS}
            prop['tot'] = prop['el'] + prop['struct']
        else:
            prop = {comp_key + axis: kwargs[comp_key][:, ax_idx] \
                    for comp_key in COMP_KEYS[-2:] for ax_idx, axis in enumerate(('-x', '-y', '-z'))}
            for ax_idx, axis in enumerate(('-x', '-y', '-z')):
                prop['tot' + axis] = prop['el' + axis] + prop['struct' + axis]
        # effective atomic charges
        prop['charge_atom'] = kwargs['charge_atom']
        # dump dict as dataframe
        if 'dump_res' in kwargs:
            if kwargs['dump_res']:
                df = pd.DataFrame.from_dict(prop)
                suffix = '' if 'suffix' not in kwargs else kwargs['suffix']
                df.to_csv(f'res{suffix:}.csv', index=False)

        # scalar property
        if scalar_prop:

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
                          f'{prop["charge_atom"][i]:>11.3f}\n'
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
                      f'{np.sum(prop["charge_atom"]) + TOLERANCE:>11.3f}\n'
            string += divider + '\n'

        # tensor property
        else:

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
                          f' {prop["el-x"][i] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["el-y"][i] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["el-z"][i] * scaling + TOLERANCE:>8.3f}  |' \
                          f' {prop["struct-x"][i] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["struct-y"][i] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["struct-z"][i] * scaling + TOLERANCE:>8.3f}  |' \
                          f' {prop["tot-x"][i] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["tot-y"][i] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["tot-z"][i] * scaling + TOLERANCE:>8.3f}  |' \
                          f'{prop["charge_atom"][i]:>+11.3f}\n'
            string += divider
            string += divider

            # total contributions
            string += f'{"sum":^6}|' \
                      f' {np.sum(prop["el-x"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {np.sum(prop["el-y"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {np.sum(prop["el-z"]) * scaling + TOLERANCE:>8.3f}  |' \
                      f' {np.sum(prop["struct-x"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {np.sum(prop["struct-y"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {np.sum(prop["struct-z"]) * scaling + TOLERANCE:>8.3f}  |' \
                      f' {np.sum(prop["tot-x"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {np.sum(prop["tot-y"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {np.sum(prop["tot-z"]) * scaling + TOLERANCE:>8.3f}  |' \
                      f'{np.sum(prop["charge_atom"]) + TOLERANCE:>11.3f}\n'
            string += divider + '\n'

        return string


def orbs(mol: gto.Mole, header: str, **kwargs: np.ndarray) -> str:
        """
        orbital-based partitioning
        """
        # init string
        string: str = ''

        # molecular dimensions
        alpha, beta = dim(kwargs['mo_occ'])
        # mo occupations
        mo_occ = np.append(kwargs['mo_occ'][0], kwargs['mo_occ'][1])
        # orbital symmetries
        orbsym = np.append(kwargs['orbsym'][0], kwargs['orbsym'][1])
        # index
        if kwargs['ndo']:
            sort_idx = np.argsort(mo_occ)
            mo_idx = np.array([[sort_idx[i], sort_idx[-(i+1)]] for i in range(sort_idx.size // 2)]).ravel()
        else:
            mo_idx = np.arange(alpha.size + beta.size)
        # property type
        scalar_prop = kwargs['el'][0].ndim == 1
        # property contributions
        if scalar_prop:
            prop = {comp_key: np.append(kwargs[comp_key][0], kwargs[comp_key][1])[mo_idx] for comp_key in COMP_KEYS[:-1]}
            prop['struct'] = np.sum(kwargs['struct'])
        else:
            prop = {'el' + axis: np.vstack((kwargs['el'][0], kwargs['el'][1]))[mo_idx[:, None], ax_idx].ravel() \
                    for ax_idx, axis in enumerate(('-x', '-y', '-z'))}
            for ax_idx, axis in enumerate(('-x', '-y', '-z')):
                prop['struct' + axis] = np.sum(kwargs['struct'], axis=0)[ax_idx]
        # add mo occupations, orbital symmetries, and structural contributions to dict
        prop['mo_occ'] = mo_occ[mo_idx]
        prop['orbsym'] = orbsym[mo_idx]
        # dump dict as dataframe
        if 'dump_res' in kwargs:
            if kwargs['dump_res']:
                df = pd.DataFrame.from_dict(prop)
                suffix = '' if 'suffix' not in kwargs else kwargs['suffix']
                df.to_csv(f'res{suffix:}.csv', index=False)

        # scalar property
        if scalar_prop:

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
            for i in range(mo_idx.size):
                string += f'  {i:>3d}     |' \
                          f'{prop["coul"][i] * scaling:>12.5f}   |' \
                          f'{prop["exch"][i] * scaling:>12.5f}   |' \
                          f'{prop["kin"][i] * scaling:>12.5f}   |' \
                          f'{prop["nuc_att"][i] * scaling:>12.5f}   |' \
                          f'{prop["solvent"][i] * scaling:>12.5f}   |' \
                          f'{prop["xc"][i] * scaling:>12.5f}   ||' \
                          f'{prop["el"][i] * scaling:>12.5f}   ||' \
                          f'{prop["mo_occ"][i]:>12.2e}   ||' \
                          f'{prop["orbsym"][i]:^15}\n'

            # summed contributions
            string += divider
            string += f'{"sum":^10}|' \
                      f'{np.sum(prop["coul"]) * scaling:>12.5f}   |' \
                      f'{np.sum(prop["exch"]) * scaling:>12.5f}   |' \
                      f'{np.sum(prop["kin"]) * scaling:>12.5f}   |' \
                      f'{np.sum(prop["nuc_att"]) * scaling:>12.5f}   |' \
                      f'{np.sum(prop["solvent"]) * scaling:>12.5f}   |' \
                      f'{np.sum(prop["xc"]) * scaling:>12.5f}   ||' \
                      f'{np.sum(prop["el"]) * scaling:>12.5f}   ||' \
                      f'{round(np.sum(prop["mo_occ"]) + TOLERANCE, 6):>12.2f}   ||\n'
            string += divider
            string += divider + '\n'

            # total contributions
            string += divider_2
            string += f'{"total sum":^{length_2-1}}|\n'
            string += divider_2
            string += f'{"electronic":^10}|{np.sum(prop["el"]) * scaling:>12.5f}   |\n'
            string += divider_2
            string += f'{"structural":^10}|{prop["struct"] * scaling:>12.5f}   |\n'
            string += divider_2
            string += divider_2
            string += f'{"total":^10}|{(np.sum(prop["el"]) + prop["struct"]) * scaling:>12.5f}   |\n'
            string += divider_2 + '\n'

        # tensor property
        else:

            # formatting
            length = 79
            divider = '-' * length + '\n'
            length_2 = 47
            divider_2 = '-' * length_2 + '\n'

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
            string += f'{"MO":^10}|{"electronic":^35}||{"occupation":^15}||{"symmetry":^15}\n'
            string += f'{"":^10}|{"x":^11}/{"y":^11}/{"z":^11}||{"":^15}||{"":^15}\n'
            string += divider

            # individual contributions
            for i in range(mo_idx.size):
                string += f'   {i:>2d}     |' \
                          f' {prop["el-x"][i] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["el-y"][i] * scaling + TOLERANCE:>8.3f}  /' \
                          f' {prop["el-z"][i] * scaling + TOLERANCE:>8.3f}  ||' \
                          f'{prop["mo_occ"][i]:>12.2e}   ||' \
                          f'{prop["orbsym"][i]:^15}\n'

            # summed contributions
            string += divider
            string += f'{"sum":^10}|' \
                      f' {np.sum(prop["el-x"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {np.sum(prop["el-y"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {np.sum(prop["el-z"]) * scaling + TOLERANCE:>8.3f}  ||' \
                      f'{round(np.sum(prop["mo_occ"]) + TOLERANCE, 6):>12.2f}   ||\n'
            string += divider
            string += divider + '\n'

            # total contributions
            string += divider_2
            string += f'{"total sum":^{length_2-1}}|\n'
            string += divider_2
            string += f'{"electronic":^10}|' \
                      f' {np.sum(prop["el-x"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {np.sum(prop["el-y"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {np.sum(prop["el-z"]) * scaling + TOLERANCE:>8.3f}  |\n'
            string += divider_2
            string += f'{"structural":^10}|' \
                      f' {prop["struct-x"] * scaling + TOLERANCE:>8.3f}  /' \
                      f' {prop["struct-y"] * scaling + TOLERANCE:>8.3f}  /' \
                      f' {prop["struct-z"] * scaling + TOLERANCE:>8.3f}  |\n'
            string += divider_2
            string += divider_2
            string += f'{"total":^10}|' \
                      f' {(np.sum(prop["el-x"]) + prop["struct-x"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {(np.sum(prop["el-y"]) + prop["struct-y"]) * scaling + TOLERANCE:>8.3f}  /' \
                      f' {(np.sum(prop["el-z"]) + prop["struct-z"]) * scaling + TOLERANCE:>8.3f}  |\n'
            string += divider_2 + '\n'

        return string


