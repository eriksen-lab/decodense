#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
results module
"""

__author__ = "Janus Juul Eriksen, Technical University of Denmark, DK"
__maintainer__ = "Janus Juul Eriksen"
__email__ = "janus@kemi.dtu.dk"
__status__ = "Development"

import numpy as np
import pandas as pd
from pyscf import gto
from typing import Dict, Tuple, Any, Optional

from .decomp import CompKeys, DecompCls
from .tools import git_version, dim

TOLERANCE = 1.0e-10

# https://en.wikipedia.org/wiki/Hartree
AU_TO_KCAL_MOL = 627.5094740631
AU_TO_EV = 27.211386245988
AU_TO_KJ_MOL = 2625.4996394799
# https://calculla.com/dipole_moment_units_converter
AU_TO_DEBYE = 2.54174623


def info(decomp: DecompCls, mol: Optional[gto.Mole] = None, **kwargs: float) -> str:
    """
    this function prints basic info
    """
    # init string & form
    string: str = ""
    form: Tuple[Any, ...] = ()

    # print geometry
    if mol is not None:
        string += "\n\n   ------------------------------------\n"
        string += "{:^43}\n"
        string += "   ------------------------------------\n"
        form += ("geometry",)
        molecule = gto.tostring(mol).split("\n")
        for i in range(len(molecule)):
            atom = molecule[i].split()
            for j in range(1, 4):
                atom[j] = float(atom[j])
            string += "   {:<3s} {:>10.5f} {:>10.5f} {:>10.5f}\n"
            form += (*atom,)
        string += "   ------------------------------------\n"

    # system info
    string += "\n\n system info:\n"
    string += " ------------\n"
    string += " property           =  {:}\n"
    string += " partitioning       =  {:}\n"
    string += " MO basis           =  {:}\n"
    string += " population scheme  =  {:}\n"
    string += " MO start guess     =  {:}\n"
    form += (
        decomp.prop,
        decomp.part,
        decomp.mo_basis,
        decomp.pop_method,
        decomp.mo_init,
    )
    if mol is not None:
        string += "\n point group        =  {:}\n"
        string += " electrons          =  {:d}\n"
        string += " basis functions    =  {:d}\n"
        form += (
            mol.groupname,
            mol.nelectron,
            mol.nao_nr(),
        )
        if "ss" in kwargs:
            string += " spin: <S^2>        =  {:.3f}\n"
            form += (kwargs["ss"] + 1.0e-6,)
        if "s" in kwargs:
            string += " spin: 2*S + 1      =  {:.3f}\n"
            form += (kwargs["s"] + 1.0e-6,)

    # git version
    string += "\n git version: {:}\n\n"
    form += (git_version(),)

    return string.format(*form)


def fmt(mol: gto.Mole, res: Dict[str, Any], unit: str, ndo: bool) -> pd.DataFrame:
    """
    this function prints the results based on either an atom- or bond-based partitioning
    """
    if CompKeys.charge_atom in res:
        return atoms(mol, res, unit)
    else:
        return orbs(mol, res, unit, ndo)


def atoms(mol: gto.Mole, res: Dict[str, Any], unit: str) -> pd.DataFrame:
    """
    atom-based partitioning
    """
    # property type
    scalar_prop = res[CompKeys.el].ndim == 1

    # units
    unit = unit.lower()
    scaling = 1.0
    if scalar_prop:
        if unit == "kcal_mol":
            scaling = AU_TO_KCAL_MOL
        elif unit == "ev":
            scaling = AU_TO_EV
        elif unit == "kj_mol":
            scaling = AU_TO_KJ_MOL
    else:
        if unit == "debye":
            scaling = AU_TO_DEBYE

    # property contributions
    if scalar_prop:
        prop = {
            comp_key: res[comp_key] * scaling
            for comp_key in res.keys()
            if comp_key != CompKeys.charge_atom
        }
        prop[CompKeys.tot] = prop[CompKeys.el] + prop[CompKeys.struct]
    else:
        prop = {
            comp_key + axis: res[comp_key][:, ax_idx] * scaling
            for comp_key in res.keys()
            for ax_idx, axis in enumerate((" (x)", " (y)", " (z)"))
            if comp_key != CompKeys.charge_atom
        }
        for ax_idx, axis in enumerate((" (x)", " (y)", " (z)")):
            prop[CompKeys.tot + axis] = (
                prop[CompKeys.el + axis] + prop[CompKeys.struct + axis]
            )
    # partial charges
    prop[CompKeys.charge_atom] = res[CompKeys.charge_atom]
    # atom symbols
    prop[CompKeys.atoms] = [f"{mol.atom_symbol(i)}{i}" for i in range(mol.natm)]

    # return as dataframe
    return pd.DataFrame.from_dict(prop).set_index(CompKeys.atoms)


def orbs(mol: gto.Mole, res: Dict[str, Any], unit: str, ndo: bool) -> pd.DataFrame:
    """
    orbital-based partitioning
    """
    # property type
    scalar_prop = res[CompKeys.el][0].ndim == 1

    # molecular dimensions
    alpha, beta = dim(res[CompKeys.mo_occ])
    # mo occupations
    mo_occ = np.append(res[CompKeys.mo_occ][0], res[CompKeys.mo_occ][1])
    # orbital symmetries
    orbsym = np.append(res[CompKeys.orbsym][0], res[CompKeys.orbsym][1])
    # index
    if ndo:
        sort_idx = np.argsort(mo_occ)
        mo_idx = np.array(
            [[sort_idx[i], sort_idx[-(i + 1)]] for i in range(sort_idx.size // 2)]
        ).ravel()
    else:
        mo_idx = np.arange(alpha.size + beta.size)

    # units
    unit = unit.lower()
    scaling = 1.0
    if scalar_prop:
        if unit == "kcal_mol":
            scaling = AU_TO_KCAL_MOL
        elif unit == "ev":
            scaling = AU_TO_EV
        elif unit == "kj_mol":
            scaling = AU_TO_KJ_MOL
    else:
        if unit == "debye":
            scaling = AU_TO_DEBYE

    # property contributions
    if scalar_prop:
        prop = {
            comp_key: np.append(res[comp_key][0], res[comp_key][1])[mo_idx]
            for comp_key in res.keys()
            if comp_key not in (CompKeys.struct, CompKeys.charge_atom)
        }
        prop[CompKeys.struct] = np.sum(res[CompKeys.struct]) / mo_occ.size
        prop[CompKeys.tot] = prop[CompKeys.el] + prop[CompKeys.struct]
    else:
        prop = {
            CompKeys.el
            + axis: np.vstack((res[CompKeys.el][0], res[CompKeys.el][1]))[
                mo_idx[:, None], ax_idx
            ].ravel()
            for ax_idx, axis in enumerate((" (x)", " (y)", " (z)"))
        }
        for ax_idx, axis in enumerate((" (x)", " (y)", " (z)")):
            prop[CompKeys.struct + axis] = (
                np.sum(res[CompKeys.struct], axis=0)[ax_idx] / mo_occ.size
            )
            prop[CompKeys.tot + axis] = (
                prop[CompKeys.el + axis] + prop[CompKeys.struct + axis]
            )
    # add mo occupations, orbital symmetries, and structural contributions to dict
    prop[CompKeys.mo_occ] = mo_occ[mo_idx]
    prop[CompKeys.orbsym] = orbsym[mo_idx]

    # orbital indices
    prop[CompKeys.orbitals] = [f"{i}" for i in range(mo_idx.size)]

    # return as dataframe
    return pd.DataFrame.from_dict(prop).set_index(CompKeys.orbitals)
