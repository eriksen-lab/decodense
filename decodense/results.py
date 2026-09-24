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
from typing import Any, Optional

from .decomp import comp_key_dict, CompKeys, DecompCls
from .tools import git_version, dim

# https://en.wikipedia.org/wiki/Hartree
AU_TO_KCAL_MOL = 627.5094740631
AU_TO_EV = 27.211386245988
AU_TO_KJ_MOL = 2625.4996394799
# https://calculla.com/dipole_moment_units_converter
AU_TO_DEBYE = 2.54174623


class ResultsCls:
    """
    class that holds decodense results
    """

    def __init__(self, mol: gto.Mole, decomp: DecompCls):
        self.mol = mol
        self.res_dict = decomp.res
        self.print_unit = decomp.unit
        self.ndo = decomp.ndo
        self.part = decomp.part
        for key, value in self.res_dict.items():
            setattr(self, comp_key_dict[key], value)

    def __str__(self):
        """
        build a string from a pandas dataframe built from the results
        """
        return str(self.to_dataframe())

    def to_dataframe(self) -> pd.DataFrame:
        """
        build a pandas dataframe from the results
        """
        return fmt(self.mol, self.res_dict, self.print_unit, self.ndo, self.part)


def info(decomp: DecompCls, mol: Optional[gto.Mole] = None, **kwargs: float) -> str:
    """
    this function prints basic info
    """
    # init string
    string = ""

    # print geometry
    if mol is not None:
        string += "\n\n   ------------------------------------\n"
        string += f"{'geometry':^43}\n"
        string += "   ------------------------------------\n"
        molecule = gto.tostring(mol).split("\n")
        for i in range(len(molecule)):
            atom = molecule[i].split()
            for j in range(1, 4):
                atom[j] = float(atom[j])
            string += (
                f"   {atom[0]:<3s} {atom[1]:>10.5f} {atom[2]:>10.5f} {atom[3]:>10.5f}\n"
            )
        string += "   ------------------------------------\n"

    # system info
    string += "\n\n system info:\n"
    string += " ------------\n"
    string += f" property            =  {decomp.prop}\n"
    string += f" partitioning        =  {decomp.part}\n"
    strin  += f" partitioning method =  {decomp.part_method}\n"
    string += f" MO basis            =  {decomp.mo_basis}\n"
    string += f" population scheme   =  {decomp.pop_method}\n"
    string += f" MO start guess      =  {decomp.mo_init}\n"
    if mol is not None:
        string += f"\n point group        =  {mol.groupname}\n"
        string += f" electrons          =  {mol.nelectron:d}\n"
        string += f" basis functions    =  {mol.nao_nr():d}\n"
        if "ss" in kwargs:
            string += f" spin: <S^2>        =  {kwargs['ss'] + 1.0e-6:.3f}\n"
        if "s" in kwargs:
            string += f" spin: 2*S + 1      =  {kwargs['s'] + 1.0e-6:.3f}\n"

    # git version
    string += f"\n git version: {git_version()}\n\n"

    return string


def fmt(mol: gto.Mole, res: dict[str, Any], unit: str, ndo: bool, part: str) -> pd.DataFrame:
    """
    this function prints the results based on either an atom-, orbital- or bond-based partitioning
    """
    if part == "atoms":
        return atoms(mol, res, unit)
    elif part == "orbitals":
        return orbs(mol, res, unit, ndo)
    elif part == "bonds":
        return bonds() #TODO: implement later, leave as placeholder for now
    else:
        raise ValueError(f"Invalid partitioning in results.py: {part!r}")


def _unit_scaling(scalar_prop: bool, unit: str) -> float:
    """
    this function returns the unit-conversion scaling factor
    """
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
    return scaling


def atoms(mol: gto.Mole, res: Dict[str, Any], unit: str) -> pd.DataFrame:
    """
    atom-based partitioning
    """
    # property type
    scalar_prop = res[CompKeys.el].ndim == 1

    # units
    scaling = _unit_scaling(scalar_prop, unit)

    # property contributions
    if scalar_prop:
        prop = {
            comp_key: res[comp_key] * scaling
            for comp_key in res.keys()
            if comp_key != CompKeys.charge_atom
        }
    else:
        prop = {
            comp_key + axis: res[comp_key][:, ax_idx] * scaling
            for comp_key in res.keys()
            for ax_idx, axis in enumerate((" (x)", " (y)", " (z)"))
            if comp_key != CompKeys.charge_atom
        }
    # atom symbols
    prop[CompKeys.atoms] = [f"{mol.atom_symbol(i)}{i}" for i in range(mol.natm)]

    # return as dataframe
    return pd.DataFrame.from_dict(prop).set_index(CompKeys.atoms)


def orbs(mol: gto.Mole, res: dict[str, Any], unit: str, ndo: bool) -> pd.DataFrame:
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
        # pair the most negative with the most positive occupation, and so on;
        # with an odd number of NDOs, the unpaired (middle) one is listed last
        sort_idx = np.argsort(mo_occ)
        n_pairs = sort_idx.size // 2
        pairs = np.column_stack((sort_idx[:n_pairs], sort_idx[::-1][:n_pairs])).ravel()
        mo_idx = np.concatenate(
            (pairs, sort_idx[n_pairs : sort_idx.size - n_pairs])
        ).astype(np.int64)

    else:
        mo_idx = np.arange(alpha.size + beta.size)

    # units
    scaling = _unit_scaling(scalar_prop, unit)

    # property contributions
    if scalar_prop:
        prop = {
            comp_key: np.append(res[comp_key][0], res[comp_key][1])[mo_idx] * scaling
            for comp_key in res.keys()
            if comp_key
            not in (
                CompKeys.struct,
                CompKeys.charge_atom,
                CompKeys.mo_occ,
                CompKeys.orbsym,
            )
        }
        prop[CompKeys.tot] = prop[CompKeys.el]
    else:
        prop = {
            CompKeys.el
            + axis: np.vstack((res[CompKeys.el][0], res[CompKeys.el][1]))[
                mo_idx[:, None], ax_idx
            ].ravel()
            * scaling
            for ax_idx, axis in enumerate((" (x)", " (y)", " (z)"))
        }
        for ax_idx, axis in enumerate((" (x)", " (y)", " (z)")):
            prop[CompKeys.tot + axis] = prop[CompKeys.el + axis]
    # add mo occupations, orbital symmetries, and structural contributions to dict
    prop[CompKeys.mo_occ] = mo_occ[mo_idx]
    prop[CompKeys.orbsym] = orbsym[mo_idx]

    # orbital indices
    prop[CompKeys.orbitals] = [f"{i}" for i in range(mo_idx.size)]

    # return as dataframe
    return pd.DataFrame.from_dict(prop).set_index(CompKeys.orbitals)

def bonds():
    raise NotImplementedError("Bond-wise decomposition schemes are not yet implemented!")
    return