#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
schemes module
"""

import numpy as np

from .orbitals import assign_rdm1s
from .properties import prop_tot

def _scheme_atoms(
    mol,
    mf,
    mo_coeff,
    mo_occ,
    rdm1,
    decomp
):
    """
    This function takes care of atom-wise decompositions:
    1. Compute atomic weights
    2. Perform the decomposition
    The difference between Eriksen's partitioning scheme
    and Nakai's EDA scheme is encoded by decomp.part_method
    """
    # 1. Compute atomic weights
    weights = assign_rdm1s(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        decomp.minao,
        decomp.pop_method,
        decomp.ndo,
        decomp.verbose
    )
    # 2. Perform the decomposition
    return prop_tot(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        rdm1,
        decomp.minao,
        decomp.pop_method,
        decomp.prop,
        decomp.part_method,
        decomp.ndo,
        decomp.gauge_origin,
        weights
    )
# end _scheme_atoms

def _scheme_orbitals(
    mol,
    mf,
    mo_coeff,
    mo_occ,
    rdm1,
    decomp
):
    """
    This function takes care of orbital-wise decompositions.
    """
    return prop_tot(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        rdm1,
        decomp.minao,
        decomp.pop_method,
        decomp.prop,
        "orbitals", #TODO: this is supposed to be decomp.part_method?
        decomp.ndo,
        decomp.gauge_origin,
        weights = None #TODO: update prop_tot so it can take weights = None
    )
# end _scheme_orbitals

def _scheme_bonds_a2b(
    mol,
    mf,
    mo_coeff,
    mo_occ,
    rdm1,
    decomp
):
    """
    This function takes care of bond-wise decompositions
    using the atoms-to-bonds scheme:
    1. Perform an atom-wise decomposition
    2. Compute bond weights
    3. Perform the bond-wise decomposition
    """
    raise NotImplementedError("Bond-wise decomposition schemes are not yet implemented!")
    # 1. Perform an atom-wise decomposition
    atom_res = _scheme_atoms(
        mol,
        mf,
        mo_coeff,
        mo_occ,
        rdm1,
        decomp
    )
    # 2. Compute bond weights
    bond_weights = None #TODO: implement a function that calculates these
    # 3. Perform the bond-wise decomposition
    return atoms_to_bonds( #TODO: implement a function that redistributed atom to bond
        atom_res,
        bond_weights
    )
# end _scheme_bonds_a2b

def _scheme_bonds_aap2b(
    mol,
    mf,
    mo_coeff,
    mo_occ,
    rdm1,
    decomp
):
    """
    This function takes care of bond-wise decompositions
    using the atoms-and-atom-pairs-to-bonds scheme:
    1. Perform an atom-and-atom-pair-wise decomposition
    2. Compute bond weights
    3. Perform the bond-wise decomposition
    """
    raise NotImplementedError("Bond-wise decomposition schemes are not yet implemented!")
    # 1. Perform an atom-and-atom-pair-wise decomposition
    aap_res = None #TODO: implement
    # 2. Compute bond weights
    aap2b_weights = None #TODO: implement
    # 3. Perform the bond-wise decomposition
    return aap_to_bonds( #TODO: implement
        aap_res,
        aap2b_weights
    )
# end _scheme_bonds_aap2b

SCHEMES = {
    ("atoms", "eriksen"):  _scheme_atoms,
    ("atoms", "eda"):    _scheme_atoms,
    ("orbitals", None):  _scheme_orbitals,
    ("bonds", "a2b"):    _scheme_bonds_a2b,
    ("bonds", "aap2b"):  _scheme_bonds_aap2b
}