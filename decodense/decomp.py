#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
decomp module
"""

__author__ = "Janus Juul Eriksen, Technical University of Denmark, DK"
__maintainer__ = "Janus Juul Eriksen"
__email__ = "janus@kemi.dtu.dk"
__status__ = "Development"

import numpy as np
from pyscf import gto, scf, dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc.lib.kpts_helper import gamma_point
from typing import List, Dict, Union, Optional, Tuple


# component keys
class CompKeys:
    coul = "Coul."
    exch = "Exch."
    kin = "Kin."
    solvent = "Solv."
    nuc_att_glob = "E_ne (1)"
    nuc_att_loc = "E_ne (2)"
    nuc_att = "E_ne"
    xc = "XC"
    struct = "Struct."
    el = "Elect."
    tot = "Total"
    charge_atom = "Charge"
    atoms = "Atom"
    orbitals = "Orbital"
    mo_occ = "Occup."
    orbsym = "Symm."


class DecompCls(object):
    """
    this class contains all decomp attributes
    """

    __slots__ = (
        "minao",
        "mo_basis",
        "pop_method",
        "mo_init",
        "loc_exp",
        "part",
        "ndo",
        "gauge_origin",
        "prop",
        "write",
        "verbose",
        "unit",
        "res",
        "charge_atom",
        "dist",
        "weights",
        "centres",
    )

    def __init__(
        self,
        minao: str = "MINAO",
        mo_basis: str = "can",
        pop_method: str = "mulliken",
        mo_init: str = "can",
        loc_exp: int = 2,
        part="atoms",
        ndo: bool = False,
        gauge_origin: np.ndarray = np.zeros(3, dtype=np.float64),
        prop: str = "energy",
        write: str = "",
        verbose: int = 0,
        unit: str = "au",
    ) -> None:
        """
        init molecule attributes
        """
        # set system defaults
        self.minao = minao
        self.mo_basis = mo_basis
        self.pop_method = pop_method
        self.mo_init = mo_init
        self.loc_exp = loc_exp
        self.part = part
        self.ndo = ndo
        self.gauge_origin = gauge_origin
        self.prop = prop
        self.write = write
        self.verbose = verbose
        self.unit = unit
        # set internal defaults
        self.res: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {}
        self.charge_atom: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.centres: Optional[np.ndarray] = None


def sanity_check(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.RHF],
    decomp: DecompCls,
    mo_coeff: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    mo_occ: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
):
    """
    this function performs sanity checks of decomp attributes
    """
    # Reference basis for IAOs
    assert decomp.minao in [
        "MINAO",
        "ANO",
    ], "invalid minao basis. valid choices: `MINAO` (default) or `ANO`"
    # MO basis
    assert decomp.mo_basis in [
        "can",
        "fb",
        "pm",
    ], "invalid MO basis. valid choices: `can` (default), `fb`, or `pm`"
    # population scheme
    assert decomp.pop_method in [
        "mulliken",
        "lowdin",
        "meta_lowdin",
        "becke",
        "iao",
    ], (
        "invalid population scheme. valid choices: `mulliken` (default), `lowdin`, "
        "`meta_lowdin`, `becke`, or `iao`"
    )
    # MO start guess (for localization)
    assert decomp.mo_init in [
        "can",
        "cholesky",
        "ibo",
    ], "invalid MO start guess. valid choices: `can` (default), `cholesky`, or `ibo`"
    # localization exponent
    assert decomp.loc_exp in [
        2,
        4,
    ], "invalid MO start guess. valid choices: 2 (default) or 4"
    # partitioning
    assert decomp.part in [
        "atoms",
        "eda",
        "orbitals",
    ], "invalid partitioning. valid choices: `atoms` (default), `eda`, or `orbitals`"
    # NDO decomposition
    assert isinstance(decomp.ndo, bool), "invalid NDO argument. must be a bool"
    # gauge origin
    assert isinstance(
        decomp.gauge_origin, (list, np.ndarray)
    ), "invalid gauge origin. must be a list or numpy array of ints/floats"
    # property
    assert decomp.prop in [
        "energy",
        "dipole",
    ], "invalid property. valid choices: `energy` (default) and `dipole`"
    # write
    assert isinstance(decomp.write, str), "invalid write format argument. must be a str"
    assert decomp.write in [
        "",
        "cube",
        "numpy",
    ], "invalid write format. valid choices: `cube` and `numpy`"
    # verbosity
    assert isinstance(
        decomp.verbose, int
    ), "invalid verbosity. valid choices: 0 <= `verbose` (default: 0)"
    assert (
        0 <= decomp.verbose
    ), "invalid verbosity. valid choices: 0 <= `verbose` (default: 0)"
    # cell object
    if isinstance(mol, pbc_gto.Cell):
        assert np.shape(mf.kpt) == (
            3,
        ), "PBC module is in development, only gamma-point methods implemented."
        assert gamma_point(
            mf.kpt
        ), "PBC module is in development, only gamma-point methods implemented."
        assert mol.dimension == 3 or mol.dimension == 1, (
            "PBC module is in development, current implementation treats 1D- and "
            "3D-cells only."
        )
        assert decomp.prop == "energy" and decomp.part in [
            "atoms",
            "eda",
        ], (
            "PBC module is in development. Only gamma-point calculation of energy for "
            "1D- and 3D-periodic systems can be decomposed into atomwise contributions."
        )
    # unit
    assert isinstance(decomp.unit, str), (
        "invalid unit. valid choices: `au` (default), `kcal_mol`, `ev`, `kj_mol`, or "
        "`debye`"
    )
    # mo coefficients
    assert isinstance(mo_coeff, np.ndarray) or isinstance(
        mo_coeff, tuple
    ), "invalid mo coefficients. must be a numpy array or tuple of numpy arrays"
    if isinstance(mo_coeff, np.ndarray):
        assert (
            mo_coeff.ndim == 2 or mo_coeff.ndim == 3
        ), "invalid mo coefficients. must be a numpy array of dimension 2 or 3"
    elif isinstance(mo_coeff, tuple):
        assert (
            len(mo_coeff) == 2
            and isinstance(mo_coeff[0], np.ndarray)
            and isinstance(mo_coeff[1], np.ndarray)
        ), "invalid mo coefficients. must be a tuple of two numpy arrays"
    # mo occupation
    assert (
        mo_occ is None or isinstance(mo_occ, np.ndarray) or isinstance(mo_occ, tuple)
    ), "invalid mo occupation. must be a numpy array or tuple of numpy arrays"
    if isinstance(mo_occ, np.ndarray):
        assert (
            mo_occ.ndim == 1 or mo_occ.ndim == 2
        ), "invalid mo occupation. must be a numpy array of dimension 1 or 2"
    elif isinstance(mo_occ, tuple):
        assert (
            len(mo_occ) == 2
            and isinstance(mo_occ[0], np.ndarray)
            and isinstance(mo_occ[1], np.ndarray)
        ), "invalid mo occupation. must be a tuple of two numpy arrays"
