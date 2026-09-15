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
from .tools import logger


# component keys
class CompKeys:
    coul = "Coul."
    exch = "Exch."
    kin = "Kin."
    solvent = "Solv."
    solvent_vdw = "Solv. (vdW)"
    nuc_att_glob = "E_ne (1)"
    nuc_att_loc = "E_ne (2)"
    nuc_att = "E_ne"
    xc = "XC"
    xc_nlc = "XC (nlc)"
    struct = "Struct."
    el = "Elect."
    tot = "Total"
    charge_atom = "Charge"
    atoms = "Atom"
    orbitals = "Orbital"
    mo_occ = "Occup."
    orbsym = "Symm."


comp_key_dict = {
    "Coul.": "coul",
    "Exch.": "exch",
    "Kin.": "kin",
    "Solv.": "solvent",
    "Solv. (vdW)": "solvent_vdw",
    "E_ne (1)": "nuc_att_glob",
    "E_ne (2)": "nuc_att_loc",
    "E_ne": "nuc_att",
    "XC": "xc",
    "XC (nlc)": "xc_nlc",
    "Struct.": "struct",
    "Elect.": "el",
    "Total": "tot",
    "Charge": "charge_atom",
    "Atom": "atoms",
    "Orbital": "orbitals",
    "Occup.": "occup",
    "Symm.": "symm",
}


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
        "writename",
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
        gauge_origin: Optional[np.ndarray] = None,
        prop: str = "energy",
        write: str = "",
        writename: str = "",
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
        self.gauge_origin = (
            np.zeros(3, dtype=np.float64) if gauge_origin is None else gauge_origin
        )
        self.prop = prop
        self.write = write
        self.writename = writename
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
    if decomp.minao not in ("MINAO", "ANO"):
        raise ValueError(
            "invalid minao basis. valid choices: `MINAO` (default) or `ANO`"
        )
    # MO basis
    if decomp.mo_basis not in ("can", "fb", "pm"):
        raise ValueError(
            "invalid MO basis. valid choices: `can` (default), `fb`, or `pm`"
        )
    # population scheme
    if decomp.pop_method not in ("mulliken", "lowdin", "meta_lowdin", "becke", "iao"):
        raise ValueError(
            "invalid population scheme. valid choices: `mulliken` (default), `lowdin`, "
            "`meta_lowdin`, `becke`, or `iao`"
        )
    # MO start guess (for localization)
    if decomp.mo_init not in ("can", "cholesky", "ibo"):
        raise ValueError(
            "invalid MO start guess. valid choices: `can` (default), `cholesky`, or "
            "`ibo`"
        )
    # localization exponent
    if decomp.loc_exp not in (2, 4):
        raise ValueError(
            "invalid localization exponent. valid choices: 2 (default) or 4"
        )
    # partitioning
    if decomp.part not in ("atoms", "eda", "orbitals"):
        raise ValueError(
            "invalid partitioning. valid choices: `atoms` (default), `eda`, or "
            "`orbitals`"
        )
    if decomp.part == "orbitals":
        logger.warning(
            "Warning: This partitioning only computes electronic energy and does not "
            "include solvent van der Waals contributions."
        )
    # NDO decomposition
    if not isinstance(decomp.ndo, bool):
        raise TypeError("invalid NDO argument. must be a bool")
    # gauge origin
    if not isinstance(decomp.gauge_origin, (list, np.ndarray)):
        raise TypeError(
            "invalid gauge origin. must be a list or numpy array of 3 ints/floats"
        )
    if len(decomp.gauge_origin) != 3 or not all(
        isinstance(coord, (int, float, np.integer, np.floating))
        for coord in decomp.gauge_origin
    ):
        raise ValueError(
            "invalid gauge origin. must be a list or numpy array of 3 ints/floats"
        )
    # property
    if decomp.prop not in ("energy", "dipole"):
        raise ValueError(
            "invalid property. valid choices: `energy` (default) and `dipole`"
        )
    # write
    if not isinstance(decomp.write, str):
        raise TypeError("invalid write format argument. must be a str")
    if not isinstance(decomp.writename, str):
        raise TypeError("invalid write name argument. must be a str")
    if decomp.write not in ("", "cube", "numpy"):
        raise ValueError("invalid write format. valid choices: `cube` and `numpy`")
    # verbosity
    if not isinstance(decomp.verbose, int):
        raise TypeError(
            "invalid verbosity. valid choices: 0 <= `verbose` <= 5 (default: 0)"
        )
    if decomp.verbose < 0 or decomp.verbose > 5:
        raise ValueError(
            "invalid verbosity. valid choices: 0 <= `verbose` <= 5 (default: 0)"
        )
    # cell object
    if isinstance(mol, pbc_gto.Cell):
        if np.shape(mf.kpt) != (3,):
            raise ValueError(
                "PBC module is in development, only gamma-point methods implemented."
            )
        if not gamma_point(mf.kpt):
            raise ValueError(
                "PBC module is in development, only gamma-point methods implemented."
            )
        if mol.dimension != 3 and mol.dimension != 1:
            raise ValueError(
                "PBC module is in development, current implementation treats 1D- and "
                "3D-cells only."
            )
        if decomp.prop != "energy" or decomp.part not in ("atoms", "eda"):
            raise ValueError(
                "PBC module is in development. Only gamma-point calculation of "
                "energy for 1D- and 3D-periodic systems can be decomposed into "
                "atomwise contributions."
            )
    # unit
    if not isinstance(decomp.unit, str):
        raise TypeError(
            "invalid unit. valid choices: `au` (default), `kcal_mol`, `ev`, "
            "`kj_mol`, or `debye`"
        )
    if decomp.unit.lower() not in ("au", "kcal_mol", "ev", "kj_mol", "debye"):
        raise ValueError(
            "invalid unit. valid choices: `au` (default), `kcal_mol`, `ev`, "
            "`kj_mol`, or `debye`"
        )
    # mo coefficients
    if not isinstance(mo_coeff, np.ndarray) and not isinstance(mo_coeff, tuple):
        raise TypeError(
            "invalid mo coefficients. must be a numpy array or tuple of numpy arrays"
        )
    if isinstance(mo_coeff, np.ndarray):
        if mo_coeff.ndim != 2 and mo_coeff.ndim != 3:
            raise ValueError(
                "invalid mo coefficients. must be a numpy array of dimension 2 or 3"
            )
    elif isinstance(mo_coeff, tuple):
        if (
            len(mo_coeff) != 2
            or not isinstance(mo_coeff[0], np.ndarray)
            or not isinstance(mo_coeff[1], np.ndarray)
        ):
            raise TypeError(
                "invalid mo coefficients. must be a tuple of two numpy arrays"
            )
    # mo occupation
    if (
        mo_occ is not None
        and not isinstance(mo_occ, np.ndarray)
        and not isinstance(mo_occ, tuple)
    ):
        raise TypeError(
            "invalid mo occupation. must be a numpy array or tuple of numpy arrays"
        )
    if isinstance(mo_occ, np.ndarray):
        if mo_occ.ndim != 1 and mo_occ.ndim != 2:
            raise ValueError(
                "invalid mo occupation. must be a numpy array of dimension 1 or 2"
            )
    elif isinstance(mo_occ, tuple):
        if (
            len(mo_occ) != 2
            or not isinstance(mo_occ[0], np.ndarray)
            or not isinstance(mo_occ[1], np.ndarray)
        ):
            raise TypeError(
                "invalid mo occupation. must be a tuple of two numpy arrays"
            )
