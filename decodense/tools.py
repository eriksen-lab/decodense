#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tools module
"""

__author__ = "Janus Juul Eriksen, Technical University of Denmark, DK"
__maintainer__ = "Janus Juul Eriksen"
__email__ = "janus@kemi.dtu.dk"
__status__ = "Development"

import sys
import logging
import os
import numpy as np
from subprocess import Popen, PIPE
from pyscf import gto, scf, dft, symm
from pyscf import tools as pyscf_tools
from pyscf.pbc import gto as pbc_gto
from typing import Tuple, List, Union

try:
    import opt_einsum as oe

    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False

MAX_CYCLE = 100
NATORB_THRES = 1.0e-12


class DecodenseLogger(logging.Logger):
    def info2(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(logging.INFO - 1):
            self._log(logging.INFO - 1, msg, args, **kwargs)

    def info3(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(logging.INFO - 2):
            self._log(logging.INFO - 2, msg, args, **kwargs)


# get logger
logger = DecodenseLogger("decodense_logger")

# remove handlers from possible previous initialization
if logger.hasHandlers():
    logger.handlers.clear()

# add new handler to log to stdout
handler = logging.StreamHandler(sys.stdout)

# create new formatter
formatter = logging.Formatter("%(message)s")

# add formatter to handler
handler.setFormatter(formatter)

# add handler to logger if it does not already exist
logger.addHandler(handler)

# prevent logger from propagating handlers from parent loggers
logger.propagate = False


def logger_config(verbose: int) -> None:
    """
    this function configures the decodense logger
    """
    # corresponding logging level
    verbose_level = {0: 30, 1: 20, 2: 19, 3: 18, 4: 10, 5: 5}

    # set level for logger
    logger.setLevel(verbose_level[verbose])


def git_version() -> str:
    """
    this function returns the git revision as a string
    """

    def _minimal_ext_cmd(cmd):
        env = {}
        for k in ["SYSTEMROOT", "PATH", "HOME"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = Popen(
            cmd, stdout=PIPE, env=env, cwd=os.path.dirname(__file__)
        ).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def dim(mo_occ: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    determine molecular dimensions
    """
    return np.where(np.abs(mo_occ[0]) > 0.0)[0], np.where(np.abs(mo_occ[1]) > 0.0)[0]


def mf_info(
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT],
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    retrieve mf information (mo coefficients & occupations)
    """
    # dimensions
    alpha, beta = dim(mf.mo_occ)
    # mo occupations
    mo_occ = (np.ones_like(alpha), np.ones_like(beta))
    # mo coefficients
    if np.asarray(mf.mo_coeff).ndim == 2:
        mo_coeff = (mf.mo_coeff[:, alpha], mf.mo_coeff[:, beta])
    else:
        mo_coeff = (mf.mo_coeff[0][:, alpha], mf.mo_coeff[1][:, beta])

    return mo_coeff, mo_occ


def orbsym(mol, mo_coeff):
    """
    this functions returns orbital symmetries
    """
    if isinstance(mo_coeff, np.ndarray):
        if mo_coeff.ndim == 2:
            try:
                orbsymm = np.array(
                    symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff),
                    dtype=object,
                )
            except:
                orbsymm = np.array(["A"] * mo_coeff.shape[1], dtype=object)
        else:
            try:
                orbsymm = np.array(
                    [
                        symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c)
                        for c in mo_coeff
                    ],
                    dtype=object,
                )
            except:
                orbsymm = np.array([["A"] * c.shape[1] for c in mo_coeff], dtype=object)
    else:
        try:
            orbsymm = np.array(
                [
                    symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c)
                    for c in mo_coeff
                ],
                dtype=object,
            )
        except:
            orbsymm = np.array([["A"] * c.shape[1] for c in mo_coeff], dtype=object)

    return orbsymm


def make_rdm1(mo: np.ndarray, occup: np.ndarray) -> np.ndarray:
    """
    this function returns an 1-RDM (in ao basis) corresponding to given mo(s)
    """
    return contract("ip,jp->ij", occup * mo, mo)


def make_natorb(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mo_coeff: np.ndarray,
    rdm1: np.ndarray,
    thres: float = NATORB_THRES,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    this function returns no coefficients and occupations corresponding
    to given mo coefficients and rdm1
    """
    # reshape mo_coeff and rdm1
    if mo_coeff.ndim == 2:
        c = np.asarray((mo_coeff,) * 2)
    else:
        c = mo_coeff
    if rdm1.ndim == 2:
        d = np.array([rdm1, rdm1]) * 0.5
    else:
        d = rdm1
    # overlap matrix
    if isinstance(mol, pbc_gto.Cell):
        s = mol.pbc_intor("int1e_ovlp_sph")
    else:
        s = mol.intor_symmetric("int1e_ovlp")
    # ao to mo transformation of dm
    rdm1_mo = contract("xpi,pq,xqr,rs,xsj->xij", c, s, d, s, c)
    # diagonalize rdm1_mo
    occ_no, u = np.linalg.eigh(rdm1_mo)
    # transform to no basis
    mo_no = contract("xip,xpj->xij", c, u)
    # retain only significant nos
    return (
        mo_no[0][:, np.where(np.abs(occ_no[0]) >= thres)[0]],
        mo_no[1][:, np.where(np.abs(occ_no[1]) >= thres)[0]],
    ), (
        occ_no[0][np.where(np.abs(occ_no[0]) >= thres)],
        occ_no[1][np.where(np.abs(occ_no[1]) >= thres)],
    )


def write_rdm1(
    mol: gto.Mole,
    part: str,
    mo_coeff: Tuple[np.ndarray, np.ndarray],
    mo_occ: Tuple[np.ndarray, np.ndarray],
    fmt: str,
    writename: str,
    weights: List[np.ndarray],
) -> None:
    """
    this function writes a 1-RDM as a numpy or cube (default) file
    """
    # assertion
    assert (
        part == "atoms"
    ), "`write_rdm1` function only implemented for `atoms` partitioning"
    assert fmt in ["cube", "numpy"], "fmt arg to `write_rdm1` must be `cube` or `numpy`"
    # molecular dimensions
    alpha, beta = dim(mo_occ)
    # compute total 1-RDM (AO basis)
    rdm1_tot = np.array(
        [make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])]
    )
    # loop over atoms
    rdm1_atom_dict = {}
    for a in range(mol.natm):
        # atom-specific rdm1
        rdm1_atom = np.zeros_like(rdm1_tot)
        # loop over spins
        for i, spin_mo in enumerate((alpha, beta)):
            # loop over spin-orbitals
            for m, j in enumerate(spin_mo):
                # get orbital(s)
                orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], -1)
                # orbital-specific rdm1
                rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                # weighted contribution to rdm1_atom
                rdm1_atom[i] += rdm1_orb * weights[i][m][a]
        if fmt == "cube":
            # write rdm1_atom as cube file
            pyscf_tools.cubegen.density(
                mol,
                f"{writename}{'_' if writename else ''}atom_"
                f"{mol.atom_symbol(a).upper():s}{a:d}_rdm1.cube",
                np.sum(rdm1_atom, axis=0),
            )
        else:
            # save rdm1_atom to dictionary
            rdm1_atom_dict[f"atom_{mol.atom_symbol(a).upper():s}{a:d}_rdm1"] = rdm1_atom
    # write rdm1_atom dictionary to npz file
    if fmt == "numpy":
        if writename:
            np.savez(f"{writename}.npz", **rdm1_atom_dict)
        else:
            np.savez(f"rdm1_atom_dict.npz", **rdm1_atom_dict)


def res_add(res_a, res_b):
    """
    this function adds two result dictionaries
    """
    return {key: res_a[key] + res_b[key] for key in res_a.keys()}


def res_sub(res_a, res_b):
    """
    this function subtracts two result dictionaries
    """
    return {key: res_a[key] - res_b[key] for key in res_a.keys()}


def contract(eqn, *tensors):
    """
    interface to optimized einsum operation
    """
    if OE_AVAILABLE:
        return oe.contract(eqn, *tensors)
    else:
        return np.einsum(eqn, *tensors, optimize=True)
