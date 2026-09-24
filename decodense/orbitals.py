#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
orbitals module
"""

__author__ = "Janus Juul Eriksen, Technical University of Denmark, DK"
__maintainer__ = "Janus Juul Eriksen"
__email__ = "janus@kemi.dtu.dk"
__status__ = "Development"

import numpy as np
from pyscf import gto, scf, dft, lo
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from typing import List, Union, Tuple

from .tools import dim, contract, logger


def assign_rdm1s(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.hf.RHF, pbc_dft.rks.RKS],
    mo_coeff: Tuple[np.ndarray, np.ndarray],
    mo_occ: Tuple[np.ndarray, np.ndarray],
    minao: str,
    pop_method: str,
    ndo: bool,
    verbose: int,
) -> List[np.ndarray]:
    """
    this function returns a list of population weights of each spin-orbital on the
    individual atoms
    """
    # dft logical
    dft_calc = isinstance(mf, dft.rks.KohnShamDFT)

    # rhf reference
    if mo_occ[0].size == mo_occ[1].size:
        rhf = np.allclose(mo_coeff[0], mo_coeff[1]) and np.allclose(
            mo_occ[0], mo_occ[1]
        )
    else:
        rhf = False

    if isinstance(mol, pbc_gto.Cell):
        s = mol.pbc_intor("int1e_ovlp_sph")
    else:
        s = mol.intor_symmetric("int1e_ovlp")

    # molecular dimensions
    alpha, beta = dim(mo_occ)

    # mol object projected into minao basis
    if pop_method == "iao":
        # ndo assertion
        if ndo:
            raise NotImplementedError(
                "IAO-based populations for NDOs is not implemented"
            )
        pmol = lo.iao.reference_mol(mol, minao=minao)
    else:
        pmol = mol

    # number of atoms
    natm = pmol.natm

    # AO labels
    ao_labels = pmol.ao_labels(fmt=None)

    # overlap matrix
    if pop_method == "mulliken":
        ovlp = s
    else:
        ovlp = np.eye(pmol.nao_nr())

    def get_weights(mo: np.ndarray, mocc: np.ndarray) -> np.ndarray:
        """
        this function computes the full set of population weights
        """
        if pop_method == "becke":
            # population weights of orb
            return _population_becke(charge_matrix, mo)
        else:
            # mulliken population of each orbital, per basis function
            overlap_mo = contract("ji,jp->ip", ovlp, mo)
            pop = mocc[None, :] * mo * overlap_mo
            # population weights of pop
            return _population_mul(natm, ao_labels, pop)

    # init population weights array
    weights = []

    # becke charge matrix, same for both spins
    if pop_method == "becke":
        if getattr(pmol, "pbc_intor", None):
            raise NotImplementedError("PM becke scheme for PBC systems")
        if dft_calc:
            grid_coords, grid_weights = mf.grids.get_partition(mol, concat=False)
            ni = mf._numint
        else:
            mf_becke = mol.RKS()
            grid_coords, grid_weights = mf_becke.grids.get_partition(mol, concat=False)
            ni = mf_becke._numint
        charge_matrix = np.zeros([natm, pmol.nao_nr(), pmol.nao_nr()], dtype=np.float64)
        for j in range(natm):
            ao = ni.eval_ao(mol, grid_coords[j], deriv=0)
            aow = np.einsum("pi,p->pi", ao, grid_weights[j])
            charge_matrix[j] = contract("ki,kj->ij", aow, ao)

    # loop over spin
    for i, spin_mo in enumerate((alpha, beta)):

        # get mo coefficients and occupation
        if pop_method == "mulliken":
            mo = mo_coeff[i][:, spin_mo]
        elif pop_method == "lowdin":
            mo = contract(
                "ki,kl,lj->ij",
                lo.orth.orth_ao(pmol, method="lowdin", s=s),
                s,
                mo_coeff[i][:, spin_mo],
            )
        elif pop_method == "meta_lowdin":
            mo = contract(
                "ki,kl,lj->ij",
                lo.orth.orth_ao(pmol, method="meta_lowdin", s=s),
                s,
                mo_coeff[i][:, spin_mo],
            )
        elif pop_method == "iao":
            iao = lo.iao.iao(mol, mo_coeff[i][:, spin_mo], minao=minao)
            iao = lo.vec_lowdin(iao, s)
            mo = contract("ki,kl,lj->ij", iao, s, mo_coeff[i][:, spin_mo])
        elif pop_method == "becke":
            mo = mo_coeff[i][:, spin_mo]
        else:
            raise ValueError(
                f"invalid pop_method: {pop_method}. valid choices: `mulliken`, "
                "`lowdin`, `meta_lowdin`, `iao`, or `becke`"
            )
        mocc = mo_occ[i][spin_mo]

        # get weights
        weights.append(get_weights(mo, mocc))

        # closed-shell reference
        if rhf:
            weights.append(weights[0].copy())
            break

    # verbose print
    if 0 < verbose:
        labels = [f"{pmol.atom_pure_symbol(k)}{k}" for k in range(pmol.natm)]

        # atomic populations (sum over both spins, also correct for rhf)
        total = np.sum(weights[0], axis=0) + np.sum(weights[1], axis=0)
        logger.info("\n *** atomic population ***")
        for k in range(pmol.natm):
            logger.info(f"  {labels[k]:>8s}   {total[k]:10.5f}")
        logger.info(f"  {'sum':>8s}   {np.sum(total):10.5f}")

        # full weight matrix to file (alpha only for rhf, since beta is identical)
        filename = _unique_filename(f"pop_weights_{pop_method}")
        try:
            with open(filename, "w") as f:
                f.write(
                    "# partial population weights"
                    + (" (rhf reference: beta weights identical to alpha)\n" if rhf else "\n")
                )
                f.write(f"# {'spin':>4s} {'MO':>6s} " + " ".join(f"{l:>10s}" for l in labels) + "\n")
                for i, spin_mo in enumerate((alpha, beta)):
                    for m, j in enumerate(spin_mo):
                        f.write(
                            f"  {'a' if i == 0 else 'b':>4s} {j:>6d} "
                            + " ".join(f"{w:10.5f}" for w in weights[i][m])
                            + "\n"
                        )
                    if rhf:
                        break
                # atomic population (sum over both spins)
                f.write(
                    f"  {'tot':>4s} {'-':>6s} "
                    + " ".join(f"{w:10.5f}" for w in total)
                    + "\n"
                )
            logger.info(f"\n full population weight matrix written to {filename}")
        except OSError as err:
            # a failed write of this diagnostic file should not abort the decomposition
            logger.info(
                f"\n WARNING: could not write population weight matrix to {filename} "
                f"({type(err).__name__}: {err})"
            )
    # end verbose print

    return weights

def _unique_filename(stem: str, ext: str = ".txt") -> str:
    """
    this function returns f"{stem}{ext}" if it does not exist, otherwise
    f"{stem}_{n}{ext}" with n one larger than the highest existing number.
    on any error, a warning is issued and f"{stem}{ext}" is returned
    """
    filename = f"{stem}{ext}"
    try:
        import os
        import re
        if not os.path.exists(filename):
            return filename
        directory = os.path.dirname(stem) or "."
        pattern = re.compile(
            rf"^{re.escape(os.path.basename(stem))}_(\d+){re.escape(ext)}$"
        )
        numbers = [
            int(match.group(1))
            for name in os.listdir(directory)
            if (match := pattern.match(name))
        ]
        return f"{stem}_{max(numbers, default=0) + 1}{ext}"
    except Exception as err:
        logger.info(
            f"\n WARNING: could not determine a unique filename "
            f"({type(err).__name__}: {err}); falling back to {filename}, "
            "which may overwrite an existing file"
        )
        return filename

def _population_mul(
    natm: int, ao_labels: List[Tuple[int, str, str, str]], pop: np.ndarray
) -> np.ndarray:
    """
    this function returns the mulliken populations on the individual atoms
    """
    # init populations
    populations = np.zeros((pop.shape[1], natm), dtype=np.float64)

    # loop over AOs
    for ao_pop, k in zip(pop, ao_labels):
        populations[:, k[0]] += ao_pop

    return populations


def _population_becke(charge_matrix: np.ndarray, orbs: np.ndarray) -> np.ndarray:
    """
    this function returns the becke populations on the individual atoms
    """
    # calculate populations
    populations = contract("mi,amn,ni->ia", orbs, charge_matrix, orbs)

    return populations
