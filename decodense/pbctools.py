#!/usr/bin/env python
# -*- coding: utf-8 -*

__author__ = "Luna Zamok, Technical University of Denmark, DK"
__maintainer__ = "Luna Zamok"
__email__ = "luza@kemi.dtu.dk"
__status__ = "Development"

import ctypes
import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.pbc import df as pbc_df
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import tools as pbc_tools
from pyscf.pbc.df import ft_ao, aft
from pyscf.pbc.gto import pseudo
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df.incore import Int3cBuilder
from pyscf.pbc.df.aft import _IntPPBuilder as pyscf_IntPPBuilder
from pyscf.pbc.df.rsdf_builder import (
    _RSGDFBuilder,
    estimate_rcut,
    estimate_ke_cutoff_for_omega,
    estimate_omega_for_ke_cutoff,
    estimate_ft_rcut,
)
from pyscf.pbc.df.rsdf_builder import _guess_omega, _ExtendedMoleFT
from scipy.special import erfc
from typing import List, Union, Optional, Callable

libpbc = lib.load_library("libpbc")

PRECISION = getattr(__config__, "pbc_df_aft_estimate_eta_precision", 1e-8)
KE_SCALING = getattr(__config__, "pbc_df_aft_ke_cutoff_scaling", 0.75)
RCUT_THRESHOLD = getattr(__config__, "pbc_scf_rsjk_rcut_threshold", 2.0)


def get_nuc_pbc(
    cell: pbc_gto.Cell, mydf: Union[pbc_df.df.GDF, pbc_df.fft.FFTDF]
) -> np.ndarray:
    """
    Nuc.-el.
    """

    if cell.pseudo:
        if isinstance(mydf, pbc_df.df.DF):
            vne = _get_pp_atomic_df(mydf, kpts=np.zeros(3))
        else:
            raise NotImplementedError(
                "Decodense code for %s object is not implemented yet. ", mydf
            )
    else:
        if isinstance(mydf, pbc_df.df.DF):
            vne = _get_all_e_atomic_df(mydf, kpts=np.zeros(3))
        else:
            raise NotImplementedError(
                "Decodense code for %s object is not implemented yet. ", mydf
            )
    return vne


def _get_all_e_atomic_df(
    mydf: Union[pbc_df.df.GDF, pbc_df.fft.FFTDF], kpts: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Nuc.-el. attraction for all electron calculation
    /The periodic nuc-el AO matrix, with G=0 removed.
    """
    cell = mydf.cell

    if mydf._prefer_ccdf or cell.omega > 0:
        # For long-range integrals _CCGDFBuilder is the only option
        # it is not implemented (yet)
        raise NotImplementedError("No implementation for el-nuc long-range integrals.")

    pp1builder = _RSNucBuilder(cell, kpts).build()
    vne_at = pp1builder.get_pp_loc_part1(with_pseudo=False)[0]
    return vne_at


def _get_pp_atomic_df(
    mydf: Union[pbc_df.df.GDF, pbc_df.fft.FFTDF], kpts: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Nuc.-el. attraction for calculation using pseudopotentials
    /The periodic nuc-el AO matrix, with G=0 removed.
    """

    cell = mydf.cell

    if mydf._prefer_ccdf or cell.omega > 0:
        # For long-range integrals _CCGDFBuilder is the only option
        # it is not implemented (yet)
        raise NotImplementedError("No implementation for el-nuc long-range integrals.")

    pp1builder = _RSNucBuilder(cell, kpts).build()
    vpp_loc1_at = pp1builder.get_pp_loc_part1()[0]

    pp2builder = _IntPPBuilder(cell, kpts)
    vpp_loc2_at = pp2builder.get_pp_loc_part2()[0]

    vpp_nl_at = _get_pp_nl(cell, kpts)[0]

    return vpp_loc1_at + vpp_loc2_at + vpp_nl_at


class _RSNucBuilder(_RSGDFBuilder):
    """
    Range-separated 3-center integral builder for Vnuc/Vpp part1 integrals
    (adpated from: pbc/df/rsdf_builder.py:_RSNucBuilder(_RSGDFBuilder) in PySCF)
    """

    exclude_dd_block = True
    exclude_d_aux = False

    def __init__(self, cell, kpts=np.zeros((1, 3))):
        self.mesh = None
        self.omega = None
        self.auxcell = self.rs_auxcell = None
        Int3cBuilder.__init__(self, cell, self.auxcell, kpts)

    def build(self, omega=None):
        cell = self.cell
        # fakenuc: a cell that contains the steep Gaussians to mimic nuclear density
        # used as the compensating background charges, defined by the PP parameters
        fakenuc = aft._fake_nuc(cell, with_pseudo=True)
        kpts = self.kpts
        nkpts = len(kpts)

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)

        if cell.dimension == 0:
            self.omega, self.mesh, self.ke_cutoff = _guess_omega(cell, kpts, self.mesh)
        else:
            if omega is None:
                omega = 1.0 / (1.0 + nkpts ** (1.0 / 9))
            ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)

            self.mesh = cell.cutoff_to_mesh(ke_cutoff)

            self.ke_cutoff = min(
                pbc_tools.mesh_to_cutoff(cell.lattice_vectors(), self.mesh)[
                    : cell.dimension
                ]
            )
            self.omega = estimate_omega_for_ke_cutoff(cell, self.ke_cutoff)
            if cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum":
                raise NotImplementedError(
                    "No implementation for nuc-el integrals for cell of dimension %s.",
                    cell.dimension,
                )
            elif cell.dimension < 2:
                self.mesh[cell.dimension :] = cell.mesh[cell.dimension :]
            self.mesh = cell.symmetrize_mesh(self.mesh)

        self.dump_flags()

        exp_min = np.hstack(cell.bas_exps()).min()
        # For each basis i in (ij|, small integrals accumulated by the lattice
        # sum for j are not negligible.
        lattice_sum_factor = max((2 * cell.rcut) ** 3 / cell.vol * 1 / exp_min, 1)
        cutoff = cell.precision / lattice_sum_factor * 0.1
        self.direct_scf_tol = cutoff / cell.atom_charges().max()

        # A cell with partially de-contracted basis for computing RS-integrals
        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD
        )
        rcut_sr = estimate_rcut(
            rs_cell, fakenuc, self.omega, exclude_dd_block=self.exclude_dd_block
        )
        # Extended mole object to mimic periodicity
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut_sr.max())
        supmol.omega = -self.omega
        self.supmol = supmol.strip_basis(rcut_sr)

        rcut = estimate_ft_rcut(rs_cell, exclude_dd_block=self.exclude_dd_block)
        # Extended Mole for Fourier Transform without dd-blocks
        supmol_ft = _ExtendedMoleFT.from_cell(rs_cell, kmesh, rcut.max())
        supmol_ft.exclude_dd_block = self.exclude_dd_block
        self.supmol_ft = supmol_ft.strip_basis(rcut)
        return self

    def _int_nuc_vloc(
        self,
        fakenuc: pbc_gto.Cell,
        intor: str = "int3c2e",
        aosym: str = "s2",
        comp: Optional[int] = None,
    ) -> np.ndarray:
        """
        Real-space integrals for SR-Vnuc
        """

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        nao_pair = nao * (nao + 1) // 2

        int3c = self.gen_int3c_kernel(
            intor, aosym, comp=comp, j_only=True, auxcell=fakenuc
        )
        bufR, bufI = int3c()

        charge = -cell.atom_charges()
        nchg = len(charge)
        nchg2 = 2 * nchg
        vj_at = np.einsum("kxz,z->kxz", bufR, charge)
        vj_at = np.einsum("kxz->kzx", vj_at)

        # G = 0 contributions to SR integrals
        # arxiv.org/abs/2012.07929v2 Eq. (21)
        if (
            self.omega != 0
            and (intor in ("int3c2e", "int3c2e_sph", "int3c2e_cart"))
            and (cell.dimension == 3)
        ):
            nucbar_at = np.pi / self.omega**2 / cell.vol * charge
            if self.exclude_dd_block:
                rs_cell = self.rs_cell
                ovlp = rs_cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts)
                smooth_ao_idx = rs_cell.get_ao_type() == ft_ao.SMOOTH_BASIS
                for s in ovlp:
                    s[smooth_ao_idx[:, None] & smooth_ao_idx] = 0
                recontract_2d = rs_cell.recontract(dim=2)
                ovlp = [recontract_2d(s) for s in ovlp]
            else:
                ovlp = cell.pbc_intor("int1e_ovlp", 1, lib.HERMITIAN, kpts)

            for k in range(nkpts):
                if aosym == "s1":
                    for i in range(nchg):
                        vj_at[k, i, :] -= nucbar_at[i] * ovlp[k].reshape(nao_pair)
                else:
                    for i in range(nchg):
                        vj_at[k, i, :] -= nucbar_at[i] * lib.pack_tril(ovlp[k])
        return vj_at

    def get_pp_loc_part1(self, mesh=None, with_pseudo=True):
        if self.rs_cell is None:
            self.build()
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        aosym = "s2"
        nao_pair = nao * (nao + 1) // 2
        mesh = self.mesh
        nchrg = np.size(cell.atom_charges())

        # fakenuc: a cell that contains the steep Gaussians to mimic nuclear density
        # used as the compensating background charges, defined by the PP parameters
        fakenuc = aft._fake_nuc(cell, with_pseudo=with_pseudo)
        # SR Vnuc integrals (with the compensating background charge)
        vj = self._int_nuc_vloc(fakenuc)
        if cell.dimension == 0:
            raise NotImplementedError(
                "No Vnuc/Vpp implementation for dimension %s.", cell.dimension
            )

        # If exclude_dd_block then compute the SR integrals containing compact densities
        # separately from those containing diffuse densities - i.e.
        # (DD| (integrals that contain diffuse densities, i.e. two diffuse Gaussians).
        # The former (compact) are computed in real space, and the latter (diffuse)
        # are computed in G-space togteher with LR part.
        if self.exclude_dd_block:
            # a cell with only the smooth part of the AO basis
            cell_d = self.rs_cell.smooth_basis_cell()
            if cell_d.nao > 0 and fakenuc.natm > 0:
                # For AO pair that are evaluated in blocks with using the basis
                # partitioning self.compact_basis_cell() and self.smooth_basis_cell(),
                # merge the DD block into the CC, CD, DC blocks (C ~ compact basis,
                # D ~ diffused basis)
                merge_dd_at = _merge_dd_at(self.rs_cell(), aosym)
                vj_dd_at = _int_dd_block_at(self, fakenuc)
                merge_dd_at(vj, vj_dd_at)
        else:
            print("exclude_dd_block is set to False in decodense")

        kpt_allow = np.zeros(3)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        b = cell.reciprocal_vectors()
        # Analytical FT transform AO: \int mu(r) exp(-ikr) dr^3
        # The output tensor shape: [nGv, nao]
        aoaux = ft_ao.ft_ao(fakenuc, Gv, None, b, gxyz, Gvbase)
        charges = -cell.atom_charges()

        if cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum":
            raise NotImplementedError(
                "No Vnuc/Vpp implementation for dimension %s.", cell.dimension
            )
        else:
            # The Coulomb kernel for all G-vectors, handling G=0
            coulG_LR = pbc_tools.get_coulG(
                cell, kpt_allow, mesh=mesh, Gv=Gv, omega=self.omega
            )
        # LR Coulomb in G-space
        wcoulG = coulG_LR * kws
        vG = np.einsum("i,xi,x->xi", charges, aoaux, wcoulG)

        # contributions due to pseudo.pp_int.get_gth_vlocG_part1
        # The V_GTH^local1 (G), i.e. the first term in PRB, 58, 3641 Eq (5)
        # for G=0
        if cell.dimension == 3:
            G0_idx = 0
            # exps of all basis in the cell with Gaussian charges
            exps = np.hstack(fakenuc.bas_exps())
            exps_chg = np.pi / exps * kws
            exps_chg *= charges
            # could in principle subtract average but I assume this makes most sense
            for i in range(len(exps_chg)):
                vG[G0_idx, i] -= exps_chg[i]

        # The analytical Fourier transform kernel for AO products
        # \sum_T exp(-i k_j * T) \int exp(-i(G+q)r) i(r) j(r-T) dr^3
        ft_kern = self.supmol_ft.gen_ft_kernel(aosym, return_complex=False, kpts=kpts)
        ngrids = Gv.shape[0]
        max_memory = max(2000, self.max_memory - lib.current_memory()[0])
        Gblksize = max(16, int(max_memory * 0.8e6 / 16 / (nao_pair * nkpts)) // 8 * 8)
        Gblksize = min(Gblksize, ngrids, 200000)
        vGR = vG.real
        vGI = vG.imag

        buf = np.empty((2, nkpts, Gblksize, nao_pair))
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            # shape of Gpq (nkpts, nGv, nao_pair)
            Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, out=buf)
            # rho_ij(G) nuc(-G) / G^2
            # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
                vR = np.einsum("ji,jx->ix", vGR[p0:p1], GpqR)
                vR += np.einsum("ji,jx->ix", vGI[p0:p1], GpqI)
                vj[k] += vR

        # unpacking the triangular vj matrices
        vj_kpts_at = []
        for k, kpt in enumerate(kpts):
            vj_1atm_kpts = []
            for i in range(len(charges)):
                vj_1atm_kpts.append(lib.unpack_tril(vj[k, i, :].real))
            vj_kpts_at.append(vj_1atm_kpts)
        return np.asarray(vj_kpts_at)


class _IntPPBuilder(Int3cBuilder):
    """
    3-center integral builder, for pp loc part2 only
    (adpated from: pbc/df/aft.py:_IntPPBuilder(Int3cBuilder) in PySCF)
    """

    def __init__(self, cell, kpts=np.zeros((1, 3))):
        # cache ovlp_mask which are reused for different types of intor
        self._supmol = None
        self._ovlp_mask = None
        self._cell0_ovlp_mask = None
        Int3cBuilder.__init__(self, cell, None, kpts)

    def get_ovlp_mask(self, cutoff, supmol=None, cintopt=None):
        if self._ovlp_mask is None or supmol is not self._supmol:
            self._ovlp_mask, self._cell0_ovlp_mask = Int3cBuilder.get_ovlp_mask(
                self, cutoff, supmol, cintopt
            )
            self._supmol = supmol
        return self._ovlp_mask, self._cell0_ovlp_mask

    def build(self):
        pass

    def get_pp_loc_part2(self):
        """
        Vloc pseudopotential part.
        PRB, 58, 3641 Eq (1), integrals associated to C1, C2, C3, C4
        Computed by concatenating the cell (containing basis func.), and the
        fakecells (containing, each, a coeff.*gaussian on each atom that has it).
        """

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        natm = cell.natm
        nao = cell.nao_nr()
        nao_pair = nao * (nao + 1) // 2

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD
        )

        intors = (
            "int3c2e",
            "int3c1e",
            "int3c1e_r2_origk",
            "int3c1e_r4_origk",
            "int3c1e_r6_origk",
        )
        fake_cells = {}
        fakebas_atm_ids_dict = {}
        # loop over coefficients (erf, C1, C2, C3, C4), put each
        # coeff.*gaussian in its own fakecell
        for cn in range(1, 5):
            fake_cell = pseudo.pp_int.fake_cell_vloc(cell, cn)
            if fake_cell.nbas > 0:
                # make a list on which atoms the gaussians sit on
                fakebas_atom_lst = []
                for i in range(fake_cell.nbas):
                    fakebas_atom_lst.append(fake_cell.bas_atom(i))
                fake_cells[cn] = fake_cell
                fakebas_atm_ids_dict[cn] = fakebas_atom_lst

        # if no fake_cells, check for elements in the system
        if not fake_cells:
            if any(cell.atom_symbol(ia) in cell._pseudo for ia in range(cell.natm)):
                pass
            else:
                raise ValueError(
                    "cell.pseudo was specified but its elements %s "
                    "were not found in the system (pp_part2).",
                    cell._pseudo.keys(),
                )
            vpp_loc2_at = [0] * nkpts
            return vpp_loc2_at

        rcut = pyscf_IntPPBuilder._estimate_rcut_3c1e(
            _IntPPBuilder, rs_cell, fake_cells
        )
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut.max())
        self.supmol = supmol.strip_basis(rcut)

        # buffer arrays to gather all integrals into before unpacking
        bufR_at = np.zeros((nkpts, natm, nao_pair))
        bufI_at = np.zeros((nkpts, natm, nao_pair))
        for (cn, fake_cell), (cn1, fakebas_atm_ids) in zip(
            fake_cells.items(), fakebas_atm_ids_dict.items()
        ):
            int3c = self.gen_int3c_kernel(
                intors[cn], "s2", comp=1, j_only=True, auxcell=fake_cell
            )
            # put the ints for this coeff. in the right places in the
            # buffer, i.e. assign to the right atom
            vR, vI = int3c()
            vR_at = np.einsum("kij->kji", vR)
            for k, kpt in enumerate(kpts):
                bufR_at[k, fakebas_atm_ids] += vR_at[k]
            if vI is not None:
                vI_at = np.einsum("kij->kji", vI)
                for k, kpt in enumerate(kpts):
                    bufI_at[k, fakebas_atm_ids] += vI_at[k]

        buf_at = bufR_at + bufI_at * 1j
        vpp_loc2_at = []
        # unpack vloc2 for each kpt, atom
        for k, kpt in enumerate(kpts):
            vloc2_1atm_kpts = []
            for i in range(natm):
                v_1atm_ints = lib.unpack_tril(buf_at[k, i, :])
                v_1atm_ints = v_1atm_ints.real
                vloc2_1atm_kpts.append(v_1atm_ints)
            vpp_loc2_at.append(vloc2_1atm_kpts)
        return np.asarray(vpp_loc2_at)


def _get_pp_nl(cell: pbc_gto.Cell, kpts: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Vnl pseudopotential part.
    PRB, 58, 3641 Eq (2), nonlocal contribution.
    Project the core basis funcs omitted by using pseudopotentials
    in by computing overlaps between basis funcs. (in cell) and
    projectors (gaussian, in fakecell).
    (adpated from: pbc/gto/pseudo/pp_int.py:get_pp_nl(cell, kpts=None) in PySCF)
    """

    if kpts is None:
        kpts_lst = np.zeros((1, 3))
    else:
        kpts_lst = np.reshape(kpts, (-1, 3))
    nkpts = len(kpts_lst)

    # generate a fake cell for V_{nl} gaussian functions, and
    # matrices of hl coeff. (for each atom, ang. mom.)
    fakecell, hl_blocks = pseudo.pp_int.fake_cell_vnl(cell)
    vppnl_half = pseudo.pp_int._int_vnl(cell, fakecell, hl_blocks, kpts_lst)
    nao = cell.nao_nr()
    natm = cell.natm
    buf = np.empty((3 * 9 * nao), dtype=np.complex128)

    # set equal to zeros in case hl_blocks loop is skipped
    vnl_at = np.zeros((nkpts, natm, nao, nao), dtype=np.complex128)
    for k, kpt in enumerate(kpts_lst):
        offset = [0] * 3
        # loop over bas_id, hl coeff. array
        for ib, hl in enumerate(hl_blocks):
            # the ang. mom. q.nr. associated with given basis
            l = fakecell.bas_angular(ib)
            # the id of the atom the coeff. belongs to
            atm_id_hl = fakecell.bas_atom(ib)
            nd = 2 * l + 1
            hl_dim = hl.shape[0]
            ilp: np.ndarray = np.ndarray(
                (hl_dim, nd, nao), dtype=np.complex128, buffer=buf
            )
            for i in range(hl_dim):
                # make sure that the right m,l sph.harm are taken in projectors
                p0 = offset[i]
                ilp[i] = vppnl_half[i][k][p0 : p0 + nd]
                offset[i] = p0 + nd
            vnl_at[k, atm_id_hl] += np.einsum("ilp,ij,jlq->pq", ilp.conj(), hl, ilp)

    return vnl_at.real


def _int_dd_block_at(
    dfbuilder: _RSNucBuilder, fakenuc: pbc_gto.Cell, intor: str = "int3c2e"
) -> np.ndarray:
    """
    The block of smooth AO basis in i and j of (ij|L) with full Coulomb kernel
    (adapted from: pbc/df/rsdf_builder.py:_int_dd_block() in PySCF)
    """
    if intor not in ("int3c2e", "int3c2e_sph", "int3c2e_cart"):
        raise NotImplementedError("% intor type in _int_dd_block_at", intor)

    cell = dfbuilder.cell
    cell_d = dfbuilder.rs_cell.smooth_basis_cell()
    nao = cell_d.nao
    kpts = dfbuilder.kpts
    nkpts = kpts.shape[0]
    if nao == 0 or fakenuc.natm == 0:
        return np.zeros((nao, nao, 1))

    mesh = cell_d.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    b = cell_d.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])

    kpt_allow = np.zeros(3)
    charges = -cell.atom_charges()
    #:rhoG = np.dot(charges, SI)
    aoaux = ft_ao.ft_ao(fakenuc, Gv, None, b, gxyz, Gvbase)
    rhoG = np.einsum("i,xi->xi", charges, aoaux)
    coulG = pbc_tools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
    rhoG = np.einsum("xi->ix", rhoG)
    vG = np.einsum("ix,x->ix", rhoG, coulG)

    if cell.dimension == 3:
        vG_G0 = np.zeros(len(charges))
        fakenucbas = np.pi / np.hstack(fakenuc.bas_exps())
        for z in range(len(charges)):
            vG_G0[z] = charges[z] * fakenucbas[z]
            vG[z, 0] -= vG_G0[z]

    elif cell.dimension == 2 and cell.low_dim_ft_type != "inf_vacuum":
        raise NotImplementedError(
            "No _int_dd_block_at implementation for dimension %s.", cell.dimension
        )

    vR = pbc_tools.ifft(vG, mesh).real

    coords = cell_d.get_uniform_grids(mesh)
    ao_ks = cell_d.pbc_eval_gto("GTOval", coords)
    j3c = np.zeros((len(charges), nao, nao, 1))
    for z in range(len(charges)):
        j3c[z] = lib.dot(ao_ks.T * vR[z], ao_ks).reshape(nao, nao, 1)
    return j3c


def _merge_dd_at(
    rscell: pbc_df.ft_ao._RangeSeparatedCell, aosym: str = "s1"
) -> Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """
    For AO pairs that are evaluated in blocks with using the basis
    partitioning rscell.compact_basis_cell() and rscell.smooth_basis_cell(),
    merge the DD block into the CC, CD, DC blocks (C ~ compact basis,
    D ~ diffused basis)
    (adapted from: pbc/df/ft_ao.py:_RangeSeparatedCell.merge_diffused_block() in PySCF)
    """
    libpbc = lib.load_library("libpbc")
    SMOOTH_BASIS = 2
    drv = getattr(libpbc, f"PBCnr3c_fuse_dd_{aosym}")

    ao_loc = rscell.ref_cell.ao_loc
    smooth_bas_idx = rscell.bas_map[rscell.bas_type == SMOOTH_BASIS]
    smooth_ao_idx = rscell.get_ao_indices(smooth_bas_idx, ao_loc)
    nao = ao_loc[-1]
    naod = smooth_ao_idx.size
    natm = rscell.ref_cell.natm

    def merge(
        j3c: np.ndarray, j3c_dd: np.ndarray, shls_slice: Optional[np.ndarray] = None
    ) -> np.ndarray:

        if j3c_dd.size == 0:
            return j3c
        # The AO index in the original cell
        if shls_slice is None:
            slice_in_cell = (0, nao, 0, nao)
        else:
            slice_in_cell = ao_loc[list(shls_slice[:4])]
        # Then search the corresponding index in the diffused block
        slice_in_cell_d = np.searchsorted(smooth_ao_idx, slice_in_cell)

        # j3c_dd may be an h5 object. Load j3c_dd to memory
        d0, d1 = slice_in_cell_d[:2]
        j3c_dd = np.asarray(j3c_dd[:, d0:d1], order="C")
        naux = j3c_dd.shape[-1]

        # the compact integrals for the right atoms
        for i in range(natm):
            if j3c_dd.size > 0:
                # the j3c arrays must be C-contigeous
                # i.e. either no slicing or only the left-most index
                j3c_atm = np.copy(j3c[:, i, :])
                j3c_dd_atm = np.copy(j3c_dd[i])
                drv(
                    j3c_atm.ctypes.data_as(ctypes.c_void_p),
                    j3c_dd_atm.ctypes.data_as(ctypes.c_void_p),
                    smooth_ao_idx.ctypes.data_as(ctypes.c_void_p),
                    (ctypes.c_int * 4)(*slice_in_cell),
                    (ctypes.c_int * 4)(*slice_in_cell_d),
                    ctypes.c_int(nao),
                    ctypes.c_int(naod),
                    ctypes.c_int(naux),
                )
                j3c[:, i, :] = np.copy(j3c_atm)
        return j3c

    return merge


def ewald_e_nuc(cell: pbc_gto.Cell) -> np.ndarray:
    """
    This function (PySCF 2.1) returns the nuc-nuc repulsion energy for a cell
    by performing real (R) and reciprocal (G) space Ewald sum,
    which consists of overlap, self and G-space sum
    (Formulation of Martin, App. F2.).
    (adapted from: pbc/gto/cell.py:ewald() from PySCF v2.1)
    """

    def cut_mesh_for_ewald(cell: pbc_gto.Cell, mesh_lst: List[int]) -> List[int]:
        mesh = np.array(mesh_lst, dtype=np.int64)
        mesh_max = np.asarray(
            np.linalg.norm(cell.lattice_vectors(), axis=1) * 2, dtype=int
        )  # roughly 2 grids per bohr
        if cell.dimension < 2 or (
            cell.dimension == 2 and cell.low_dim_ft_type == "inf_vacuum"
        ):
            mesh_max[cell.dimension :] = mesh[cell.dimension :]

        mesh_max[mesh_max < 80] = 80
        mesh[mesh > mesh_max] = mesh_max[mesh > mesh_max]
        return mesh.tolist()

    if cell.natm == 0:
        return np.array([], dtype=np.float64)

    ew_eta, ew_cut = cell.get_ewald_params()[0], cell.get_ewald_params()[1]
    chargs, coords = cell.atom_charges(), cell.atom_coords()

    # lattice translation vectors for nearby images (in bohr)
    Lall = cell.get_lattice_Ls(rcut=ew_cut)

    # coord. difference between atoms in the cell and its nearby images
    rLij = coords[:, None, :] - coords[None, :, :] + Lall[:, None, None, :]
    # euclidean distances
    r = np.sqrt(np.einsum("Lijx,Lijx->Lij", rLij, rLij))
    rLij = None
    # "eliminate" self-distances for R-space sum
    r[r < 1e-16] = 1e200

    # overlap (between the smeared Gaussian charge dist.) term in R-space sum
    ewovrl_atomic = 0.5 * np.einsum("i,j,Lij->i", chargs, chargs, erfc(ew_eta * r) / r)

    # self term in R-space term (last line of Eq. (F.5) in Martin)
    # cancel the self-terms in G-space sum
    ewself_factor = -0.5 * 2 * ew_eta / np.sqrt(np.pi)
    ewself_atomic = np.einsum("i,i->i", chargs, chargs)
    ewself_atomic = ewself_atomic.astype(float)
    ewself_atomic *= ewself_factor
    if cell.dimension == 3:
        ewself_atomic += (
            -0.5
            * (chargs * np.sum(chargs)).astype(float)
            * np.pi
            / (ew_eta**2 * cell.vol)
        )

    # G-space sum (corrected Eq. (F.6) in Electronic Structure by Richard M. Martin)
    # get G-grid (consisting of reciprocal lattice vectors)
    mesh = cut_mesh_for_ewald(cell, cell.mesh)
    Gv, Gvbase, Gv_weights = cell.get_Gv_weights(mesh)
    absG2 = np.einsum("gi,gi->g", Gv, Gv)
    # exclude the G=0 vector
    absG2[absG2 == 0] = 1e200

    if cell.dimension != 2 or cell.low_dim_ft_type == "inf_vacuum":
        coulG = 4 * np.pi / absG2
        coulG *= Gv_weights
        # get the structure factors
        ZSI_total = np.einsum("i,ij->j", chargs, cell.get_SI(Gv))
        ZSI_atomic = np.einsum("i,ij->ij", chargs, cell.get_SI(Gv))
        ZexpG2_atomic = ZSI_atomic * np.exp(-absG2 / (4 * ew_eta**2))
        ewg_atomic = (
            0.5 * np.einsum("j,ij,j->i", ZSI_total.conj(), ZexpG2_atomic, coulG).real
        )

    else:
        raise NotImplementedError("No Ewald sum for dimension %s.", cell.dimension)
    return ewovrl_atomic + ewself_atomic + ewg_atomic
