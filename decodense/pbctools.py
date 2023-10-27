#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
PBC module
most of the code adapted from functions 
in the following PySCF modules:
pbc/df/aft.py
pbc/df/fft.py
pbc/df/ft_ao.py
pbc/df/incore.py
pbc/gto/cell.py
"""

__author__ = 'Luna Zamok, Technical University of Denmark, DK'
__maintainer__ = 'Luna Zamok'
__email__ = 'luza@kemi.dtu.dk'
__status__ = 'Development'

import copy
import ctypes
import numpy as np
from pyscf import __config__
from pyscf import gto, lib
from pyscf.pbc import tools
from pyscf.pbc import df as pbc_df  
from pyscf.pbc import gto as pbc_gto  
from pyscf.pbc import scf as pbc_scf 
from pyscf.pbc import tools as pbc_tools
from pyscf.pbc.df import ft_ao, aft
from pyscf.pbc.gto import pseudo
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df.incore import Int3cBuilder
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder, estimate_rcut, estimate_ke_cutoff_for_omega, estimate_omega_for_ke_cutoff, estimate_ft_rcut 
from pyscf.pbc.df.rsdf_builder import _guess_omega, _ExtendedMoleFT, _int_dd_block
from pyscf.pbc.df.gdf_builder import _CCGDFBuilder
#from pyscf.pbc.df.incore import _Int3cBuilder, _compensate_nuccell, _fake_nuc, _strip_basis, aux_e2
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from scipy.special import erf, erfc
from typing import List, Tuple, Dict, Union, Any

libpbc = lib.load_library('libpbc')

PRECISION = getattr(__config__, 'pbc_df_aft_estimate_eta_precision', 1e-8)
KE_SCALING = getattr(__config__, 'pbc_df_aft_ke_cutoff_scaling', 0.75)
RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 2.0)

# FIXME new pp and nuc calls
def get_nuc_atomic_df(mydf: Union[pbc_df.df.GDF, pbc_df.fft.FFTDF],  \
                      kpts: Union[List[float], np.ndarray] = None) -> np.ndarray:
    """ 
    Nucl.-el. attraction for all electron calculation
    /Get the periodic nuc-el AO matrix, with G=0 removed.
    """ 
   # if kpts is None:
   #     kpts_lst = np.zeros((1,3))
   # else:
   #     kpts_lst = np.reshape(kpts, (-1,3))
    
    vne_at = get_pp_loc_part1(mydf, kpts, with_pseudo=False)
    #print('shape vne_at', np.shape(vne_at))

    #dfbuilder = _IntNucBuilder(mydf.cell, kpts_lst)
    #vne_at = dfbuilder.get_nuc(mydf.mesh, with_pseudo=False)
    #if kpts is None or np.shape(kpts) == (3,):
    #    # if gamma point
    #    if np.allclose(kpts_lst, np.zeros((1,3))):
    #        vne_at = vne_at[0].real
    #    else:
    #        vne_at = vne_at[0]
    return vne_at


def get_pp_atomic_df(mydf: Union[pbc_df.df.GDF, pbc_df.fft.FFTDF],  \
                     kpts: Union[List[float], np.ndarray] = None) -> np.ndarray:
    """ 
    Nucl.-el. attraction for calculation using pseudopotentials
    /Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    """ 
    #if kpts is None:
    #    kpts_lst = np.zeros((1,3))
    #else:
    #    kpts_lst = np.reshape(kpts, (-1,3))
    #dfbuilder = _IntNucBuilder(mydf.cell, kpts_lst)

    kpts, is_single_kpt = _check_kpts(mydf, kpts)
    cell = mydf.cell
    print('cell.omega', cell.omega)
    #vpp_loc1_at = get_pp_loc_part1(mydf, kpts, with_pseudo=True)
    if mydf._prefer_ccdf or cell.omega > 0:
        # For long-range integrals _CCGDFBuilder is the only option
        print('CCNuc builder')
        print('CCNuc builder')
        print('CCNuc builder')
        pp1builder = _CCNucBuilder(cell, kpts).build()
    else:
        print('RSNuc builder')
        print('RSNuc builder')
        print('RSNuc builder')
        pp1builder = _RSNucBuilder(cell, kpts).build()

    #vpp_loc1_at = get_pp_part1_from_rsdfbuilder(mydf, kpts, with_pseudo=True)
    vpp_loc1_at = pp1builder.get_pp_loc_part1()
    print('shape vpp_loc1_at', np.shape(vpp_loc1_at))

    pp2builder = _IntPPBuilder(cell, kpts)
    vpp_loc2_at = pp2builder.get_pp_loc_part2()
    print('shape pyscf vpp_loc2_at', np.shape(vpp_loc2_at))
    vpp_nl_at = get_pp_nl(cell, kpts)
    print('shape pyscf vpp_nl_at', np.shape(vpp_nl_at))
    
    vpp_total = vpp_loc1_at + vpp_loc2_at + vpp_nl_at
    if is_single_kpt:   
        vpp_total = vpp_total[0]
        vpp_loc1_at = vpp_loc1_at[0]
        vpp_loc2_at = vpp_loc2_at[0]
        vpp_nl_at   = vpp_nl_at[0]
    return vpp_total, vpp_loc1_at+vpp_loc2_at, vpp_nl_at
    # returns nkpts x nao x nao
    #vnl_at = dfbuilder.get_pp_nl()
    #vpp_total = vloc1_at + vloc2_at + vnl_at

# FIXME rewrite these later
#def get_nuc_atomic_fftdf(mydf: Union[pbc_df.df.GDF, pbc_df.fft.FFTDF],  \
#                         kpts: Union[List[float], np.ndarray] = None) -> np.ndarray:
#    """ 
#    Nucl.-el. attraction for all electron calculation with FFT 
#    density fitting (not recommended)
#    """ 
#    if kpts is None:
#        kpts_lst = np.zeros((1,3))
#    else:
#        kpts_lst = np.reshape(kpts, (-1,3))
#
#    cell = mydf.cell
#    mesh = mydf.mesh
#    charge = -cell.atom_charges()
#    Gv = cell.get_Gv(mesh)
#    SI = cell.get_SI(Gv)
#    natm, ngrids = np.shape(SI)
#    nkpts = len(kpts_lst)
#    nao = cell.nao_nr()
#
#    rhoG_at = np.einsum('z,zg->zg', charge, SI)
#
#    # Coulomb kernel for all G-vectors
#    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
#    vneG_at = np.einsum('zg,g->zg', rhoG_at, coulG)
#    vneR_at = np.zeros((natm, ngrids))
#    # vne evaluated in R-space
#    for a in range(natm):
#        vneR_at[a] = tools.ifft(vneG_at[a], mesh).real
#
#    vne_at = np.zeros((nkpts, natm, nao, nao))
#    # ao values on a R-grid
#    for a in range(natm):
#        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_lst):
#            ao_ks = ao_ks_etc[0]
#            for k, ao in enumerate(ao_ks):
#                vne_at[k,a] += lib.dot(ao.T.conj()*vneR_at[a,p0:p1], ao)
#            ao = ao_ks = None
#
#    if kpts is None or np.shape(kpts) == (3,):
#        if np.allclose(kpts_lst, np.zeros((1,3))):
#            vne_at = vne_at[0].real
#        else:
#            vne_at = vne_at[0]
#    return np.asarray(vne_at)
#
#
#def get_pp_atomic_fftdf(mydf: Union[pbc_df.df.GDF, pbc_df.fft.FFTDF],  \
#                        kpts: Union[List[float], np.ndarray] = None) -> np.ndarray:
#    """ 
#    Nucl.-el. attraction for calculation using pseudopotentials, 
#    FFT density fitting
#    """ 
#    cell = mydf.cell
#    if kpts is None:
#        kpts_lst = np.zeros((1,3))
#    else:
#        kpts_lst = np.reshape(kpts, (-1,3))
#
#    nkpts = len(kpts_lst)
#    nao = cell.nao_nr()
#
#    mesh = mydf.mesh
#    SI = cell.get_SI()
#    Gv = cell.get_Gv(mesh)
#    # get local pp kernel in G-space
#    vlocG = pseudo.get_vlocG(cell, Gv)
#    natm, ngrids = np.shape(vlocG)
#    vlocG_at = -np.einsum('ij,ij->ij', SI, vlocG)
#
#    # vloc evaluated in R-space
#    vlocR_at = np.zeros((natm, ngrids))
#    for a in range(natm):
#        vlocR_at[a] = tools.ifft(vlocG_at[a], mesh).real
#
#    vloc_at = np.zeros((nkpts, natm, nao, nao), dtype=np.complex128)
#    # ao values on a R-grid
#    for a in range(natm):
#        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_lst):
#            ao_ks = ao_ks_etc[0]
#            for k, ao in enumerate(ao_ks):
#                vloc_at[k,a] += lib.dot(ao.T.conj()*vlocR_at[a, p0:p1], ao)
#            ao = ao_ks = None
#
#    # generate a fake cell for V_{nl} gaussian functions, and 
#    # matrices of hl coeff. (for each atom, ang. mom.)
#    fakemol = gto.Mole()
#    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
#    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
#    ptr = gto.PTR_ENV_START
#    fakemol._env = np.zeros(ptr+10)
#    fakemol._bas[0,gto.NPRIM_OF ] = 1
#    fakemol._bas[0,gto.NCTR_OF  ] = 1
#    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
#    fakemol._bas[0,gto.PTR_COEFF] = ptr+4
#
#    buf = np.empty((48,ngrids), dtype=np.complex128)
#    def vppnl_by_k(kpt: np.ndarray) -> np.ndarray:
#        """
#        Vnl for each kpt
#        """
#        Gk = Gv + kpt
#        G_rad = lib.norm(Gk, axis=1)
#        # analytical FT AO-pair product
#        aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (1/cell.vol)**.5
#
#        vnl_at = np.zeros((natm, nao, nao), dtype=np.complex128)
#        # check if atoms have pp
#        for ia in range(cell.natm):
#            symb = cell.atom_symbol(ia)
#            if symb not in cell._pseudo:
#                continue
#            pp = cell._pseudo[symb]
#            p1 = 0
#            # check which shells are omitted by using pp
#            for l, proj in enumerate(pp[5:]):
#                rl, nl, hl = proj
#                # if the shell is in pp, need coeff. to project
#                if nl > 0:
#                    fakemol._bas[0,gto.ANG_OF] = l
#                    fakemol._env[ptr+3] = .5*rl**2
#                    fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
#                    pYlm_part = fakemol.eval_gto('GTOval', Gk)
#
#                    # make sure that the right shells are taken in projectors
#                    p0, p1 = p1, p1+nl*(l*2+1)
#                    pYlm = np.ndarray((nl,l*2+1,ngrids), dtype=np.complex128, buffer=buf[p0:p1])
#                    for k in range(nl):
#                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
#                        pYlm[k] = pYlm_part.T * qkl
#
#            # check if there are diff. orientations of ang. mom.
#            if p1 > 0:
#                SPG_lmi = buf[:p1]
#                SPG_lmi *= SI[ia].conj()
#                SPG_lm_aoGs = lib.zdot(SPG_lmi, aokG)
#                p1 = 0
#                # loop over shells with l>0, get the coeff. and integrals
#                for l, proj in enumerate(pp[5:]):
#                    rl, nl, hl = proj
#                    if nl > 0:
#                        p0, p1 = p1, p1+nl*(l*2+1)
#                        hl = np.asarray(hl)
#                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
#                        tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
#                        # pack in correct place for each atom
#                        vnl_at[ia] += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
#        return vnl_at * (1./cell.vol)
#    
#    vpp_total = np.zeros((nkpts, natm, nao, nao), dtype=np.complex128)
#    vnl_at = np.zeros((nkpts, natm, nao, nao), dtype=np.complex128)
#    # vnl evaluated in G-space
#    for k, kpt in enumerate(kpts_lst):
#        vnl_at[k] = vppnl_by_k(kpt)
#        if gamma_point(kpt):
#            vpp_total[k] = vloc_at[k].real + vnl_at[k].real
#            vpp_total[k] = vpp_total[k].real 
#            vloc_at[k] = vloc_at[k].real 
#            vnl_at[k] = vnl_at[k].real
#        else:
#            vpp_total[k] = vloc_at[k] + vnl_at[k]
#
#    if kpts is None or np.shape(kpts) == (3,):
#        if np.allclose(kpts_lst, np.zeros((1,3))):
#            vpp_total = vpp_total[0].real
#            vloc_at = vloc_at[0].real
#            vnl_at = vnl_at[0].real
#        else:
#            vpp_total = vpp_total[0]
#            vloc_at = vloc_at[0]
#            vnl_at = vnl_at[0]
#    return vpp_total, vloc_at, vnl_at

# FIXME the new PP class
#class _IntNucBuilder(_Int3cBuilder):
#    """
#    The integral builder for E_ne term when GDF is used. 
#    """
#    def __init__(self, cell: pbc_gto.Cell, \
#                 kpts: Union[List[float], np.ndarray] = np.zeros((1,3))) -> None:
#        # cache ovlp_mask
#        self._supmol = None
#        self._ovlp_mask = None
#        self._cell0_ovlp_mask = None
#        _Int3cBuilder.__init__(self, cell, None, kpts)
#
#    def get_ovlp_mask(self, cutoff: float, supmol: pbc_df.ft_ao._ExtendedMole = None, \
#                      cintopt: Any = None) -> np.ndarray:
#        """
#        ovlp_mask can be reused for different types of intor
#        """
#        if self._ovlp_mask is None or supmol is not self._supmol:
#            self._ovlp_mask, self._cell0_ovlp_mask = \
#                    _Int3cBuilder.get_ovlp_mask(self, cutoff, supmol, cintopt)
#            self._supmol = supmol
#        return self._ovlp_mask, self._cell0_ovlp_mask
#
#    def _int_nuc_vloc(self, nuccell:  pbc_gto.Cell, intor: str = 'int3c2e', \
#                      aosym: str = 's2', comp: int = None, with_pseudo: bool = True, \
#                      supmol: pbc_df.ft_ao._ExtendedMole = None) -> np.ndarray:
#        """
#        Vnuc - Vloc in R-space
#        """
#        cell = self.cell
#        kpts = self.kpts
#        nkpts = len(kpts)
#        nao = cell.nao_nr()
#        nao_pair = nao * (nao+1) // 2
#
#        # use the 3c2e code with steep s gaussians to mimic nuclear density
#        # (nuccell is the cell for model charges)
#        fakenuc = _fake_nuc(cell, with_pseudo=with_pseudo)
#        fakenuc._atm, fakenuc._bas, fakenuc._env = \
#                gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
#                             fakenuc._atm, fakenuc._bas, fakenuc._env)
#        int3c = self.gen_int3c_kernel(intor, aosym, comp=comp, j_only=True,
#                                      auxcell=fakenuc, supmol=supmol)
#        bufR, bufI = int3c()
#
#        charge = cell.atom_charges()
#        nchg   = len(charge)
#        # charge-of-nuccell, charge-of-fakenuc
#        charge = np.append(charge, -charge) 
#        nchg2  = len(charge)
#        # sum over halves, chrg and -chrg ints 
#        if is_zero(kpts):
#            vj_at1 = np.einsum('kxz,z->kzx', bufR, charge)
#            vj_at  = vj_at1[:,nchg:,:] + vj_at1[:,:nchg,:] 
#        else:
#            vj_at1 = (np.einsum('kxz,z->kzx', bufR, charge) +
#                      np.einsum('kxz,z->kzx', bufI, charge) * 1j)
#            vj_at  = vj_at1[:,nchg:,:] + vj_at1[:,:nchg,:] 
#
#        # vbar is the interaction between the background charge
#        # and the compensating function.  0D, 1D, 2D do not have vbar.
#        if cell.dimension == 3 and intor in ('int3c2e', 'int3c2e_sph',
#                                             'int3c2e_cart'):
#            charge = -cell.atom_charges()
#
#            nucbar = np.asarray([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
#            nucbar *= np.pi/cell.vol
#
#            ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
#            for k in range(nkpts):
#                if aosym == 's1':
#                    for i in range(nchg):
#                        vj_at[k,i,:] -= nucbar[i] * ovlp[k].reshape(nao_pair) 
#                else:
#                    for i in range(nchg):
#                        vj_at[k,i,:] -= nucbar[i] * lib.pack_tril(ovlp[k])
#        return vj_at
#
#    def get_nuc(self, mesh: Union[List[int], np.ndarray] = None, \
#                with_pseudo: bool = False) -> np.ndarray:
#        """
#        Vnuc term 
#        """
#        from pyscf.pbc.df.gdf_builder import _guess_eta
#
#        cell = self.cell
#        charges = cell.atom_charges()
#        kpts = self.kpts
#        nkpts = len(kpts)
#        nao = cell.nao_nr()
#        aosym = 's2'
#        nao_pair = nao * (nao+1) // 2
#
#        kpt_allow = np.zeros(3)
#        eta, mesh, ke_cutoff = _guess_eta(cell, kpts, mesh)
#
#        # check for cell with partially de-conracted basis 
#        if self.rs_cell is None:
#            self.build()
#        # initialize an extended Mole object to mimic periodicity
#        # remote basis removed if they do not contribute to the FT of basis product
#        self.supmol = supmol = _strip_basis(self.supmol, eta)
#
#        # initialize a cell of the compensated Gaussian charges for nucleus
#        modchg_cell = _compensate_nuccell(cell, eta)
#        # R-space integrals for Vnuc - Vloc
#        vj_at = self._int_nuc_vloc(modchg_cell, with_pseudo=with_pseudo,
#                                supmol=supmol)
#
#        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
#        # Coulomb kernel for all G-vectors
#        coulG = tools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv) * kws
#        # analytical FT AO-pair product
#        aoaux = ft_ao.ft_ao(modchg_cell, Gv)
#        # G-space integrals for Vnuc - Vloc
#        vG1 = np.einsum('i,xi->xi', -charges, aoaux) 
#        vG_at = np.einsum('x,xi->xi', coulG, vG1)
#
#        # initialize an extended Mole object to mimic periodicity
#        supmol_ft = ft_ao._ExtendedMole.from_cell(self.rs_cell, self.bvk_kmesh)
#        # remote basis removed if they do not contribute to the FT of basis product
#        supmol_ft = supmol_ft.strip_basis()
#        # generate the analytical FT kernel for AO products
#        ft_kern = supmol_ft.gen_ft_kernel(aosym, return_complex=False)
#
#        Gv, Gvbase, kws = modchg_cell.get_Gv_weights(mesh)
#        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
#        ngrids = Gv.shape[0]
#        # TODO do i need this mem limit? for assigning blocks of ints size
#        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
#        Gblksize = max(16, int(max_memory*1e6/16/nao_pair/nkpts))
#        Gblksize = min(Gblksize, ngrids, 200000)
#        vG_atR = vG_at.real
#        vG_atI = vG_at.imag
#
#        buf = np.empty((2, nkpts, Gblksize, nao_pair))
#        for p0, p1 in lib.prange(0, ngrids, Gblksize):
#            # analytical FT kernel for AO-products (ao values on a G-grid)
#            Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts, out=buf)
#            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
#                # contract potential on grid points with value of the ao on that grid point 
#                vR_at  = np.einsum('ji,jx->ix', vG_atR[p0:p1], GpqR)
#                vR_at += np.einsum('ji,jx->ix', vG_atI[p0:p1], GpqI)
#                vj_at[k] += vR_at
#                if not is_zero(kpts[k]):
#                    vI_at  = np.einsum('ji,jx->ix', vG_atR[p0:p1], GpqI)
#                    vI_at += np.einsum('ji,jx->ix', vG_atI[p0:p1], GpqR)
#                    vj_at[k] += vI_at * 1j
#
#        # unpacking the triangular vj matrices
#        vj_kpts_at = []
#        for k, kpt in enumerate(kpts):
#            if is_zero(kpt):
#                vj_1atm_kpts = []
#                for i in range(len(charges)):
#                    vj_1atm_kpts.append(lib.unpack_tril(vj_at[k,i,:].real))
#                vj_kpts_at.append(vj_1atm_kpts)
#            else:
#                vj_1atm_kpts = []
#                for i in range(len(charges)):
#                    vj_1atm_kpts.append(lib.unpack_tril(vj_at[k,i,:]))
#                vj_kpts_at.append(vj_1atm_kpts)
#        return np.asarray(vj_kpts_at)
#
#    def get_pp_loc_part1(self, mesh: Union[List[int], np.ndarray] = None) -> np.ndarray:
#        return self.get_nuc(mesh, with_pseudo=True)
#
#    def get_pp_loc_part2(self) -> np.ndarray:
#        """
#        Vloc pseudopotential part.
#        PRB, 58, 3641 Eq (1), integrals associated to C1, C2, C3, C4
#        Computed by concatenating the cell (containing basis func.), and the 
#        fakecells (containing, each, a coeff.*gaussian on each atom that has it).
#        """
#        if self.rs_cell is None:
#            self.build()
#        cell = self.cell
#        # initialize an extended Mole object to mimic periodicity
#        supmol = self.supmol
#        if supmol.nbas == supmol.bas_mask.size: 
#            supmol = self.supmol.strip_basis(inplace=False)
#        kpts = self.kpts
#        nkpts = len(kpts)
#        natm = cell.natm
#        nao = cell.nao_nr()
#        nao_pair = nao * (nao+1) // 2
#
#        intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
#                  'int3c1e_r4_origk', 'int3c1e_r6_origk')
#
#        # buffer arrays to gather all integrals into before unpacking
#        bufR_at = np.zeros((nkpts, natm, nao_pair))
#        bufI_at = np.zeros((nkpts, natm, nao_pair))
#        # loop over coefficients (erf, C1, C2, C3, C4), put each 
#        # coeff.*gaussian in its own fakecell
#        for cn in range(1, 5):
#            fake_cell = pseudo.pp_int.fake_cell_vloc(cell, cn)
#            if fake_cell.nbas > 0:
#                # make a list on which atoms the gaussians sit on
#                fakebas_atom_lst = []
#                for i in range(fake_cell.nbas):
#                    fakebas_atom_lst.append(fake_cell.bas_atom(i))
#                fakebas_atom_ids = np.array(fakebas_atom_lst)
#                
#                int3c = self.gen_int3c_kernel(intors[cn], 's2', comp=1, j_only=True,
#                                              auxcell=fake_cell, supmol=supmol)
#                vR, vI = int3c()
#                # put the ints for this coeff. in the right places in the 
#                # buffer, i.e. assign to the right atom
#                vR_at = np.einsum('kij->kji', vR) 
#                for k, kpt in enumerate(kpts):
#                    bufR_at[k, fakebas_atom_lst] += vR_at[k]
#                if vI is not None:
#                    vI_at = np.einsum('kij->kji', vI) 
#                    for k, kpt in enumerate(kpts):
#                        bufI_at[k, fakebas_atom_lst] += vI_at[k]
#
#        # if buffer consists of zeros, check for elements in the system 
#        if not np.any(bufR_at) :
#            if any(cell.atom_symbol(ia) in cell._pseudo for ia in range(cell.natm)):
#                pass
#            else:
#               raise ValueError('cell.pseudo was specified but its elements %s '
#                             'were not found in the system (pp_part2).', cell._pseudo.keys())
#            vloc2_at = [0] * nkpts
#        else:
#            buf_at = (bufR_at + bufI_at * 1j)
#            vloc2_at = []
#            # unpack vloc2 for each kpt, atom
#            for k, kpt in enumerate(kpts):
#                vloc2_1atm_kpts = [] 
#                for i in range(natm): 
#                    v_1atm_ints = lib.unpack_tril(buf_at[k,i,:]) 
#                    if abs(kpt).sum() < 1e-9: 
#                         v_1atm_ints = v_1atm_ints.real 
#                    vloc2_1atm_kpts.append(v_1atm_ints) 
#                vloc2_at.append(vloc2_1atm_kpts) 
#        return np.asarray(vloc2_at)
#
#    def get_pp_nl(self) -> np.ndarray:
#        """
#        Vnl pseudopotential part.
#        PRB, 58, 3641 Eq (2), nonlocal contribution.
#        Project the core basis funcs omitted by using pseudopotentials 
#        out by computing overlaps between basis funcs. (in cell) and 
#        projectors (gaussian, in fakecell).
#        """
#        cell = self.cell
#        kpts = self.kpts
#        if kpts is None:
#            kpts_lst = np.zeros((1,3))
#        else:
#            kpts_lst = np.reshape(kpts, (-1,3))
#        nkpts = len(kpts_lst)
#
#        # generate a fake cell for V_{nl} gaussian functions, and 
#        # matrices of hl coeff. (for each atom, ang. mom.)
#        fakecell, hl_blocks = pseudo.pp_int.fake_cell_vnl(cell)
#        vppnl_half = pseudo.pp_int._int_vnl(cell, fakecell, hl_blocks, kpts_lst)
#        nao = cell.nao_nr()
#        natm = cell.natm
#        buf = np.empty((3*9*nao), dtype=np.complex128)
#
#        # set equal to zeros in case hl_blocks loop is skipped
#        vnl_at = np.zeros((nkpts,natm,nao,nao), dtype=np.complex128)
#        for k, kpt in enumerate(kpts_lst):
#            offset = [0] * 3
#            # loop over bas_id, hl coeff. array 
#            for ib, hl in enumerate(hl_blocks):
#                # the ang. mom. q.nr. associated with given basis
#                l = fakecell.bas_angular(ib)
#                # the id of the atom the coeff. belongs to
#                atm_id_hl = fakecell.bas_atom(ib)
#                nd = 2 * l + 1
#                hl_dim = hl.shape[0]
#                ilp = np.ndarray((hl_dim,nd,nao), dtype=np.complex128, buffer=buf)
#                for i in range(hl_dim):
#                    # make sure that the right m,l sph.harm are taken in projectors
#                    p0 = offset[i]
#                    ilp[i] = vppnl_half[i][k][p0:p0+nd]
#                    offset[i] = p0 + nd
#                vnl_at[k,atm_id_hl] += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
#        
#        if abs(kpts_lst).sum() < 1e-9: 
#            vnl_at = vnl_at.real
#        return vnl_at
#

# FIXME pp part1, nl calls/functions
#def _int_nuc_vloc(self, nuccell:  pbc_gto.Cell, intor: str = 'int3c2e', \
#                  aosym: str = 's2', comp: int = None, with_pseudo: bool = True, \
#                  supmol: pbc_df.ft_ao._ExtendedMole = None) -> np.ndarray:
#    """
#    Vnuc - Vloc in R-space
#    """
#    cell = self.cell
#    kpts = self.kpts
#    nkpts = len(kpts)
#    nao = cell.nao_nr()
#    nao_pair = nao * (nao+1) // 2
#
#    # use the 3c2e code with steep s gaussians to mimic nuclear density
#    # (nuccell is the cell for model charges)
#    fakenuc = _fake_nuc(cell, with_pseudo=with_pseudo)
#    fakenuc._atm, fakenuc._bas, fakenuc._env = \
#            gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
#                         fakenuc._atm, fakenuc._bas, fakenuc._env)
#    int3c = self.gen_int3c_kernel(intor, aosym, comp=comp, j_only=True,
#                                  auxcell=fakenuc, supmol=supmol)
#    bufR, bufI = int3c()
#
#    charge = cell.atom_charges()
#    nchg   = len(charge)
#    # charge-of-nuccell, charge-of-fakenuc
#    charge = np.append(charge, -charge) 
#    nchg2  = len(charge)
#    # sum over halves, chrg and -chrg ints 
#    if is_zero(kpts):
#        vj_at1 = np.einsum('kxz,z->kzx', bufR, charge)
#        vj_at  = vj_at1[:,nchg:,:] + vj_at1[:,:nchg,:] 
#    else:
#        vj_at1 = (np.einsum('kxz,z->kzx', bufR, charge) +
#                  np.einsum('kxz,z->kzx', bufI, charge) * 1j)
#        vj_at  = vj_at1[:,nchg:,:] + vj_at1[:,:nchg,:] 
#
#    # vbar is the interaction between the background charge
#    # and the compensating function.  0D, 1D, 2D do not have vbar.
#    if cell.dimension == 3 and intor in ('int3c2e', 'int3c2e_sph',
#                                         'int3c2e_cart'):
#        charge = -cell.atom_charges()
#
#        nucbar = np.asarray([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
#        nucbar *= np.pi/cell.vol
#
#        ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
#        for k in range(nkpts):
#            if aosym == 's1':
#                for i in range(nchg):
#                    vj_at[k,i,:] -= nucbar[i] * ovlp[k].reshape(nao_pair) 
#            else:
#                for i in range(nchg):
#                    vj_at[k,i,:] -= nucbar[i] * lib.pack_tril(ovlp[k])
#    return vj_at

##OK
## ngrids ~= 8*naux = prod(mesh)
#def _guess_omega(cell, kpts, mesh=None):
#    if cell.dimension == 0:
#        if mesh is None:
#            mesh = cell.mesh
#        ke_cutoff = pbc_tools.mesh_to_cutoff(cell.lattice_vectors(), mesh).min()
#        return 0, mesh, ke_cutoff
#
#    # requiring Coulomb potential < cell.precision at rcut is often not
#    # enough to truncate the interaction.
#    # omega_min = estimate_omega_min(cell, cell.precision*1e-2)
#    omega_min = OMEGA_MIN
#    ke_min = estimate_ke_cutoff_for_omega(cell, omega_min, cell.precision)
#    a = cell.lattice_vectors()
#
#    if mesh is None:
#        nkpts = len(kpts)
#        ke_cutoff = 20. * nkpts**(-1./3)
#        ke_cutoff = max(ke_cutoff, ke_min)
#        mesh = cell.cutoff_to_mesh(ke_cutoff)
#    else:
#        mesh = np.asarray(mesh)
#        mesh_min = cell.cutoff_to_mesh(ke_min)
#        if np.any(mesh[:cell.dimension] < mesh_min[:cell.dimension]):
#            logger.warn(cell, 'mesh %s is not enough to converge to the required '
#                        'integral precision %g.\nRecommended mesh is %s.',
#                        mesh, cell.precision, mesh_min)
#    ke_cutoff = min(pbc_tools.mesh_to_cutoff(a, mesh)[:cell.dimension])
#    omega = estimate_omega_for_ke_cutoff(cell, ke_cutoff, cell.precision)
#    return omega, mesh, ke_cutoff
#
##OK, from rsdf_builder
#def estimate_ke_cutoff_for_omega(cell, omega, precision=None):
#    '''Energy cutoff for AFTDF to converge attenuated Coulomb in moment space
#    '''
#    if precision is None:
#        precision = cell.precision
#    exps, cs = pbc_gto.cell._extract_pgto_params(cell, 'max')
#    ls = cell._bas[:,gto.ANG_OF]
#    cs = gto.gto_norm(ls, exps)
#    Ecut = aft._estimate_ke_cutoff(exps, ls, cs, precision, omega)
#    return Ecut.max()
#
#OMEGA_MIN = 0.08
#
##OK, from rsdf_builder
#def estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision=None):
#    '''The minimal omega in attenuated Coulombl given energy cutoff
#    '''
#    if precision is None:
#        precision = cell.precision
#    # esitimation based on \int dk 4pi/k^2 exp(-k^2/4omega) sometimes is not
#    # enough to converge the 2-electron integrals. A penalty term here is to
#    # reduce the error in integrals
#    precision *= 1e-2
#    # Consider l>0 basis here to increate Ecut for slightly better accuracy
#    lmax = np.max(cell._bas[:,gto.ANG_OF])
#    kmax = (ke_cutoff*2)**.5
#    log_rest = np.log(precision / (16*np.pi**2 * kmax**lmax))
#    omega = (-.5 * ke_cutoff / log_rest)**.5
#    return omega

#def get_pp_loc_part1(mydf, mesh: Union[List[int], np.ndarray] = None, \
def get_pp_loc_part1(mydf, kpts=None, with_pseudo: bool = True) -> np.ndarray:
    """
    Vnuc term 
    """

# TODO check this, introduce
    kpts, is_single_kpt = _check_kpts(mydf, kpts)
    cell = mydf.cell
    mydf.build()
    mesh = np.asarray(mydf.mesh)
    print('mesh from df obj', mesh)
    charges = cell.atom_charges()
    #kpts = self.kpts
    natm = cell.natm
    nkpts = len(kpts)
    nao = cell.nao_nr()
    #aosym = 's2'
    nao_pair = nao * (nao+1) // 2
    #print('nao_pair', nao_pair, nao)

    kpt_allow = np.zeros(3)
    auxcell = mydf.auxcell
    print('auxcell from mydf', auxcell)
    omega, mesh, _ = _guess_omega(auxcell, kpts, mydf.mesh)
    print('mesh1 from rsgdf_builder._guess_omega and auxcell', mesh)
    mesh = cell.symmetrize_mesh(mesh)
    print('mesh sym from rsgdf_builder._guess_omega and auxcell', mesh)

    #mesh_guess = cell.cutoff_to_mesh(ke_guess)
    # TODO see if necessary
    #if np.any(mesh < mesh_guess*KE_SCALING):
    #    # TODO make a custom warning: '(mydf, 'mesh %s is not enough for AFTDF.get_nuc function to get integral accuracy %g.\nRecommended mesh is %s.', mesh, cell.precision, mesh_guess)'
    #    raise Warning

    ## check for cell with partially de-conracted basis 
    #if self.rs_cell is None:
    #    self.build()
    ## initialize an extended Mole object to mimic periodicity
    ## remote basis removed if they do not contribute to the FT of basis product
    #self.supmol = supmol = _strip_basis(self.supmol, eta)

    ## initialize a cell of the compensated Gaussian charges for nucleus
    #modchg_cell = _compensate_nuccell(cell, eta)
    ## R-space integrals for Vnuc - Vloc
    #vj_at = self._int_nuc_vloc(modchg_cell, with_pseudo=with_pseudo,
    #                        supmol=supmol)

    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    ngrids = Gv.shape[0]
    #print('shapes Gv, Gvbase, kws', np.shape(Gv), np.shape(Gvbase), np.shape(kws))

    if with_pseudo:
        # FIXME make this atomic, just a quick test here for now
        vpplocG = pseudo.pp_int.get_gth_vlocG_part1(cell, Gv)
        #print('shape vpplocG', np.shape(vpplocG))
        #print('mesh that is made here', np.shape(mesh))
        #vpplocG = -np.einsum('ij,ij->j', cell.get_SI(Gv), vpplocG)
        vpplocG_at = -np.einsum('xi,xi->xi', cell.get_SI(Gv), vpplocG)
        #print('shape vpplocG_at', np.shape(vpplocG_at))
    else:
        fakenuc = _fake_nuc(cell, with_pseudo=with_pseudo)
        # analytical FT AO-pair product
        aoaux = ft_ao.ft_ao(fakenuc, Gv)
        charges = cell.atom_charges()
        # Coulomb kernel for all G-vectors
        coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
        # G-space integrals for Vnuc - Vloc
        vpplocG    = np.einsum('i,xi->xi', -charges, aoaux)
        #print('shape no pseudo vpplocG', np.shape(vpplocG))
        vpplocG_at = np.einsum('i,xi->xi', coulG, vpplocG)
        #print('shape no pseudo vpplocG_at', np.shape(vpplocG_at))

    vpplocG_at *= kws
    #print('shape vpplocG_at*kws', np.shape(vpplocG_at))
    vGR_at = vpplocG_at.real
    vGI_at = vpplocG_at.imag
    #print('shape vGR_at', np.shape(vGR_at))

##
#    ## Coulomb kernel for all G-vectors
#    #coulG = tools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv) * kws
#    ## analytical FT AO-pair product
#    #aoaux = ft_ao.ft_ao(modchg_cell, Gv)
#    ## G-space integrals for Vnuc - Vloc
#    #vG1 = np.einsum('i,xi->xi', -charges, aoaux) 
#    #vG_at = np.einsum('x,xi->xi', coulG, vG1)
#
#    # initialize an extended Mole object to mimic periodicity
#    supmol_ft = ft_ao._ExtendedMole.from_cell(self.rs_cell, self.bvk_kmesh)
#    # remote basis removed if they do not contribute to the FT of basis product
#    supmol_ft = supmol_ft.strip_basis()
#    # generate the analytical FT kernel for AO products
#    ft_kern = supmol_ft.gen_ft_kernel(aosym, return_complex=False)
#
#    Gv, Gvbase, kws = modchg_cell.get_Gv_weights(mesh)
#    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
#    ngrids = Gv.shape[0]
#    max_memory = max(2000, self.max_memory-lib.current_memory()[0])
#    Gblksize = max(16, int(max_memory*1e6/16/nao_pair/nkpts))
#    Gblksize = min(Gblksize, ngrids, 200000)
###

    vjR_at = np.zeros((nkpts, natm, nao_pair))
    vjI_at = np.zeros((nkpts, natm, nao_pair))
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for Gpq, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts, aosym='s2',
                                    max_memory=max_memory, return_complex=False):
        # analytical FT kernel for AO-products (ao values on a G-grid)
        # shape of Gpq (nkpts, nGv, nao_pair)
        for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
            #print(' shape of Gpq (nkpts, nGv, nao_pair), real, imag: ', np.shape(Gpq), np.shape(GpqR), np.shape(GpqI) )
            #print(' shape of vGR_at[p0:p1], GpqR: ', np.shape(vGR_at[p0:p1]), np.shape(GpqR))
            #print(' shape of vGR_at, p0,p1: ', np.shape(vGR_at), p0, p1)

            # rho_ij(G) nuc(-G) / G^2
            # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
            # contract potential on grid points with value of the ao on that grid point 
            vjR_at[k] += np.einsum('ij,jx->ix', vGR_at[:,p0:p1], GpqR)
            vjR_at[k] += np.einsum('ij,jx->ix', vGI_at[:,p0:p1], GpqI)
            if not is_zero(kpts[k]):
                vjI_at[k] += np.einsum('ji,jx->ix', vGR_at[:,p0:p1], GpqI)
                vjI_at[k] -= np.einsum('ji,jx->ix', vGI_at[:,p0:p1], GpqR)

##
    #buf = np.empty((2, nkpts, Gblksize, nao_pair))
    #for p0, p1 in lib.prange(0, ngrids, Gblksize):
    #    # analytical FT kernel for AO-products (ao values on a G-grid)
    #    Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts, out=buf)
    #    for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
    #        vR_at  = np.einsum('ji,jx->ix', vG_atR[p0:p1], GpqR)
    #        vR_at += np.einsum('ji,jx->ix', vG_atI[p0:p1], GpqI)
    #        vj_at[k] += vR_at
    #        if not is_zero(kpts[k]):
    #            vI_at  = np.einsum('ji,jx->ix', vG_atR[p0:p1], GpqI)
    #            vI_at += np.einsum('ji,jx->ix', vG_atI[p0:p1], GpqR)
    #            vj_at[k] += vI_at * 1j
##

    # unpacking the triangular vj matrices
    vj_kpts_at = []
    for k, kpt in enumerate(kpts):
        if is_zero(kpt):
            vj_1atm_kpts = []
            for i in range(len(charges)):
                vj_1atm_kpts.append(lib.unpack_tril(vjR_at[k,i,:].real))
            vj_kpts_at.append(vj_1atm_kpts)
        else:
            vj_1atm_kpts = []
            for i in range(len(charges)):
                vj_1atm_kpts.append(lib.unpack_tril(vjR_at[k,i,:] + vjI_at[k,i,:]*1j) )  
            vj_kpts_at.append(vj_1atm_kpts)
    #print('shape vj_kpts_at', np.shape(np.asarray(vj_kpts_at)))
    return np.asarray(vj_kpts_at)


class _IntPPBuilder(Int3cBuilder):
    '''3-center integral builder for pp loc part2 only
    '''
    def __init__(self, cell, kpts=np.zeros((1,3))):
        # cache ovlp_mask which are reused for different types of intor
        self._supmol = None
        self._ovlp_mask = None
        self._cell0_ovlp_mask = None
        Int3cBuilder.__init__(self, cell, None, kpts)

    def get_ovlp_mask(self, cutoff, supmol=None, cintopt=None):
        if self._ovlp_mask is None or supmol is not self._supmol:
            self._ovlp_mask, self._cell0_ovlp_mask = \
                    Int3cBuilder.get_ovlp_mask(self, cutoff, supmol, cintopt)
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
        nao_pair = nao * (nao+1) // 2
    
        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
    
        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD)
            #cell, self.ke_cutoff, RCUT_THRESHOLD, verbose=log)

        intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
                  'int3c1e_r4_origk', 'int3c1e_r6_origk')
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
                raise ValueError('cell.pseudo was specified but its elements %s '
                             'were not found in the system (pp_part2).', cell._pseudo.keys())
            vpp_loc2_at = [0] * nkpts
            return vpp_loc2_at

        rcut = self._estimate_rcut_3c1e(rs_cell, fake_cells)
        #supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut.max(), log)
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut.max())
        self.supmol = supmol.strip_basis(rcut)

        # buffer arrays to gather all integrals into before unpacking
        bufR_at = np.zeros((nkpts, natm, nao_pair))
        bufI_at = np.zeros((nkpts, natm, nao_pair))
        for (cn, fake_cell), (cn1, fakebas_atm_ids) in zip(fake_cells.items(), fakebas_atm_ids_dict.items()):
            int3c = self.gen_int3c_kernel(
                intors[cn], 's2', comp=1, j_only=True, auxcell=fake_cell)
            # put the ints for this coeff. in the right places in the 
            # buffer, i.e. assign to the right atom
            vR, vI = int3c()
            #print('np.shape(vR)', np.shape(vR))
            # TODO check if needs this
            vR_at = np.einsum('kij->kji', vR) 
            for k, kpt in enumerate(kpts):
                bufR_at[k, fakebas_atm_ids] += vR_at[k]
            if vI is not None:
                vI_at = np.einsum('kij->kji', vI) 
                for k, kpt in enumerate(kpts):
                    bufI_at[k, fakebas_atm_ids] += vI_at[k]
            #
            #bufR += np.einsum('...i->...', vR)
            #if vI is not None:
            #    bufI += np.einsum('...i->...', vI)

        buf_at = (bufR_at + bufI_at * 1j)#.reshape(nkpts,-1)
        vpp_loc2_at = []
        # unpack vloc2 for each kpt, atom
        for k, kpt in enumerate(kpts):
           vloc2_1atm_kpts = [] 
           for i in range(natm):
               v_1atm_ints = lib.unpack_tril(buf_at[k,i,:])
               if is_zero(kpt):  # gamma_point:
                    v_1atm_ints = v_1atm_ints.real
               vloc2_1atm_kpts.append(v_1atm_ints)
           vpp_loc2_at.append(vloc2_1atm_kpts)
        return np.asarray(vpp_loc2_at)

    def _estimate_rcut_3c1e(self, cell, fake_cells):
        '''Estimate rcut for pp-loc part2 based on 3-center overlap integrals.
        '''
        precision = cell.precision
        exps = np.array([e.min() for e in cell.bas_exps()])
        if exps.size == 0:
            return np.zeros(1)

        ls = cell._bas[:,gto.ANG_OF]
        cs = gto.gto_norm(ls, exps)
        ai_idx = exps.argmin()
        ai = exps[ai_idx]
        li = cell._bas[ai_idx,gto.ANG_OF]
        ci = cs[ai_idx]

        r0 = cell.rcut  # initial guess
        rcut = []
        for lk, fake_cell in fake_cells.items():
            nuc_exps = np.hstack(fake_cell.bas_exps())
            ak_idx = nuc_exps.argmin()
            ak = nuc_exps[ak_idx]
            ck = abs(fake_cell._env[fake_cell._bas[ak_idx,gto.PTR_COEFF]])

            aij = ai + exps
            ajk = exps + ak
            aijk = aij + ak
            aijk1 = aijk**-.5
            theta = 1./(1./aij + 1./ak)
            norm_ang = ((2*li+1)*(2*ls+1))**.5/(4*np.pi)
            c1 = ci * cs * ck * norm_ang
            sfac = aij*exps/(aij*exps + ai*theta)
            rfac = ak / (aij * ajk)
            fl = 2
            fac = 2**(li+1)*np.pi**2.5 * aijk1**3 * c1 / theta * fl / precision

            r0 = (np.log(fac * r0 * (rfac*exps*r0+aijk1)**li *
                         (rfac*ai*r0+aijk1)**ls + 1.) / (sfac*theta))**.5
            r0 = (np.log(fac * r0 * (rfac*exps*r0+aijk1)**li *
                         (rfac*ai*r0+aijk1)**ls + 1.) / (sfac*theta))**.5
            rcut.append(r0)
        return np.max(rcut, axis=0)

def get_pp_nl(cell, kpts=None):
    """
    Vnl pseudopotential part.
    PRB, 58, 3641 Eq (2), nonlocal contribution.
    Project the core basis funcs omitted by using pseudopotentials 
    out by computing overlaps between basis funcs. (in cell) and 
    projectors (gaussian, in fakecell).
    """
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    # generate a fake cell for V_{nl} gaussian functions, and 
    # matrices of hl coeff. (for each atom, ang. mom.)
    fakecell, hl_blocks = pseudo.pp_int.fake_cell_vnl(cell)
    vppnl_half = pseudo.pp_int._int_vnl(cell, fakecell, hl_blocks, kpts_lst)
    nao = cell.nao_nr()
    natm = cell.natm
    buf = np.empty((3*9*nao), dtype=np.complex128)

    # set equal to zeros in case hl_blocks loop is skipped
    vnl_at = np.zeros((nkpts,natm,nao,nao), dtype=np.complex128)
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
            ilp = np.ndarray((hl_dim,nd,nao), dtype=np.complex128, buffer=buf)
            for i in range(hl_dim):
                # make sure that the right m,l sph.harm are taken in projectors
                p0 = offset[i]
                ilp[i] = vppnl_half[i][k][p0:p0+nd]
                offset[i] = p0 + nd
            vnl_at[k,atm_id_hl] += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
    
    if abs(kpts_lst).sum() < 1e-9: 
        vnl_at = vnl_at.real
    return vnl_at
#



# FIXME double check this against a new version
def ewald_e_nuc(cell: pbc_gto.Cell) -> np.ndarray:
    """
    This function returns the nuc-nuc repulsion energy for a cell
    by performing real (R) and reciprocal (G) space Ewald sum, 
    which consists of overlap, self and G-space sum 
    (Formulation of Martin, App. F2.).
    """ 
    def cut_mesh_for_ewald(cell: pbc_gto.Cell, mesh: List[int]) -> List[int]:
        mesh = np.copy(mesh)
        mesh_max = np.asarray(np.linalg.norm(cell.lattice_vectors(), axis=1) * 2,
                              dtype=int)  # roughly 2 grids per bohr
        if (cell.dimension < 2 or
            (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
            mesh_max[cell.dimension:] = mesh[cell.dimension:]

        mesh_max[mesh_max<80] = 80
        mesh[mesh>mesh_max] = mesh_max[mesh>mesh_max]
        return mesh

    if cell.natm == 0:
        return 0

    ew_eta, ew_cut = cell.get_ewald_params()[0], cell.get_ewald_params()[1]
    chargs, coords = cell.atom_charges(), cell.atom_coords()

    # lattice translation vectors for nearby images (in bohr)
    Lall = cell.get_lattice_Ls(rcut=ew_cut)

    # coord. difference between atoms in the cell and its nearby images
    rLij = coords[:,None,:] - coords[None,:,:] + Lall[:,None,None,:]
    # euclidean distances 
    r = np.sqrt(np.einsum('Lijx,Lijx->Lij', rLij, rLij))
    rLij = None
    # "eliminate" self-distances 
    r[r<1e-16] = 1e200
    
    # overlap term in R-space sum 
    ewovrl_atomic = .5 * np.einsum('i,j,Lij->i', chargs, chargs, erfc(ew_eta * r) / r)
    
    # self term in R-space term (last line of Eq. (F.5) in Martin)
    ewself_factor = -.5 * 2 * ew_eta / np.sqrt(np.pi)
    ewself_atomic = np.einsum('i,i->i', chargs,chargs)
    ewself_atomic = ewself_atomic.astype(float)
    ewself_atomic *= ewself_factor 
    if cell.dimension == 3:
        ewself_atomic += -.5 * (chargs*np.sum(chargs)).astype(float) * np.pi/(ew_eta**2 * cell.vol)

    # G-space sum (corrected Eq. (F.6) in Electronic Structure by Richard M. Martin)
    # get G-grid (consisting of reciprocal lattice vectors)
    mesh = cut_mesh_for_ewald(cell, cell.mesh)
    Gv, Gvbase, Gv_weights = cell.get_Gv_weights(mesh)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    # exclude the G=0 vector
    absG2[absG2==0] = 1e200

    if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
        coulG = 4*np.pi / absG2
        coulG *= Gv_weights
        # get the structure factors
        ZSI_total = np.einsum("i,ij->j", chargs, cell.get_SI(Gv))
        ZSI_atomic = np.einsum("i,ij->ij", chargs, cell.get_SI(Gv)) 
        ZexpG2_atomic = ZSI_atomic * np.exp(-absG2/(4*ew_eta**2))
        ewg_atomic = .5 * np.einsum('j,ij,j->i', ZSI_total.conj(), ZexpG2_atomic, coulG).real

    else:
        raise NotImplementedError('No Ewald sum for dimension %s.', cell.dimension)
    
    return ewovrl_atomic + ewself_atomic + ewg_atomic


def _check_kpts(mydf, kpts):
    '''Check if the argument kpts is a single k-point'''
    if kpts is None:
        kpts = np.asarray(mydf.kpts)
        # mydf.kpts is initialized to np.zeros((1,3)). Here is only a guess
        # based on the value of mydf.kpts.
        is_single_kpt = kpts.ndim == 1 or is_zero(kpts)
    else:
        kpts = np.asarray(kpts)
        is_single_kpt = kpts.ndim == 1
    kpts = kpts.reshape(-1,3)
    return kpts, is_single_kpt

def _estimate_ke_cutoff(alpha, l, c, precision, omega=0):
    '''Energy cutoff estimation for 4-center Coulomb repulsion integrals'''
    norm_ang = ((2*l+1)/(4*np.pi))**2
    fac = 8*np.pi**5 * c**4*norm_ang / (2*alpha)**(4*l+2) / precision
    Ecut = 20.
    if omega <= 0:
        Ecut = np.log(fac * (Ecut*.5)**(2*l-.5) + 1.) * 2*alpha
        Ecut = np.log(fac * (Ecut*.5)**(2*l-.5) + 1.) * 2*alpha
    else:
        theta = 1./(1./(2*alpha) + 1./(2*omega**2))
        Ecut = np.log(fac * (Ecut*.5)**(2*l-.5) + 1.) * theta
        Ecut = np.log(fac * (Ecut*.5)**(2*l-.5) + 1.) * theta
    return Ecut

def estimate_ke_cutoff(cell, precision=None):
    '''Energy cutoff estimation for 4-center Coulomb repulsion integrals'''
    if cell.nbas == 0:
        return 0.
    if precision is None:
        precision = cell.precision
    exps, cs = pbc_gto.cell._extract_pgto_params(cell, 'max')
    ls = cell._bas[:,gto.ANG_OF]
    cs = gto.gto_norm(ls, exps)
    Ecut = _estimate_ke_cutoff(exps, ls, cs, precision)
    return Ecut.max()
######################################
######################################
######################################
class _RSNucBuilder(_RSGDFBuilder):

    # TODO after fixing merge_dd, change back to True
    exclude_dd_block = False
    exclude_d_aux = False

    def __init__(self, cell, kpts=np.zeros((1,3))):
        self.mesh = None
        self.omega = None
        self.auxcell = self.rs_auxcell = None
        Int3cBuilder.__init__(self, cell, self.auxcell, kpts)

    def build(self, omega=None):
        cell = self.cell
        fakenuc = aft._fake_nuc(cell, with_pseudo=True)
        kpts = self.kpts
        nkpts = len(kpts)

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)

        if cell.dimension == 0:
            self.omega, self.mesh, self.ke_cutoff = _guess_omega(cell, kpts, self.mesh)
        else:
            if omega is None:
                omega = 1./(1.+nkpts**(1./9))
            ke_cutoff = estimate_ke_cutoff_for_omega(cell, omega)
            self.mesh = cell.cutoff_to_mesh(ke_cutoff)
            self.ke_cutoff = min(pbc_tools.mesh_to_cutoff(
                cell.lattice_vectors(), self.mesh)[:cell.dimension])
            self.omega = estimate_omega_for_ke_cutoff(cell, self.ke_cutoff)
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                #self.mesh[2] = _estimate_meshz(cell)
                raise NotImplementedError('No implemenmtation for el-nuc integrals for cell of dimension %s.', cell.dimension)
            elif cell.dimension < 2:
                self.mesh[cell.dimension:] = cell.mesh[cell.dimension:]
            self.mesh = cell.symmetrize_mesh(self.mesh)

        self.dump_flags()

        exp_min = np.hstack(cell.bas_exps()).min()
        # For each basis i in (ij|, small integrals accumulated by the lattice
        # sum for j are not negligible.
        lattice_sum_factor = max((2*cell.rcut)**3/cell.vol * 1/exp_min, 1)
        cutoff = cell.precision / lattice_sum_factor * .1
        self.direct_scf_tol = cutoff / cell.atom_charges().max()
        #log.debug('Set _RSNucBuilder.direct_scf_tol to %g', cutoff)

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD)
        rcut_sr = estimate_rcut(rs_cell, fakenuc, self.omega,
                                rs_cell.precision, self.exclude_dd_block)
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut_sr.max())
        supmol.omega = -self.omega
        self.supmol = supmol.strip_basis(rcut_sr)
        #log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  #supmol.nbas, supmol.nao, supmol.npgto_nr())

        rcut = estimate_ft_rcut(rs_cell, cell.precision, self.exclude_dd_block)
        supmol_ft = _ExtendedMoleFT.from_cell(rs_cell, kmesh, rcut.max())
        supmol_ft.exclude_dd_block = self.exclude_dd_block
        self.supmol_ft = supmol_ft.strip_basis(rcut)
        #log.debug('sup-mol-ft nbas = %d cGTO = %d pGTO = %d',
        #          supmol_ft.nbas, supmol_ft.nao, supmol_ft.npgto_nr())
        #log.timer_debug1('initializing supmol', *cpu0)
        return self

    def _int_nuc_vloc(self, fakenuc:  pbc_gto.Cell, intor: str = 'int3c2e', \
                      aosym: str = 's2', comp: int = None) -> np.ndarray:
    #def _int_nuc_vloc(self, fakenuc, intor='int3c2e', aosym='s2', comp=None):
        '''Real space integrals {intor} for SR-Vnuc
        '''

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        # TODO rm if not needed
        nao = cell.nao_nr()
        nao_pair = nao * (nao+1) // 2

        int3c = self.gen_int3c_kernel(intor, aosym, comp=comp, j_only=True,
                                      auxcell=fakenuc)
        bufR, bufI = int3c()

        charge = -cell.atom_charges()
        #charge = cell.atom_charges()
        nchg   = len(charge)
        nchg2 = 2*nchg
        print('nchg, nao, nao_pair', nchg, nao, nao_pair)
        print('nchg', nchg)
        print('nchg', nchg)
        ## charge-of-nuccell, charge-of-fakenuc
        #charge = np.append(charge, -charge)
        #nchg2  = len(charge)
        if is_zero(kpts):
            print('shape bfR', np.shape(bufR) )
            mat = np.einsum('k...z,z->k...', bufR, charge)
            vj_at1 = np.einsum('kxz,z->kxz', bufR, charge)
            print('shape mat', np.shape(mat) )
            print('shape vj_at1', np.shape(vj_at1) )
            print('allclose mat and vj_at1?', np.allclose(mat, np.einsum('kxz->kx', vj_at1)) )
        else:
            mat = (np.einsum('k...z,z->k...', bufR, charge) +
                   np.einsum('k...z,z->k...', bufI, charge) * 1j)
            vj_at1 = (np.einsum('kxz,z->kxz', bufR, charge) +
                      np.einsum('kxz,z->kxz', bufI, charge) * 1j)
        vj_at1 = np.einsum('kxz->kzx', vj_at1)
        print('vj_at1 rearranged', np.shape(vj_at1) )

        # G = 0 contributions to SR integrals
        if (self.omega != 0 and
            (intor in ('int3c2e', 'int3c2e_sph', 'int3c2e_cart')) and
            (cell.dimension == 3)):
            #logger.debug2(self, 'G=0 part for %s', intor)
            nucbar = np.pi / self.omega**2 / cell.vol * charge.sum()
            nucbar_at = np.pi / self.omega**2 / cell.vol * charge
            print('shape nucbar', np.shape(nucbar), nucbar)
            print('shape nucbar', np.shape(nucbar_at), nucbar_at)
            if self.exclude_dd_block:
                rs_cell = self.rs_cell
                ovlp = rs_cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
                print('shape ovlp', np.shape(ovlp) )
                smooth_ao_idx = rs_cell.get_ao_type() == ft_ao.SMOOTH_BASIS
                for s in ovlp:
                    print('s in ovlp', np.shape(s) )
                    s[smooth_ao_idx[:,None] & smooth_ao_idx] = 0
                    print('s shape AFTER', np.shape(s) )
                recontract_2d = rs_cell.recontract(dim=2)
                ovlp = [recontract_2d(s) for s in ovlp]
                print('shape ovlp recontracted', np.shape(ovlp) )
            else:
                ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
                print('shape ovlp', np.shape(ovlp) )

            for k in range(nkpts):
                if aosym == 's1':
                    mat[k] -= nucbar * ovlp[k].ravel()
                else:
                    mat[k] -= nucbar * lib.pack_tril(ovlp[k])
            #atomic
            for k in range(nkpts):
                if aosym == 's1':
                    for i in range(nchg):
                        print('k, i, vj_at1[k,i,:]', k, i, np.shape(vj_at1[k,i,:]) )
                        vj_at1[k,i,:] -= nucbar_at[i] * ovlp[k].reshape(nao_pair)
                else:
                    for i in range(nchg):
                        print('k, i, vj_at1[k,i,:]', k, i, np.shape(vj_at1[k,i,:]) )
                        vj_at1[k,i,:] -= nucbar_at[i] * lib.pack_tril(ovlp[k])
            print('vj_at1 and mat, allclose?', np.allclose(mat, np.einsum('kix->kx', vj_at1) ) )
            print('vj_at1 returned shape', np.shape(vj_at1) )
        return vj_at1, mat

    _int_dd_block = _int_dd_block

    def get_pp_loc_part1(self, mesh=None, with_pseudo=True):
        if self.rs_cell is None:
            self.build()
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        aosym = 's2'
        nao_pair = nao * (nao+1) // 2
        mesh = self.mesh

        fakenuc = aft._fake_nuc(cell, with_pseudo=with_pseudo)
        vj, vj_tot = self._int_nuc_vloc(fakenuc)
        if cell.dimension == 0:
            raise NotImplementedError('No Ewald sum for dimension %s.', cell.dimension)
            #return lib.unpack_tril(vj)
        
        # TODO continue by re-checking here if merge_dd is fixed
        if self.exclude_dd_block:
            print('exclude_dd_block set to True')
            cell_d = self.rs_cell.smooth_basis_cell()
            if cell_d.nao > 0 and fakenuc.natm > 0:
                merge_dd = self.rs_cell.merge_diffused_block(aosym)
                if is_zero(kpts):
                    print('computing vj_dd')
                    vj_dd = self._int_dd_block(fakenuc)
                    vj_dd_at = _int_dd_block_at(self, fakenuc)
                    print('shape vj_dd', np.shape(vj_dd) )
                    print('shape vj_dd', np.shape(vj_dd_at) )
                    print('are they close?', np.allclose(vj_dd, np.einsum('kix->kx', vj_dd_at)) )
                    print('shape vj', np.shape(vj) )
                    vj1 = np.empty(np.shape(vj))
                    for i in range(2):
                        vj1[:,i,:] = merge_dd(vj[:,i,:], vj_dd_at[:,i,:])
                    #merge_dd(vj, vj_dd_at)
                    merge_dd(vj_tot, vj_dd)
                    print('shape after merge:', np.shape(vj),np.shape(vj_tot) )
                    print('shape1 after merge:', np.shape(vj1),np.shape(vj_tot) )
                else:
                    print('exclude_dd_block set to False')
                    vj_ddR, vj_ddI = self._int_dd_block(fakenuc)
                    for k in range(nkpts):
                        outR = vj[k].real.copy()
                        outI = vj[k].imag.copy()
                        merge_dd(outR, vj_ddR[k])
                        merge_dd(outI, vj_ddI[k])
                        vj[k] = outR + outI * 1j
        #print('vj_tot-vj', vj_tot[0] - vj[0,0,:] - vj[0,1,:])
        #print('vj_tot and vj after merge_dd', np.allclose(vj_tot, vj[0,0,:]+vj[0,1,:]) )
        #print('vj1_tot-vj', vj_tot[0] - vj1[0,0,:] - vj1[0,1,:])
        #print('vj_tot and vj1 after merge_dd', np.allclose(vj_tot, vj1[0,0,:]+vj1[0,1,:]) )
        # TODO continue by re-checking here if merge_dd is fixed

        print('vj_tot and vj, allclose?', np.allclose(vj_tot, np.einsum('kix->kx', vj)) )

        kpt_allow = np.zeros(3)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        b = cell.reciprocal_vectors()
        aoaux = ft_ao.ft_ao(fakenuc, Gv, None, b, gxyz, Gvbase)
        charges = -cell.atom_charges()

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            raise NotImplementedError('No Ewald sum for dimension %s.', cell.dimension)
            #coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
            #with lib.temporary_env(cell, dimension=3):
            #    coulG_SR = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv,
            #                                  omega=-self.omega)
            #coulG_LR = coulG - coulG_SR
        else:
            coulG_LR = pbc_tools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv,
                                          omega=self.omega)
        print('coulG_LR shape, kws', np.shape(coulG_LR), kws )
        wcoulG = coulG_LR * kws
        vG_tot = np.einsum('i,xi,x->x', charges, aoaux, wcoulG)
        vG = np.einsum('i,xi,x->xi', charges, aoaux, wcoulG)
        print('vG', np.shape(vG))
        print('allclose?', np.allclose(np.einsum('xi->x', vG), vG_tot))

        # contributions due to pseudo.pp_int.get_gth_vlocG_part1
        if cell.dimension == 3:
            G0_idx = 0
            exps = np.hstack(fakenuc.bas_exps())
            exps_chg = np.pi/exps * kws
            exps_chg  *= charges
            vG_tot[G0_idx] -= charges.dot(np.pi/exps) * kws
            for i in range(len(exps_chg)):
                vG[G0_idx,i] -= exps_chg[i]
        print('vG and vG_tto now close?', np.allclose(vG_tot, np.einsum('xi->x', vG)) )

        ft_kern = self.supmol_ft.gen_ft_kernel(aosym, return_complex=False,
                                               kpts=kpts)
        ngrids = Gv.shape[0]
        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
        Gblksize = max(16, int(max_memory*.8e6/16/(nao_pair*nkpts))//8*8)
        Gblksize = min(Gblksize, ngrids, 200000)
        vGR_tot = vG_tot.real
        vGI_tot = vG_tot.imag
        vGR = vG.real
        vGI = vG.imag
        print('shape vGR_tot, vGR', np.shape(vGR_tot), np.shape(vGR))

        buf = np.empty((2, nkpts, Gblksize, nao_pair))
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            # shape of Gpq (nkpts, nGv, nao_pair)
            Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, out=buf)
            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
                # rho_ij(G) nuc(-G) / G^2
                # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
                vR_tot = np.einsum('k,kx->x', vGR_tot[p0:p1], GpqR)
                vR_tot += np.einsum('k,kx->x', vGI_tot[p0:p1], GpqI)
                vj_tot[k] += vR_tot
                if not is_zero(kpts[k]):
                    vI_tot = np.einsum('k,kx->x', vGR_tot[p0:p1], GpqI)
                    vI_tot -= np.einsum('k,kx->x', vGI_tot[p0:p1], GpqR)
                    vj_tot[k].imag += vI_tot

                vR  = np.einsum('ji,jx->ix', vGR[p0:p1], GpqR)
                vR += np.einsum('ji,jx->ix', vGI[p0:p1], GpqI)
                vj[k] += vR
                if not is_zero(kpts[k]):
                    vI  = np.einsum('ji,jx->ix', vGR[p0:p1], GpqI)
                    vI += np.einsum('ji,jx->ix', vGI[p0:p1], GpqR)
                    vj[k] += vI * 1j
        print('vj and vj_tot', np.shape(vj), np.shape(vj_tot))
        print('vj and vj_tot close?', np.allclose(np.einsum('kix->kx', vj), vj_tot))
        print('vj and vj_tot close?', np.allclose(np.einsum('kix->kx', vj), vj_tot))

        #vj_kpts = []
        #for k, kpt in enumerate(kpts):
        #    if is_zero(kpt):
        #        vj_kpts.append(lib.unpack_tril(vj[k].real))
        #    else:
        #        vj_kpts.append(lib.unpack_tril(vj[k]))
        #return np.asarray(vj_kpts)
        # unpacking the triangular vj matrices
        vj_kpts_at = []
        for k, kpt in enumerate(kpts):
            if is_zero(kpt):
                vj_1atm_kpts = []
                for i in range(len(charges)):
                    vj_1atm_kpts.append(lib.unpack_tril(vj[k,i,:].real))
                vj_kpts_at.append(vj_1atm_kpts)
            else:
                vj_1atm_kpts = []
                for i in range(len(charges)):
                    vj_1atm_kpts.append(lib.unpack_tril(vj[k,i,:]))
                vj_kpts_at.append(vj_1atm_kpts)
        return np.asarray(vj_kpts_at)

# keep
def _int_dd_block_at(dfbuilder, fakenuc, intor='int3c2e', comp=None):
    '''
    The block of smooth AO basis in i and j of (ij|L) with full Coulomb kernel
    '''
    if intor not in ('int3c2e', 'int3c2e_sph', 'int3c2e_cart'):
        raise NotImplementedError

    cell = dfbuilder.cell
    cell_d = dfbuilder.rs_cell.smooth_basis_cell()
    nao = cell_d.nao
    kpts = dfbuilder.kpts
    nkpts = kpts.shape[0]
    if nao == 0 or fakenuc.natm == 0:
        if is_zero(kpts): 
            return np.zeros((nao,nao,1))
        else:
            return np.zeros((2,nkpts,nao,nao,1))

    mesh = cell_d.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    b = cell_d.reciprocal_vectors()
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])

    kpt_allow = np.zeros(3)
    charges = -cell.atom_charges()
    #:rhoG = np.dot(charges, SI)
    aoaux = ft_ao.ft_ao(fakenuc, Gv, None, b, gxyz, Gvbase) 
    print('int dd atomic: aoaux', np.shape(aoaux))
    rhoG = np.einsum('i,xi->xi', charges, aoaux)
    rhoG_tot = np.einsum('i,xi->x', charges, aoaux)
    print('int dd atomic: rhoG', np.shape(rhoG))
    print('int dd atomic: rhoG allclose?', np.allclose(rhoG_tot, np.einsum('xi->x', rhoG) ))
    coulG = pbc_tools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
    print('int dd atomic: coulG', np.shape(coulG))
    vG_tot = rhoG_tot * coulG
    #vG = rhoG * coulG
    rhoG = np.einsum('xi->ix', rhoG)
    print('int dd atomic: reordered rhoG', np.shape(rhoG))
    vG = [i*coulG for i in rhoG]
    print('int dd atomic: vG', np.shape(vG), np.allclose(vG_tot, vG[0]+vG[1]) )
    print('type vG', type(vG))
    vG = np.asarray(vG)
    print('vG as array', np.shape(vG))
    
    if cell.dimension == 3:
        print('vG_tot[0] before', vG_tot[0])
        vG_G0_tot = charges.dot(np.pi/np.hstack(fakenuc.bas_exps()))
        vG_G0 = np.zeros(len(charges))
        fakenucbas = np.pi/np.hstack(fakenuc.bas_exps())
        print('fakenucbas size', np.shape(fakenucbas), np.size(fakenucbas), fakenucbas)
        for z in range(len(charges)):
            vG_G0[z] = charges[z]*fakenucbas[z]
            print('z, vG_G0[z]', z, vG_G0[z])
            print('z, vG[z,0]', z, vG[z,0])
            vG[z,0] -= vG_G0[z]
            print('z, vG[z,0]', z, vG[z,0])
        print('vG_G0. sum, tot', vG_G0, np.sum(vG_G0), vG_G0_tot)
        vG_tot[0] -= charges.dot(np.pi/np.hstack(fakenuc.bas_exps()))
        print('vG_tot[0] after bckgr', vG_tot[0])
        
    else:
        raise NotImplementedError('No Ewald sum for dimension %s.', cell.dimension)
    print('int dd atomic: vG after bckrg', np.shape(vG), np.allclose(vG_tot, vG[0]+vG[1]) )

    vR_tot = pbc_tools.ifft(vG_tot, mesh).real
    print('vR_tot', np.shape(vR_tot))
    vR = pbc_tools.ifft(vG, mesh).real
    print('vR', np.shape(vR), np.shape(vR_tot))
    print('vR and vR_tot close?', np.allclose((vR[0]+vR[1]).real, vR_tot))

    coords = cell_d.get_uniform_grids(mesh)
    if is_zero(kpts):
        ao_ks = cell_d.pbc_eval_gto('GTOval', coords)
        # rearrange to (n_g x nao)
        #ao_ks = np.einsum('xa->ax', ao_ks)
        print('np.shape ao_ks', np.shape(ao_ks) )
        j3c = np.zeros((len(charges),nao,nao,1))
        for z in range(len(charges)):
            j3c[z] = lib.dot(ao_ks.T * vR[z], ao_ks).reshape(nao,nao,1)    
        j3c_tot = lib.dot(ao_ks.T * vR_tot, ao_ks).reshape(nao,nao,1)
        print('j3c_tot', np.shape(j3c_tot))
        print('j3c', np.shape(j3c), np.allclose(j3c_tot, j3c[0]+j3c[1]))

    else:
        ao_ks = cell_d.pbc_eval_gto('GTOval', coords, kpts=kpts)
        j3cR = np.empty((nkpts, len(charges), nao, nao))
        j3cI = np.empty((nkpts, len(charges), nao, nao))
        for k in range(nkpts):
            for z in range(len(charges)):
                v = lib.dot(ao_ks[k].conj().T * vR[z], ao_ks[k])
                j3cR[k,z,:] = v.real
                j3cI[k,z,:] = v.imag
        j3c = j3cR.reshape(nkpts,len(charges),nao,nao,1), j3cI.reshape(nkpts,len(charges),nao,nao,1)
    return j3c
######################################
######################################
######################################
class _CCNucBuilder(_CCGDFBuilder):

    exclude_dd_block = True

    def __init__(self, cell, kpts=np.zeros((1,3))):
        self.mesh = None
        self.fused_cell = None
        self.modchg_cell = None
        self.auxcell = self.rs_auxcell = None
        Int3cBuilder.__init__(self, cell, self.auxcell, kpts)

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
        logger.info(self, 'ke_cutoff = %s', self.ke_cutoff)
        logger.info(self, 'eta = %s', self.eta)
        logger.info(self, 'j2c_eig_always = %s', self.j2c_eig_always)
        return self

    def build(self, eta=None):
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        if cell.dimension == 0:
            self.eta, self.mesh, self.ke_cutoff = _guess_eta(cell, kpts, self.mesh)
        else:
            if eta is None:
                eta = max(.5/(.5+nkpts**(1./9)), ETA_MIN)
            ke_cutoff = estimate_ke_cutoff_for_eta(cell, eta)
            self.mesh = cell.cutoff_to_mesh(ke_cutoff)
            self.ke_cutoff = min(pbctools.mesh_to_cutoff(
                cell.lattice_vectors(), self.mesh)[:cell.dimension])
            self.eta = estimate_eta_for_ke_cutoff(cell, self.ke_cutoff)
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                self.mesh[2] = rsdf_builder._estimate_meshz(cell)
            elif cell.dimension < 2:
                self.mesh[cell.dimension:] = cell.mesh[cell.dimension:]
            self.mesh = cell.symmetrize_mesh(self.mesh)

        self.dump_flags()

        self.modchg_cell = _compensate_nuccell(cell, self.eta)
        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, rsdf_builder.RCUT_THRESHOLD, verbose=log)
        rcut = estimate_rcut(rs_cell, self.modchg_cell,
                             exclude_dd_block=self.exclude_dd_block)
        rcut_max = rcut.max()
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut_max, log)
        supmol.exclude_dd_block = self.exclude_dd_block
        self.supmol = supmol.strip_basis(rcut)
        log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())

        exp_min = np.hstack(cell.bas_exps()).min()
        lattice_sum_factor = max((2*cell.rcut)**3/cell.vol * 1/exp_min, 1)
        cutoff = cell.precision / lattice_sum_factor * .1
        self.direct_scf_tol = cutoff / cell.atom_charges().max()
        log.debug('Set _CCNucBuilder.direct_scf_tol to %g', cutoff)

        rcut = rsdf_builder.estimate_ft_rcut(rs_cell, cell.precision,
                                             self.exclude_dd_block)
        supmol_ft = rsdf_builder._ExtendedMoleFT.from_cell(rs_cell, kmesh,
                                                           rcut.max(), log)
        supmol_ft.exclude_dd_block = self.exclude_dd_block
        self.supmol_ft = supmol_ft.strip_basis(rcut)
        log.debug('sup-mol-ft nbas = %d cGTO = %d pGTO = %d',
                  supmol_ft.nbas, supmol_ft.nao, supmol_ft.npgto_nr())
        log.timer_debug1('initializing supmol', *cpu0)
        return self

    def _int_nuc_vloc(self, fakenuc, intor='int3c2e', aosym='s2', comp=None,
                      supmol=None):
        '''Vnuc - Vloc.
        '''
        logger.debug2(self, 'Real space integrals %s for Vnuc - Vloc', intor)

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)

        charge = -cell.atom_charges()
        if cell.dimension > 0:
            mod_cell = self.modchg_cell
            fakenuc = copy.copy(fakenuc)
            fakenuc._atm, fakenuc._bas, fakenuc._env = \
                    gto.conc_env(mod_cell._atm, mod_cell._bas, mod_cell._env,
                                 fakenuc._atm, fakenuc._bas, fakenuc._env)
            charge = np.append(-charge, charge)

        int3c = self.gen_int3c_kernel(intor, aosym, comp=comp, j_only=True,
                                      auxcell=fakenuc, supmol=supmol)
        bufR, bufI = int3c()

        if is_zero(kpts):
            mat = np.einsum('k...z,z->k...', bufR, charge)
        else:
            mat = (np.einsum('k...z,z->k...', bufR, charge) +
                   np.einsum('k...z,z->k...', bufI, charge) * 1j)

        # vbar is the interaction between the background charge
        # and the compensating function.  0D, 1D, 2D do not have vbar.
        if ((intor in ('int3c2e', 'int3c2e_sph', 'int3c2e_cart')) and
            (cell.dimension == 3 or
             (cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum'))):
            logger.debug2(self, 'G=0 part for %s', intor)

            # Note only need to remove the G=0 for mod_cell. when fakenuc is
            # constructed for pseudo potentail, don't remove its G=0 contribution
            charge = -cell.atom_charges()
            nucbar = (charge / np.hstack(mod_cell.bas_exps())).sum()
            nucbar *= np.pi/cell.vol
            if self.exclude_dd_block:
                rs_cell = self.rs_cell
                ovlp = rs_cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
                smooth_ao_idx = rs_cell.get_ao_type() == ft_ao.SMOOTH_BASIS
                for s in ovlp:
                    s[smooth_ao_idx[:,None] & smooth_ao_idx] = 0
                recontract_2d = rs_cell.recontract(dim=2)
                ovlp = [recontract_2d(s) for s in ovlp]
            else:
                ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)

            for k in range(nkpts):
                if aosym == 's1':
                    mat[k] -= nucbar * ovlp[k].ravel()
                else:
                    mat[k] -= nucbar * lib.pack_tril(ovlp[k])
        return mat

    #_int_dd_block = rsdf_builder._int_dd_block

    def get_pp_loc_part1(self, mesh=None, with_pseudo=True):
        log = logger.Logger(self.stdout, self.verbose)
        t0 = t1 = (logger.process_clock(), logger.perf_counter())
        if self.rs_cell is None:
            self.build()
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        aosym = 's2'
        nao_pair = nao * (nao+1) // 2
        mesh = self.mesh

        fakenuc = aft._fake_nuc(cell, with_pseudo=with_pseudo)
        vj = self._int_nuc_vloc(fakenuc)
        if cell.dimension == 0:
            return lib.unpack_tril(vj)

        if self.exclude_dd_block:
            cell_d = self.rs_cell.smooth_basis_cell()
            if cell_d.nao > 0 and fakenuc.natm > 0:
                merge_dd = self.rs_cell.merge_diffused_block(aosym)
                if is_zero(kpts):
                    vj_dd = self._int_dd_block(fakenuc)
                    merge_dd(vj, vj_dd)
                else:
                    vj_ddR, vj_ddI = self._int_dd_block(fakenuc)
                    for k in range(nkpts):
                        outR = vj[k].real.copy()
                        outI = vj[k].imag.copy()
                        merge_dd(outR, vj_ddR[k])
                        merge_dd(outI, vj_ddI[k])
                        vj[k] = outR + outI * 1j
        t0 = t1 = log.timer_debug1('vnuc pass1: analytic int', *t0)

        kpt_allow = np.zeros(3)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        b = cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        charges = -cell.atom_charges()

        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
            with lib.temporary_env(cell, dimension=3):
                coulG_full = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
            aoaux = ft_ao.ft_ao(self.modchg_cell, Gv, None, b, gxyz, Gvbase)
            vG = np.einsum('i,xi,x->x', charges, aoaux, coulG_full * kws)
            aoaux = ft_ao.ft_ao(fakenuc, Gv, None, b, gxyz, Gvbase)
            vG += np.einsum('i,xi,x->x', charges, aoaux, (coulG-coulG_full)*kws)
        else:
            aoaux = ft_ao.ft_ao(self.modchg_cell, Gv, None, b, gxyz, Gvbase)
            coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
            vG = np.einsum('i,xi,x->x', charges, aoaux, coulG * kws)

        ft_kern = self.supmol_ft.gen_ft_kernel(aosym, return_complex=False,
                                               verbose=log)
        ngrids = Gv.shape[0]
        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
        Gblksize = max(16, int(max_memory*.8e6/16/(nao_pair*nkpts))//8*8)
        Gblksize = min(Gblksize, ngrids, 200000)
        vGR = vG.real
        vGI = vG.imag
        log.debug1('max_memory = %s  Gblksize = %s  ngrids = %s',
                   max_memory, Gblksize, ngrids)

        buf = np.empty((2, nkpts, Gblksize, nao_pair))
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            # shape of Gpq (nkpts, nGv, nao_pair)
            Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts, out=buf)
            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
                # rho_ij(G) nuc(-G) / G^2
                # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
                vR = np.einsum('k,kx->x', vGR[p0:p1], GpqR)
                vR+= np.einsum('k,kx->x', vGI[p0:p1], GpqI)
                vj[k] += vR
                if not is_zero(kpts[k]):
                    vI = np.einsum('k,kx->x', vGR[p0:p1], GpqI)
                    vI-= np.einsum('k,kx->x', vGI[p0:p1], GpqR)
                    vj[k] += vI * 1j
            t1 = log.timer_debug1('contracting Vnuc [%s:%s]'%(p0, p1), *t1)
        log.timer_debug1('contracting Vnuc', *t0)

        vj_kpts = []
        for k, kpt in enumerate(kpts):
            if is_zero(kpt):
                vj_kpts.append(lib.unpack_tril(vj[k].real))
            else:
                vj_kpts.append(lib.unpack_tril(vj[k]))
        return np.asarray(vj_kpts)
