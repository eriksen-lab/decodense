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
from pyscf.pbc.df import ft_ao
from pyscf.pbc.gto import pseudo
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.df.incore import Int3cBuilder
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
    vpp_loc1_at = get_pp_loc_part1(mydf, kpts, with_pseudo=True)
    print('shape vpp_loc1_at', np.shape(vpp_loc1_at))

    pp2builder = _IntPPBuilder(cell, kpts)
    vpp_loc2_at = pp2builder.get_pp_loc_part2()
    print('shape pyscf vpp2', np.shape(vpp2))
    vpp_nl_at = get_pp_nl(cell, kpts)
    print('shape pyscf vnl', np.shape(vnl))
    
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

#def get_pp_loc_part1(mydf, mesh: Union[List[int], np.ndarray] = None, \
def get_pp_loc_part1(mydf, kpts=None, with_pseudo: bool = True) -> np.ndarray:
    """
    Vnuc term 
    """
# TODO needed ?
#    from pyscf.pbc.df.gdf_builder import _guess_eta

# TODO check this, introduce
    kpts, is_single_kpt = _check_kpts(mydf, kpts)
    cell = mydf.cell
    mesh = np.asarray(mydf.mesh)
    charges = cell.atom_charges()
    #kpts = self.kpts
    natm = cell.natm
    nkpts = len(kpts)
    nao = cell.nao_nr()
    #aosym = 's2'
    nao_pair = nao * (nao+1) // 2
    #print('nao_pair', nao_pair, nao)

    kpt_allow = np.zeros(3)
# TODO check this, introduce
#    eta, mesh, ke_cutoff = _guess_eta(cell, kpts, mesh)
    ke_guess = estimate_ke_cutoff(cell, cell.precision)
    mesh = cell.mesh
    mesh_guess = cell.cutoff_to_mesh(ke_guess)
    if np.any(mesh < mesh_guess*KE_SCALING):
        # TODO make a custom warning: '(mydf, 'mesh %s is not enough for AFTDF.get_nuc function to get integral accuracy %g.\nRecommended mesh is %s.', mesh, cell.precision, mesh_guess)'
        raise Warning

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
