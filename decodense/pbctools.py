#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
pbc module
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
from pyscf.pbc import gto as pbc_gto  
from pyscf.pbc import scf as pbc_scf 
from pyscf.pbc.df import ft_ao
from pyscf.pbc.gto import pseudo
from pyscf.pbc.tools import pbc as pyscf_pbctools
from pyscf.pbc.df.incore import _Int3cBuilder, _compensate_nuccell, _fake_nuc, _strip_basis, aux_e2
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from scipy.special import erf, erfc
from typing import List, Tuple, Dict, Union, Any

libpbc = lib.load_library('libpbc')

PRECISION = getattr(__config__, 'pbc_df_aft_estimate_eta_precision', 1e-8)


class _IntNucBuilder(_Int3cBuilder):
    """
    The integral builder for E_ne term when GDF is used. 
    """
    def __init__(self, cell, kpts=np.zeros((1,3))):
        # cache ovlp_mask
        self._supmol = None
        self._ovlp_mask = None
        self._cell0_ovlp_mask = None
        _Int3cBuilder.__init__(self, cell, None, kpts)

    def get_ovlp_mask(self, cutoff, supmol=None, cintopt=None):
        """
        ovlp_mask can be reused for different types of intor
        """
        if self._ovlp_mask is None or supmol is not self._supmol:
            self._ovlp_mask, self._cell0_ovlp_mask = \
                    _Int3cBuilder.get_ovlp_mask(self, cutoff, supmol, cintopt)
            self._supmol = supmol
        return self._ovlp_mask, self._cell0_ovlp_mask

    def _int_nuc_vloc(self, nuccell, intor='int3c2e', aosym='s2', comp=None,
                      with_pseudo=True, supmol=None):
        """
        Vnuc - Vloc in R-space
        """
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        nao_pair = nao * (nao+1) // 2

        # use the 3c2e code with steep s gaussians to mimic nuclear density
        # (nuccell is the cell for model charges)
        fakenuc = _fake_nuc(cell, with_pseudo=with_pseudo)
        fakenuc._atm, fakenuc._bas, fakenuc._env = \
                gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                             fakenuc._atm, fakenuc._bas, fakenuc._env)
        int3c = self.gen_int3c_kernel(intor, aosym, comp=comp, j_only=True,
                                      auxcell=fakenuc, supmol=supmol)
        bufR, bufI = int3c()

        charge = cell.atom_charges()
        nchg   = len(charge)
        # charge-of-nuccell, charge-of-fakenuc
        charge = np.append(charge, -charge) 
        nchg2  = len(charge)
        # sum over halves, chrg and -chrg ints 
        if is_zero(kpts):
            vj_at1 = np.einsum('kxz,z->kzx', bufR, charge)
            vj_at  = vj_at1[:,nchg:,:] + vj_at1[:,:nchg,:] 
        else:
            vj_at1 = (np.einsum('kxz,z->kzx', bufR, charge) +
                      np.einsum('kxz,z->kzx', bufI, charge) * 1j)
            vj_at  = vj_at1[:,nchg:,:] + vj_at1[:,:nchg,:] 

        # vbar is the interaction between the background charge
        # and the compensating function.  0D, 1D, 2D do not have vbar.
        if cell.dimension == 3 and intor in ('int3c2e', 'int3c2e_sph',
                                             'int3c2e_cart'):
            charge = -cell.atom_charges()

            nucbar = np.asarray([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
            nucbar *= np.pi/cell.vol

            ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
            for k in range(nkpts):
                if aosym == 's1':
                    for i in range(nchg):
                        vj_at[k,i,:] -= nucbar[i] * ovlp[k].reshape(nao_pair) 
                else:
                    for i in range(nchg):
                        vj_at[k,i,:] -= nucbar[i] * lib.pack_tril(ovlp[k])
        return vj_at

    def get_nuc(self, mesh=None, with_pseudo=False):
        """
        Vnuc term 
        """
        from pyscf.pbc.df.gdf_builder import _guess_eta

        cell = self.cell
        charges = cell.atom_charges()
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        aosym = 's2'
        nao_pair = nao * (nao+1) // 2

        kpt_allow = np.zeros(3)
        eta, mesh, ke_cutoff = _guess_eta(cell, kpts, mesh)

        # check for cell with partially de-conracted basis 
        if self.rs_cell is None:
            self.build()
        # initialize an extended Mole object to mimic periodicity
        # remote basis removed if they do not contribute to the FT of basis product
        self.supmol = supmol = _strip_basis(self.supmol, eta)

        # initialize a cell of the compensated Gaussian charges for nucleus
        modchg_cell = _compensate_nuccell(cell, eta)
        # R-space integrals for Vnuc - Vloc
        vj_at = self._int_nuc_vloc(modchg_cell, with_pseudo=with_pseudo,
                                supmol=supmol)

        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        # Coulomb kernel for all G-vectors
        coulG = pyscf_pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv) * kws
        # analytical FT AO-pair product
        aoaux = ft_ao.ft_ao(modchg_cell, Gv)
        # G-space integrals for Vnuc - Vloc
        vG1 = np.einsum('i,xi->xi', -charges, aoaux) 
        vG_at = np.einsum('x,xi->xi', coulG, vG1)

        # initialize an extended Mole object to mimic periodicity
        supmol_ft = ft_ao._ExtendedMole.from_cell(self.rs_cell, self.bvk_kmesh)
        # remote basis removed if they do not contribute to the FT of basis product
        supmol_ft = supmol_ft.strip_basis()
        # generate the analytical FT kernel for AO products
        ft_kern = supmol_ft.gen_ft_kernel(aosym, return_complex=False)

        Gv, Gvbase, kws = modchg_cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = Gv.shape[0]
        # TODO do i need this mem limit? for assigning blocks of ints size
        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
        Gblksize = max(16, int(max_memory*1e6/16/nao_pair/nkpts))
        Gblksize = min(Gblksize, ngrids, 200000)
        vG_atR = vG_at.real
        vG_atI = vG_at.imag

        buf = np.empty((2, nkpts, Gblksize, nao_pair))
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            # analytical FT kernel for AO-products (ao values on a G-grid)
            Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts, out=buf)
            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
                # contract potential on grid points with value of the ao on that grid point 
                vR_at  = np.einsum('ji,jx->ix', vG_atR[p0:p1], GpqR)
                vR_at += np.einsum('ji,jx->ix', vG_atI[p0:p1], GpqI)
                vj_at[k] += vR_at
                if not is_zero(kpts[k]):
                    vI_at  = np.einsum('ji,jx->ix', vG_atR[p0:p1], GpqI)
                    vI_at += np.einsum('ji,jx->ix', vG_atI[p0:p1], GpqR)
                    vj_at[k] += vI_at * 1j

        # unpacking the triangular vj matrices
        vj_kpts_at = []
        for k, kpt in enumerate(kpts):
            if is_zero(kpt):
                vj_1atm_kpts = []
                for i in range(len(charges)):
                    vj_1atm_kpts.append(lib.unpack_tril(vj_at[k,i,:].real))
                vj_kpts_at.append(vj_1atm_kpts)
            else:
                vj_1atm_kpts = []
                for i in range(len(charges)):
                    vj_1atm_kpts.append(lib.unpack_tril(vj_at[k,i,:]))
                vj_kpts_at.append(vj_1atm_kpts)
        return np.asarray(vj_kpts_at)

    def get_pp_loc_part1(self, mesh=None):
        return self.get_nuc(mesh, with_pseudo=True)

        # TODO continue here
    def get_pp_loc_part2(self):
        '''PRB, 58, 3641 Eq (1), integrals associated to C1, C2, C3, C4
        '''
        '''
            Fake cell created to "house" each coeff.*gaussian (on each atom that has it) 
            for V_loc of pseudopotential (1 fakecell has max 1 gaussian per atom). 
            Ergo different nr of coeff. ->diff. nr of ints to loop and sum over for diff. atoms
            See: "Each term of V_{loc} (erf, C_1, C_2, C_3, C_4) is a gaussian type
            function. The integral over V_{loc} can be transfered to the 3-center
            integrals, in which the auxiliary basis is given by the fake cell."
            Later the cell and fakecells are concatenated to compute 3c overlaps between 
            basis funcs on the real cell & coeff*gaussians on fake cell?
            TODO check if this is correct
            <X_P(r)| sum_A^Nat [ -Z_Acore/r erf(r/sqrt(2)r_loc) + sum_i C_iA (r/r_loc)^(2i-2) ] |X_Q(r)>
            -> 
            int X_P(r - R_P)     :X_P actual basis func. that sits on atom P  ??
            * Ci                 :coeff for atom A, coeff nr i
        '''
        if self.rs_cell is None:
            self.build()
        cell = self.cell
        supmol = self.supmol
        if supmol.nbas == supmol.bas_mask.size:  # supmol not stripped
            supmol = self.supmol.strip_basis(inplace=False)
        kpts = self.kpts
        nkpts = len(kpts)
        natm = cell.natm
        nao = cell.nao_nr()
        # nao_pairs for i<=j upper triangular fx, incl diagonal
        nao_pair = nao * (nao+1) // 2

        intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
                  'int3c1e_r4_origk', 'int3c1e_r6_origk')

        bufR_at = np.zeros((nkpts, natm, nao_pair))
        bufI_at = np.zeros((nkpts, natm, nao_pair))
        # Loop over coefficients to generate: erf, C1, C2, C3, C4
        # each coeff.-gaussian put in its own fakecell
        # If cn = 0, the erf term is generated.  C_1,..,C_4 are generated with cn = 1..4
        # buf is a buffer array to gather all integrals into before unpacking
        for cn in range(1, 5):
            fake_cell = pseudo.pp_int.fake_cell_vloc(cell, cn)
            if fake_cell.nbas > 0:
                # Make a list on which atoms the gaussians sit on (for the current Ci coeff.)
                fakebas_atom_lst = []
                for i in range(fake_cell.nbas):
                    fakebas_atom_lst.append(fake_cell.bas_atom(i))
                fakebas_atom_ids = np.array(fakebas_atom_lst)
                #
                int3c = self.gen_int3c_kernel(intors[cn], 's2', comp=1, j_only=True,
                                              auxcell=fake_cell, supmol=supmol)
                # The int over V_{loc} can be transfered to the 3-center
                # integrals, in which the aux. basis is given by the fake cell.
                # v is (nkpts, naopairs, naux)
                vR, vI = int3c()
                # Put the ints for this Ci coeff. in the right places in the 
                # buffer (i.e. assign to the right atom)
                # k is kpt, i is aux, j is aopair
                vR_at = np.einsum('kij->kji', vR) 
                for k, kpt in enumerate(kpts):
                    bufR_at[k, fakebas_atom_lst] += vR_at[k]
                if vI is not None:
                    vI_at = np.einsum('kij->kji', vI) 
                    for k, kpt in enumerate(kpts):
                        bufI_at[k, fakebas_atom_lst] += vI_at[k]

        # if fakecell.nbas are all < 0, buf consists of zeros and we check for elements in the system 
        if not np.any(bufR_at) :
            if any(cell.atom_symbol(ia) in cell._pseudo for ia in range(cell.natm)):
                pass
            else:
               raise ValueError('cell.pseudo was specified but its elements %s '
                             'were not found in the system (pp_part2).', cell._pseudo.keys())
            # list of zeros, length nkpts returned when no pp found on atoms
            vpploc_at = [0] * nkpts
        else:
            # rearrange with einsum
            buf_at = (bufR_at + bufI_at * 1j)
            vpploc_at = []
            # now have the triangular matrix for each k (triangular of nao x nao is n_aopairs)
            # unpack here to nao x nao for each atom
            for k, kpt in enumerate(kpts):
                vpploc_1atm_kpts = [] #
                for i in range(natm): #
                    v_1atm_ints = lib.unpack_tril(buf_at[k,i,:]) #
                    if abs(kpt).sum() < 1e-9:  # gamma_point:
                         v_1atm_ints = v_1atm_ints.real #
                    vpploc_1atm_kpts.append(v_1atm_ints) #
                vpploc_at.append(vpploc_1atm_kpts) #
        return np.asarray(vpploc_at)

    def get_pp_nl(self):
        '''Nonlocal; contribution. See PRB, 58, 3641 Eq (2).
           Done by generating a fake cell for putting V_{nl} gaussian 
           function p_i^l Y_{lm} in (on the atoms the corr. core basis 
           func. would sit on). Later the cells are concatenated to 
           compute overlaps between basis funcs in the real cell & proj. 
           in fake cell (splitting the ints into two ints to multiply).
           ------------------------------------------------------------
            <X_P(r)| sum_A^Nat sum_i^3 sum_j^3 sum_m^(2l+1) Y_lm(r_A) p_lmi(r_A) h^l_i,j p_lmj(r'_A) Y*_lm(r'_A) |X_Q(r')>
            -> (Y_lm implicit in p^lm)
            int X_P(r - R_P) p^lm_i(r - R_A) dr  
            * h^A,lm_i,j                    
            int p^lm_j(r' - R_A) X(r' - R_Q) dr  
           ------------------------------------------------------------
           Y_lm: spherical harmonic, l ang.mom. qnr
           p_i^l: Gaussian projectors (PRB, 58, 3641 Eq 3)
           hl_blocks: coeff. for nonlocal projectors
           h^A,lm_i,j: coeff for atom A, lm,ij 
           (i & j run up to 3: never larger atom cores than l=3 (d-orbs))
           X_P: actual basis func. that sits on atom P
           X_Q: actual basis func. that sits on atom Q
           A sums over all atoms since each might have a pp 
           that needs projecting out core sph. harm.
        '''
        cell = self.cell
        kpts = self.kpts
        if kpts is None:
            kpts_lst = np.zeros((1,3))
        else:
            kpts_lst = np.reshape(kpts, (-1,3))
        nkpts = len(kpts_lst)

        # Generate a fake cell for V_{nl}.gaussian functions p_i^l Y_{lm}. 
        fakecell, hl_blocks = pseudo.pp_int.fake_cell_vnl(cell)
        ppnl_half = pseudo.pp_int._int_vnl(cell, fakecell, hl_blocks, kpts_lst)
        nao = cell.nao_nr()
        natm = cell.natm
        buf = np.empty((3*9*nao), dtype=np.complex128)

        # Set ppnl equal to zeros in case hl_blocks loop is skipped
        # and ppnl is returned
        ppnl = np.zeros((nkpts,natm,nao,nao), dtype=np.complex128)
        for k, kpt in enumerate(kpts_lst):
            offset = [0] * 3
            # hlblocks: for each atom&ang.mom. there is a matrix of coeff. 
            # e.g. 2ang.mom. on two atoms A and B would give A1 1x1 matrix, 
            # A2 1x1 matrix, B1 1x1 matrix, B2 1x1 matrix (if only one kind 
            # of a projector for these ang.mom. for these atoms).
            for ib, hl in enumerate(hl_blocks):
                # This loop is over hlij for all atoms and ang.momenta
                # I think this is shell, hl coeff pair.
                # Either way ib is bas_id and called with bas_atom gives 
                # the atom id the coeff. belongs to. 
                # l is the angular mom. qnr associated with given basis
                l = fakecell.bas_angular(ib)
                atm_id_hl = fakecell.bas_atom(ib)
                # orb magn nr 2L+1
                nd = 2 * l + 1
                # dim of the hl coeff. array
                hl_dim = hl.shape[0]
                ilp = np.ndarray((hl_dim,nd,nao), dtype=np.complex128, buffer=buf)
                for i in range(hl_dim):
                    # p0 takes care that the right m,l sph.harm are taken in projectors
                    p0 = offset[i]
                    ilp[i] = ppnl_half[i][k][p0:p0+nd]
                    offset[i] = p0 + nd
                ppnl[k,atm_id_hl] += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
        
        if abs(kpts_lst).sum() < 1e-9:  # gamma_point:
            ppnl = ppnl.real
        return ppnl


def get_nuc_atomic_df(mydf, kpts=None):
    ''' Nucl.-el. attraction for all el. calculation '''
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    dfbuilder = _IntNucBuilder(mydf.cell, kpts_lst)
    vne_at = dfbuilder.get_nuc(mydf.mesh, with_pseudo=False)
    if kpts is None or np.shape(kpts) == (3,):
        # if gamma point
        if np.allclose(kpts_lst, np.zeros((1,3))):
            vne_at = vne_at[0].real
        else:
            vne_at = vne_at[0]
    return vne_at

def get_pp_atomic_df(mydf, kpts=None):
    ''' Nucl.-el. attraction for calculation using pseudopotentials'''
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    dfbuilder = _IntNucBuilder(mydf.cell, kpts_lst)

    # returns nkpts x nao x nao
    cell = dfbuilder.cell
    kpts = dfbuilder.kpts
    vloc1_at = dfbuilder.get_pp_loc_part1(mydf.mesh)
    vloc2_at = dfbuilder.get_pp_loc_part2()
    vnl_at = dfbuilder.get_pp_nl()
    vpp_total = vloc1_at + vloc2_at + vnl_at

    if abs(kpts_lst).sum() < 1e-9:
        vpp_total = vpp_total[0]
        vloc1_at = vloc1_at[0]
        vloc2_at = vloc2_at[0]
        vnl_at   = vnl_at[0]
    return vpp_total, vloc1_at+vloc2_at, vnl_at



#====================FFTDF====================#
def get_nuc_atomic_fftdf(mydf, kpts=None):
    ''' V_nuc for all el. calc. with FFT density fitting (not recommended)  '''
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    cell = mydf.cell
    mesh = mydf.mesh
    charge = -cell.atom_charges()
    Gv = cell.get_Gv(mesh)
    # SI: ngrids
    SI = cell.get_SI(Gv)
    natm, ngrids = np.shape(SI)
    nkpts = len(kpts_lst)
    nao = cell.nao_nr()

    rhoG_at = np.einsum('z,zg->zg', charge, SI)

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    vneG_at = np.einsum('zg,g->zg', rhoG_at, coulG)
    # vne evaluated in real-space
    vneR_at = np.zeros((natm, ngrids))
    for a in range(natm):
        vneR_at[a] = tools.ifft(vneG_at[a], mesh).real

    # vneR: natm x ngrids
    # vne: nkpts x natm x nao x nao
    vne_at = np.zeros((nkpts, natm, nao, nao))
    for a in range(natm):
        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_lst):
            ao_ks = ao_ks_etc[0]
            for k, ao in enumerate(ao_ks):
                vne_at[k,a] += lib.dot(ao.T.conj()*vneR_at[a,p0:p1], ao)
            ao = ao_ks = None

    if kpts is None or np.shape(kpts) == (3,):
        # if gamma point
        if np.allclose(kpts_lst, np.zeros((1,3))):
            vne_at = vne_at[0].real
        else:
            vne_at = vne_at[0]
    return np.asarray(vne_at)

def get_pp_atomic_fftdf(mydf, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    from pyscf import gto
    cell = mydf.cell
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    nkpts = len(kpts_lst)
    nao = cell.nao_nr()

    mesh = mydf.mesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(mesh)
    # vpplocG: natm x ngrid
    vpplocG = pseudo.get_vlocG(cell, Gv)
    natm, ngrids = np.shape(vpplocG)
    vpplocG_at = -np.einsum('ij,ij->ij', SI, vpplocG)

    # vpploc evaluated in real-space
    vpplocR_at = np.zeros((natm, ngrids))
    for a in range(natm):
        vpplocR_at[a] = tools.ifft(vpplocG_at[a], mesh).real

    vpp_at = np.zeros((nkpts, natm, nao, nao), dtype=np.complex128)
    for a in range(natm):
        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_lst):
            ao_ks = ao_ks_etc[0]
            for k, ao in enumerate(ao_ks):
                vpp_at[k,a] += lib.dot(ao.T.conj()*vpplocR_at[a, p0:p1], ao)
            ao = ao_ks = None

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    # buf for SPG_lmi upto l=0..3 and nl=3
    buf = np.empty((48,ngrids), dtype=np.complex128)
    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (1/cell.vol)**.5

        vppnl_at = np.zeros((natm, nao, nao), dtype=np.complex128)
        # loop over atoms, check if they have pp
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            # check which shells are in the pp, which hl coeff. it has
            # l is shell/ang.mom., rl is r_loc, nl is nr of l/these shells,
            # hl is the h^shell coefficients 
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                # if this l in pp, need coeff/ints to project out 
                if nl > 0:
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                    # pYlm_part: nPWgrid x nr of ml
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)

                    p0, p1 = p1, p1+nl*(l*2+1)
                    # pYlm is real, SI[ia] is complex
                    # pYlm: nPWgrid x nr of ml
                    pYlm = np.ndarray((nl,l*2+1,ngrids), dtype=np.complex128, buffer=buf[p0:p1])
                    # loop over these shells, e.g. 1s, 2s,3s..
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl

            # i think this checks if there are ml, diff. orientations of ang. mom.
            if p1 > 0:
                # n is nr of 2e aos in pp
                # SPG_lmi: n x nPWgrid 
                SPG_lmi = buf[:p1]
                # SI: natm x nPWgrid
                SPG_lmi *= SI[ia].conj()
                # SPG_lm_aoGs: n x nao 
                SPG_lm_aoGs = lib.zdot(SPG_lmi, aokG)
                p1 = 0
                # loop over shells with l>0, get the coeff hl and ints
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = np.asarray(hl)
                        # SPG_lm_aoG, tmp: n_hl_dim x n_ml x nao
                        # n_hl_dim: nr of type of shells (s, p, d) 
                        # indices: j is n_hl_dim 
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        # vppnl_at: natm x nao x nao for one kpt
                        # pack here in correct place for atom
                        vppnl_at[ia] += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl_at * (1./cell.vol)
    
    vpp_tot_at = np.zeros((nkpts, natm, nao, nao), dtype=np.complex128)
    vppnl_at = np.zeros((nkpts, natm, nao, nao), dtype=np.complex128)
    for k, kpt in enumerate(kpts_lst):
        vppnl_at[k] = vppnl_by_k(kpt)
        if gamma_point(kpt):
            vpp_tot_at[k] = vpp_at[k].real + vppnl_at[k].real
            vpp_tot_at[k] = vpp_tot_at[k].real 
            vpp_at[k] = vpp_at[k].real 
            vppnl_at[k] = vppnl_at[k].real
        else:
            vpp_tot_at[k] = vpp_at[k] + vppnl_at[k]

    if kpts is None or np.shape(kpts) == (3,):
        # if gamma point
        if np.allclose(kpts_lst, np.zeros((1,3))):
            vpp_tot_at = vpp_tot_at[0].real
            vpp_at = vpp_at[0].real
            vppnl_at = vppnl_at[0].real
        else:
            vpp_tot_at = vpp_tot_at[0]
            vpp_at = vpp_at[0]
            vppnl_at = vppnl_at[0]
    return vpp_tot_at, vpp_at, vppnl_at


def ewald_e_nuc(cell: pbc_gto.Cell) -> np.ndarray:
    """
    This function returns the nuc-nuc repulsion energy for a cell
    by performing real (R) and reciprocal (G) space Ewald sum, 
    which consists of overlap, self and G-space sum 
    (Formulation of Martin, App. F2.).
    """ 
    def cut_mesh_for_ewald(cell, mesh):
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

