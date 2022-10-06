#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
pbc module
"""

__author__ = 'Luna Zamok, Technical University of Denmark, DK'
__maintainer__ = 'Luna Zamok'
__email__ = 'luza@kemi.dtu.dk'
__status__ = 'Development'

import numpy as np
from scipy.special import erf, erfc
#from typing import List, Tuple, Dict, Union, Any

import copy
import ctypes
from pyscf import gto
from pyscf import lib
from pyscf import __config__
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc import gto as pbc_gto  
from pyscf.pbc import scf as pbc_scf 
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.incore import _Int3cBuilder, _compensate_nuccell, _fake_nuc, _strip_basis, aux_e2
from pyscf.pbc.gto import pseudo
from pyscf.pbc.tools import pbc as pyscf_pbctools
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point

libpbc = lib.load_library('libpbc')

PRECISION = getattr(__config__, 'pbc_df_aft_estimate_eta_precision', 1e-8)

''' Nuclear repulsion term '''
# almost identical to ewald in cell.py
# TODO: only works for 3D cells, extend to lower dim.
def ewald_e_nuc(cell: pbc_gto.Cell) -> np.ndarray:
    """
    this function returns the nuc-nuc repulsion energy for a cell
    """ 
    '''Perform real (R) and reciprocal (G) space Ewald sum for the energy,
       partitioned into atomic contributions.
    Formulation of Martin, App. F2.
    Returns:
        array of floats
            The Ewald energy consisting of overlap, self, and G-space sum.
    See Also:
        pyscf.pbc.gto.get_ewald_params
    '''
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

    # If lattice parameter is not set, the cell object is treated as a mole
    # object. The nuclear repulsion energy is computed.
    if cell.a is None:
        return mole.energy_nuc(cell)

    if cell.natm == 0:
        return 0

    # get the Ewald 'eta' (exponent eta^2 of the model Gaussian charges) and 
    # 'cut' (real space cut-off) parameters 
    #if ew_eta is None: ew_eta = mol.get_ewald_params()[0]
    #if ew_cut is None: ew_cut = mol.get_ewald_params()[1]
    ew_eta = cell.get_ewald_params()[0]
    ew_cut = cell.get_ewald_params()[1]

    # atom coord: [a,b] a=atom, b=cart.coord
    chargs = cell.atom_charges()
    coords = cell.atom_coords()

    # (Cartesian, unitful) lattice translation vectors for nearby images
    # in bohr (prim. lattice vectors (cell.a) in Å)
    Lall = cell.get_lattice_Ls(rcut=ew_cut)

    # distances between atoms in cell 0 and nearby images
    # [L,i,j,d] where L is cell index; i is atom index in cell 0; 
    # j is atom index in cell L; d is cart. component
    rLij = coords[:,None,:] - coords[None,:,:] + Lall[:,None,None,:]
    # euclidean distances 
    # (n_neighb_cells x n_atoms x n_atoms)
    r = np.sqrt(np.einsum('Lijx,Lijx->Lij', rLij, rLij))
    rLij = None
    # "eliminate" self-distances -> self-terms skipped (R) sum? 
    r[r<1e-16] = 1e200
    
    # (R) Ewald sum (shape: n_atoms)
    ewovrl_atomic = .5 * np.einsum('i,j,Lij->i', chargs, chargs, erfc(ew_eta * r) / r)
    
    # Ewald self-term: cancels self-contribution in (G) sum
    # last line of Eq. (F.5) in Martin
    ewself_factor = -.5 * 2 * ew_eta / np.sqrt(np.pi)
    ewself_atomic = np.einsum('i,i->i', chargs,chargs)
    ewself_atomic = ewself_atomic.astype(float)
    ewself_atomic *= ewself_factor 
    if cell.dimension == 3:
        ewself_atomic += -.5 * (chargs*np.sum(chargs)).astype(float) * np.pi/(ew_eta**2 * cell.vol)

    # g-space sum (using g grid) (Eq. (F.6) in Electronic Structure by Richard M. Martin
    #, but note errors as below)
    # Eq. (F.6) in Martin is off by a factor of 2, the
    # exponent is wrong (8->4) and the square is in the wrong place
    #
    # Formula should be
    #   1/2 * 4\pi / Omega \sum_I \sum_{G\neq 0} |ZS_I(G)|^2 \exp[-|G|^2/4\eta^2]
    # where
    #   ZS_I(G) = \sum_a Z_a exp (i G.R_a)
    # See also Eq. (32) of ewald.pdf at
    #   http://www.fisica.uniud.it/~giannozz/public/ewald.pdf
    #
    # (g-grid) of reciprocal lattice vectors
    mesh = cut_mesh_for_ewald(cell, cell.mesh)
    Gv, Gvbase, Gv_weights = cell.get_Gv_weights(mesh)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    # exclude the G=0 vector
    absG2[absG2==0] = 1e200

    if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
        coulG = 4*np.pi / absG2
        # todo is this omega in eq.(5)?
        coulG *= Gv_weights
        # get_SI(k_vecs) gets the structure factors, n_atm*n_grid 
        # todo get them only once, save?
        ZSI_total = np.einsum("i,ij->j", chargs, cell.get_SI(Gv))
        ZSI_atomic = np.einsum("i,ij->ij", chargs, cell.get_SI(Gv)) 
        ZexpG2_atomic = ZSI_atomic * np.exp(-absG2/(4*ew_eta**2))
        # todo diff if atomic part conjugated insead?
        ewg_atomic = .5 * np.einsum('j,ij,j->i', ZSI_total.conj(), ZexpG2_atomic, coulG).real

    else:
        logger.warn(cell, 'No method for PBC dimension %s, dim-type %s.',
                    cell.dimension)
        raise NotImplementedError
    
    ##TODO maybe our own warnings instead of pyscf logger
    #logger.debug(cell, 'Ewald components = %.15g, %.15g, %.15g', ewovrl_atomic, ewself_atomic, ewg_atomic)
    return ewovrl_atomic + ewself_atomic + ewg_atomic


#====================DF====================#
def get_nuc_atomic(mydf, kpts=None):
    ''' Nucl.-el. attraction '''
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    dfbuilder = _IntNucBuilder(mydf.cell, kpts_lst)
    vj_at = dfbuilder.get_nuc(mydf.mesh)
    if kpts is None or np.shape(kpts) == (3,):
        vj_at = vj_at[0]
    return vj_at

def get_pp_atomic(mydf, kpts=None):
    # this is from aft/get_pp and df/incore/get_pp levels
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    dfbuilder = _IntNucBuilder(mydf.cell, kpts_lst)

    # rn returns nkpts x nao x nao
    # FIXME is there a better way?
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
    return vpp_total, vloc1_at, vloc2_at, vnl_at


class _IntNucBuilder(_Int3cBuilder):
    '''In this builder, ovlp_mask can be reused for different types of intor
    '''
    def __init__(self, cell, kpts=np.zeros((1,3))):
        # cache ovlp_mask
        self._supmol = None
        self._ovlp_mask = None
        self._cell0_ovlp_mask = None
        _Int3cBuilder.__init__(self, cell, None, kpts)

    def get_ovlp_mask(self, cutoff, supmol=None, cintopt=None):
        if self._ovlp_mask is None or supmol is not self._supmol:
            self._ovlp_mask, self._cell0_ovlp_mask = \
                    _Int3cBuilder.get_ovlp_mask(self, cutoff, supmol, cintopt)
            self._supmol = supmol
        return self._ovlp_mask, self._cell0_ovlp_mask

    def _int_nuc_vloc(self, nuccell, intor='int3c2e', aosym='s2', comp=None,
                      with_pseudo=True, supmol=None):
        '''Vnuc - Vloc. nuccell is the cell for model charges
        '''

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        nao_pair = nao * (nao+1) // 2

        # Use the 3c2e code with steep s gaussians to mimic nuclear density
        fakenuc = _fake_nuc(cell, with_pseudo=with_pseudo)
        fakenuc._atm, fakenuc._bas, fakenuc._env = \
                gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                             fakenuc._atm, fakenuc._bas, fakenuc._env)

        int3c = self.gen_int3c_kernel(intor, aosym, comp=comp, j_only=True,
                                      auxcell=fakenuc, supmol=supmol)
        # nkpts x nao_pair x nchg2 (nchg2: nr of real and model charges)
        bufR, bufI = int3c()

        charge = cell.atom_charges()
        nchg   = len(charge)
        charge = np.append(charge, -charge)  # (charge-of-nuccell, charge-of-fakenuc)
        nchg2  = len(charge)
        # mat is nkpts x natm x naopair
        # check for component not necessary anymore? 
        # sum over halves of z, chrg and -chrg ints 
        if is_zero(kpts):
            mat_at1 = np.einsum('kxz,z->kzx', bufR, charge)
            mat_at  = mat_at1[:,nchg:,:] + mat_at1[:,:nchg,:] 
        else:
            mat_at1 = (np.einsum('kxz,z->kzx', bufR, charge) +
                      np.einsum('kxz,z->kzx', bufI, charge) * 1j)
            mat_at  = mat_at1[:,nchg:,:] + mat_at1[:,:nchg,:] 

        # vbar is the interaction between the background charge
        # and the compensating function.  0D, 1D, 2D do not have vbar.
        if cell.dimension == 3 and intor in ('int3c2e', 'int3c2e_sph',
                                             'int3c2e_cart'):
            charge = -cell.atom_charges()

            #nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
            nucbar = np.asarray([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
            #nucbar *= np.pi/cell.vol
            nucbar *= np.pi/cell.vol

            # nkpts x nao x nao (ravel -> nao x nao)
            ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
            # mat is nkpts x natm x naopair
            # inner loop over chrg i: mat[k,i,:] nucbar[i] reshape ovlp to naopair
            for k in range(nkpts):
                if aosym == 's1':
                    for i in range(nchg):
                        mat_at[k,i,:] -= nucbar[i] * ovlp[k].reshape(nao_pair) 
                else:
                    for i in range(nchg):
                        mat_at[k,i,:] -= nucbar[i] * lib.pack_tril(ovlp[k])
        return mat_at

    def _get_nuc(self, mesh=None, with_pseudo=False):
        ''' Vnuc '''
        from pyscf.pbc.df.gdf_builder import _guess_eta
        log = logger.Logger(self.stdout, self.verbose)
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        aosym = 's2'
        # nao_pairs for i<=j upper triangular fx, incl diagonal
        nao_pair = nao * (nao+1) // 2

        kpt_allow = np.zeros(3)
        eta, mesh, ke_cutoff = _guess_eta(cell, kpts, mesh)

        # Initialize self.supmol
        if self.rs_cell is None:
            self.build()
        self.supmol = supmol = _strip_basis(self.supmol, eta)

        modchg_cell = _compensate_nuccell(cell, eta)
        # vj is nkpts x natm x nao_pairs
        vj_at = self._int_nuc_vloc(modchg_cell, with_pseudo=with_pseudo,
                                supmol=supmol)

        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        coulG = pyscf_pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv) * kws
        # ngrid x natm
        aoaux = ft_ao.ft_ao(modchg_cell, Gv)
        charges = cell.atom_charges()
        # ngrid x natm
        vG1 = np.einsum('i,xi->xi', -charges, aoaux) 
        vG_at = np.einsum('x,xi->xi', coulG, vG1)

        supmol_ft = ft_ao._ExtendedMole.from_cell(self.rs_cell, self.bvk_kmesh, verbose=log)
        supmol_ft = supmol_ft.strip_basis()
        ft_kern = supmol_ft.gen_ft_kernel(aosym, return_complex=False, verbose=log)

        Gv, Gvbase, kws = modchg_cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
        ngrids = Gv.shape[0]
        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
        Gblksize = max(16, int(max_memory*1e6/16/nao_pair/nkpts))
        Gblksize = min(Gblksize, ngrids, 200000)
        vG_atR = vG_at.real
        vG_atI = vG_at.imag
        log.debug1('max_memory = %s  Gblksize = %s  ngrids = %s',
                   max_memory, Gblksize, ngrids)

        buf = np.empty((2, nkpts, Gblksize, nao_pair))
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            # shape of Gpq (2, nkpts, nGv, nao_pair), 2 as in R, I
            # Gpq for each ao? check this
            # for each k, column is ao*ao value on a grid grid=nGv
            Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt_allow, kpts, out=buf)
            # for each k, separates R and I into 2 matrices (ao values on the grid) 
            # ngrid x natm
            for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
                # rho_ij(G) nuc(-G) / G^2
                # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
                # contract potential on grid points with value of the ao on that grid point (column in Gpq is ao*ao value on a grid)
                # x is ao pair index (maps to specific ij pair) in triangular matrix
                # logically each vj[k] is a matrix
                # vj[k] choose matrix for k; ji,jx->ix where i is natm, j is gridpoint index
                vR_at  = np.einsum('ji,jx->ix', vG_atR[p0:p1], GpqR)
                vR_at += np.einsum('ji,jx->ix', vG_atI[p0:p1], GpqI)
                vj_at[k] += vR_at
                # if not gamma point
                if not is_zero(kpts[k]):
                    vI_at  = np.einsum('ji,jx->ix', vG_atR[p0:p1], GpqI)
                    vI_at += np.einsum('ji,jx->ix', vG_atI[p0:p1], GpqR)
                    vj_at[k] += vI_at * 1j

        # now there's a triangular matrix for each k (triangular of nao x nao is nao_pairs)
        # unpack here to nao x nao
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

    def get_nuc(self, mesh=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.

        Kwargs:
            mesh: custom mesh grids. By default mesh is determined by the
            function _guess_eta from module pbc.df.gdf_builder.
        '''
        t0 = (logger.process_clock(), logger.perf_counter())
        nuc_at = self._get_nuc(mesh, with_pseudo=False)
        logger.timer(self, 'get_nuc', *t0)
        return nuc_at

    def get_pp_loc_part1(self, mesh=None):
        return self._get_nuc(mesh, with_pseudo=True)

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
            #fake_cell = pp_int.fake_cell_vloc(cell, cn)
            if fake_cell.nbas > 0:
                # Make a list on which atoms the gaussians sit (for the current Ci coeff.)
                fakebas_atom_lst = []
                for i in range(fake_cell.nbas):
                    fakebas_atom_lst.append(fake_cell.bas_atom(i))
                fakebas_atom_ids = np.array(fakebas_atom_lst)
                #
                int3c = self.gen_int3c_kernel(intors[cn], 's2', comp=1, j_only=True,
                                              auxcell=fake_cell, supmol=supmol)
                # The int over V_{loc} can be transfered to the 3-center
                # integrals, in which the aux. basis is given by the fake cell.
                ##### v is (check) (nkpts, naopairs, naux)
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
               warnings.warn('cell.pseudo was specified but its elements %s '
                             'were not found in the system (pp_part2).', cell._pseudo.keys())
            # list of zeros, length nkpts returned when no pp found on atoms
            vpploc_at = [0] * nkpts
        else:
            # my old code: reshape natm, nkpts, -1 or similar, here seems to not be needed
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
                # Used to put into the right spot in ppnl[nkpts, NATM, nao, nao]
                # l is the angular mom. qnr associated with given basis
                l = fakecell.bas_angular(ib)
                atm_id_hl = fakecell.bas_atom(ib)
                # orb magn nr 2L+1
                nd = 2 * l + 1
                # dim of the hl coeff. array
                hl_dim = hl.shape[0]
                ilp = np.ndarray((hl_dim,nd,nao), dtype=np.complex128, buffer=buf)
                for i in range(hl_dim):
                    # p0 takes care that the right m,l sph.harm are taken in projectors?
                    p0 = offset[i]
                    ilp[i] = ppnl_half[i][k][p0:p0+nd]
                    offset[i] = p0 + nd
                ppnl[k,atm_id_hl] += np.einsum('ilp,ij,jlq->pq', ilp.conj(), hl, ilp)
        
        if abs(kpts_lst).sum() < 1e-9:  # gamma_point:
            ppnl = ppnl.real
        return ppnl

    def get_pp(self, mesh=None):
        '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.

    
        Kwargs:
            mesh: custom mesh grids. By default mesh is determined by the
            function _guess_eta from module pbc.df.gdf_builder.
        '''
        vloc1 = self.get_pp_loc_part1(mesh)
        vloc2 = self.get_pp_loc_part2()
        vpp = get_pp_nl_atomic(self.cell, self.kpts)
        vpp2 = self.get_pp_nl()
        nkpts = len(self.kpts)
        vpp_tot = np.copy(vpp)
        for k in range(nkpts):
            vpp_tot[k] += vloc1[k] + vloc2[k]
        return vpp_tot, vloc1+vloc2, vpp2


#====================FFTDF====================#
def get_nuc_fftdf(mydf, kpts=None):
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
        vne_at = vne_at[0]
    return np.asarray(vne_at)

def get_pp_fftdf(mydf, kpts=None):
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

    vpp_at = np.zeros((nkpts, natm, nao, nao))
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
                print('ia, symb, l, proj', ia, symb, l, np.shape(proj) )
                print('rl, nl, hl', rl, nl, hl)
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
                print('p1 > 0')
                # n is nr of 2e aos in pp
                # SPG_lmi: n x nPWgrid 
                SPG_lmi = buf[:p1]
                print('!! SPG_lmi', np.shape(SPG_lmi))
                # SI: natm x nPWgrid
                SPG_lmi *= SI[ia].conj()
                # SPG_lm_aoGs: n x nao 
                SPG_lm_aoGs = lib.zdot(SPG_lmi, aokG)
                print('!! SPG_lm_aoGs', np.shape(SPG_lm_aoGs))
                p1 = 0
                # loop over shells with l>0, get the coeff hl and ints
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = np.asarray(hl)
                        print('hl', np.shape(hl), hl)
                        # SPG_lm_aoG, tmp: n_hl_dim x n_ml x nao
                        # n_hl_dim: nr of type of shells (s, p, d) 
                        # indices: j is n_hl_dim 
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        print('SPG_lm_aoG', np.shape(SPG_lm_aoG) )
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
        else:
            vpp_tot_at[k] = vpp_at[k] + vppnl_at[k]

    if kpts is None or np.shape(kpts) == (3,):
        vpp_tot_at = vpp_tot_at[0]
        vpp_at = vpp_at[0]
        vppnl_at = vppnl_at[0]
    return vpp_tot_at, vpp_at, vppnl_at
#    for k, kpt in enumerate(kpts_lst):
#        vppnl_at = vppnl_by_k(kpt)
#        if gamma_point(kpt):
#            vpp_at[k] = vpp_at[k].real + vppnl_at.real
#        else:
#            vpp_at[k] += vppnl_at
#
#    if kpts is None or np.shape(kpts) == (3,):
#        vpp_at = vpp_at[0]
#    return vpp_at


