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

#import pyscf.lib
import copy
import ctypes
from pyscf import gto
from pyscf import lib
from pyscf import __config__
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.tools import pbc as pyscf_pbctools
from pyscf.pbc import gto as pbc_gto  
from pyscf.pbc import scf as pbc_scf 
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import incore
#from pyscf.pbc.df.incore import _IntNucBuilder, _compensate_nuccell
from pyscf.pbc.df.incore import _Int3cBuilder, _compensate_nuccell, _fake_nuc, _strip_basis
from pyscf.pbc.gto import pseudo
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point

libpbc = lib.load_library('libpbc')

PRECISION = getattr(__config__, 'pbc_df_aft_estimate_eta_precision', 1e-8)

''' Nuclear repulsion term '''
# almost identical to ewald in cell.py
# TODO: only works for 3D cells, extend to lower dim.
# TODO: only implemented for nuc-nuc repulsion
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

''' Nuclear attraction term '''
# inspired by the analogues in pyscf/pbc/df/aft
# TODO: only implemented for nuc-nel attraction, all-el. calculations
def get_nuc_atomic(mydf, kpts=None):
    '''Get the bare periodic nuc-el AO matrix, with G=0 removed.
       for each atom in a cell.
       See Martin (12.16)-(12.21).
    '''
    # Pseudopotential is ignored when computing just the nuclear attraction
    with lib.temporary_env(mydf.cell, _pseudo={}):
        nuc = get_pp_loc_part1_atomic_v0(mydf, kpts)
    # when only gamma point, the n_k x natm x nao x nao tensor -> natm x nao x nao matrix 
    if kpts is None or np.shape(kpts) == (3,) or np.shape(kpts) == (1,3):
        nuc = nuc[0]
    return nuc


def get_pp_atomic(mydf, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    cell = mydf.cell
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    # vloc1 is electron-core part in calc. with pseudopotential
    vloc1 = get_pp_loc_part1_atomic_v0(mydf, kpts_lst)
    
    # vloc2 is the electron-pseudopotential part (local)
    vloc2 = get_pp_loc_part2_atomic(cell, kpts_lst)
    
    # vpp is the nonlocal part, i.e. core shells projected out (nonlocal) 
    vpp = get_pp_nl_atomic(cell, kpts_lst)
    
    # TODO see if to leave this out
    vpp_total = np.zeros(np.shape(vloc1))
    for k in range(nkpts):
        vpp_total[k] += vloc1[k] + vloc2[k] + vpp[k]

    # when only gamma point, the n_k x natm x nao x nao tensor -> natm x nao x nao matrix 
    if kpts is None or np.shape(kpts) == (3,) or np.shape(kpts) == (1,3):
        vpp_total = vpp_total[0]
        vloc1 = vloc1[0]
        vloc2 = vloc2[0]
        vpp = vpp[0]
    return vpp_total, vloc1, vloc2, vpp


def get_pp_loc_part1_atomic_v0(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    mesh = np.asarray(mydf.mesh)
    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    # nao_pair (size for saving e.g. upper triangular matrix as array, (i<=j,incl diagonal))
    nao_pair = nao * (nao+1) // 2
    charges = cell.atom_charges()

    kpt_allow = np.zeros(3)
    # TODO test + check if kws is always scalar
    if mydf.eta == 0:
        if cell.dimension > 0:
            ke_guess = estimate_ke_cutoff(cell, cell.precision)
            mesh_guess = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_guess)
            if np.any(mesh[:cell.dimension] < mesh_guess[:cell.dimension]*.8):
                logger.warn(mydf, 'mesh %s is not enough for AFTDF.get_nuc function '
                            'to get integral accuracy %g.\nRecommended mesh is %s.',
                            mesh, cell.precision, mesh_guess)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)

        # vpplocG is n_at x n_grid
        vpplocG = pseudo.pp_int.get_gth_vlocG_part1(cell, Gv)
        vpplocG = -np.einsum('ij,ij->ji', cell.get_SI(Gv), vpplocG)

        vpplocG *= kws
        vG = vpplocG
        # nao_pairs i<=j upper triangular fx, incl diagonal
        vj = np.zeros((nkpts,len(charges),nao_pair), dtype=np.complex128)

    else:
        if cell.dimension > 0:
            ke_guess = _estimate_ke_cutoff_for_eta(cell, mydf.eta, cell.precision)
            mesh_guess = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_guess)
            if np.any(mesh < mesh_guess*.8):
                logger.warn(mydf, 'mesh %s is not enough for AFTDF.get_nuc function '
                            'to get integral accuracy %g.\nRecommended mesh is %s.',
                            mesh, cell.precision, mesh_guess)
            mesh_min = np.min((mesh_guess, mesh), axis=0)
            if cell.dimension < 2 or cell.low_dim_ft_type == 'inf_vacuum':
                mesh[:cell.dimension] = mesh_min[:cell.dimension]
            else:
                mesh = mesh_min
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)

        nuccell = _compensate_nuccell_v0(mydf)
        # PP-loc part1 is handled by fakenuc in _int_nuc_vloc
        # TODO why lib.asarray? without also returned as array..
        vj = lib.asarray(_int_nuc_vloc_atomic_v0(mydf, nuccell, kpts_lst))

        coulG = tools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv) * kws
        aoaux = ft_ao.ft_ao(nuccell, Gv)
        vG1 = np.einsum('i,xi->xi', -charges, aoaux)        
        vG = np.einsum('x,xi->xi', coulG, vG1)        

    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for aoaoks, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts_lst,
                                       max_memory=max_memory, aosym='s2'):
        # aoaoks for each ao
        for k, aoao in enumerate(aoaoks):
            # rho_ij(G) nuc(-G) / G^2
            # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
            if gamma_point(kpts_lst[k]):
            # contract potential on grid points with value of the ao on that grid point (column in aoao is ao*ao value on a grid)
            # x is ao pair index (maps to specific ij pair) in triangular matrix
            # logically each vj[k] is a matrix
            # vj[k] choose matrix for k; ji,jx->ix where i is n_at, j is gridpoint index
                vj[k] += np.einsum('ji,jx->ix', vG[p0:p1].real, aoao.real)
                vj[k] += np.einsum('ji,jx->ix', vG[p0:p1].imag, aoao.imag)
            else:
                vj[k] += np.einsum('ji,jx->ix', vG[p0:p1].conj(), aoao)
    
    # now there's a triangular matrix for each k (triangular of nao x nao is nao_pairs)
    # unpack here to nao x nao
    vj_kpts = []
    for k, kpt in enumerate(kpts_lst):
        if gamma_point(kpt):
            vj_1atm_kpts = []
            for i in range(len(charges)):
                vj_1atm_kpts.append(lib.unpack_tril(vj[k,i,:].real.copy()))
            vj_kpts.append(vj_1atm_kpts)
        else:
            vj_1atm_kpts = []
            for i in range(len(charges)):
                vj_1atm_kpts.append(lib.unpack_tril(vj[k,i,:]))
            vj_kpts.append(vj_1atm_kpts)

    return np.asarray(vj_kpts)

def get_nuc_atomic_v2(mydf, kpts=None):
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    dfbuilder = _IntNucBuilder(mydf.cell, kpts_lst)
    vj_at = dfbuilder.get_nuc(mydf.mesh)
    if kpts is None or numpy.shape(kpts) == (3,):
        vj_at = vj_at[0]
    return vj_at

def get_pp_atomic_v2(mydf, kpts=None):
    # this is from aft/get_pp and df/incore/get_pp levels
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    dfbuilder = _IntNucBuilder(mydf.cell, kpts_lst)

    # rn returns nkpts x nao x nao
    #vloc1, vloc1_at = dfbuilder._get_nuc(mydf.mesh, with_pseudo=True)
    vloc1, vloc1_at = dfbuilder.get_pp_loc_part1(mydf.mesh)
    return vloc1[0], vloc1_at[0]
    #return vpp_total, vloc1, vloc2, vpp



def get_pp_loc_part2_atomic(cell, kpts=None):
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
    from pyscf.pbc.df import incore
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    natm = cell.natm
    nao = cell.nao_nr()
    # nao_pairs for i<=j upper triangular fx, incl diagonal
    nao_pair = nao * (nao+1) // 2

    intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
              'int3c1e_r4_origk', 'int3c1e_r6_origk')
    kptij_lst = np.hstack((kpts_lst,kpts_lst)).reshape(-1,2,3)

    # Loop over coefficients to generate: erf, C1, C2, C3, C4
    # each coeff.-gaussian put in its own fakecell
    # If cn = 0, the erf term is generated.  C_1,..,C_4 are generated with cn = 1..4
    # buf is a buffer array to gather all integrals into before unpacking
    buf = np.zeros((natm, nao_pair))
    for cn in range(1, 5):
        fakecell = pseudo.pp_int.fake_cell_vloc(cell, cn)
        if fakecell.nbas > 0:
            # Make a list on which atoms the gaussians sit (for the current Ci coeff.)
            fakebas_atom_lst = []
            for i in range(fakecell.nbas):
                fakebas_atom_lst.append(fakecell.bas_atom(i))
            fakebas_atom_ids = np.array(fakebas_atom_lst)
             
            # The int over V_{loc} can be transfered to the 3-center
            # integrals, in which the aux. basis is given by the fake cell.
            # v is (naopairs, naux)
            v = incore.aux_e2(cell, fakecell, intors[cn], aosym='s2', comp=1,
                              kptij_lst=kptij_lst)
            # Put the ints for this Ci coeff. in the right places in the 
            # buffer (i.e. assign to the right atom)
            v = np.einsum('ij->ji', v)
            buf[fakebas_atom_lst] += v
    
    # if fakecell.nbas are all < 0, buf consists of zeros and we check for elements in the system 
    all_zeros = not np.any(buf)
    if all_zeros:
        if any(cell.atom_symbol(ia) in cell._pseudo for ia in range(cell.natm)):
            pass
        else:
             warnings.warn('cell.pseudo was specified but its elements %s '
                             'were not found in the system (pp_part2).', cell._pseudo.keys())
        # list of zeros, length nkpts returned when no pp found on atoms
        vpploc = [0] * nkpts
    else:
        buf = buf.reshape(natm, nkpts,-1)
        # indices: k-kpoint, i-atom, x-aopair
        buf = np.einsum('ikx->kix', buf)
        vpploc = []
        # now have the triangular matrix for each k (triangular of nao x nao is n_aopairs)
        # unpack here to nao x nao for each atom
        for k, kpt in enumerate(kpts_lst):
            vpploc_1atm_kpts = []
            for i in range(natm):
                v_1atm_ints = lib.unpack_tril(buf[k,i,:])
                if abs(kpt).sum() < 1e-9:  # gamma_point:
                    v_1atm_ints = v_1atm_ints.real
                vpploc_1atm_kpts.append(v_1atm_ints)
            vpploc.append(vpploc_1atm_kpts)

    return np.asarray(vpploc)


def get_pp_nl_atomic(cell, kpts=None):
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
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    # Generate a fake cell for V_{nl}.gaussian functions p_i^l Y_{lm}. 
    fakecell, hl_blocks = pseudo.pp_int.fake_cell_vnl(cell)
    ppnl_half = pseudo.pp_int._int_vnl(cell, fakecell, hl_blocks, kpts_lst)
    #ppnl_half = _int_vnl(cell, fakecell, hl_blocks, kpts_lst)
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

    #if kpts is None or np.shape(kpts) == (3,):
    #    ppnl = ppnl[0]
    return ppnl


# local functions mainly copied from pyscf
def _estimate_ke_cutoff_for_eta(cell, eta, precision=PRECISION):
    '''Given eta, the lower bound of ke_cutoff to produce the required
    precision in AFTDF Coulomb integrals.
    '''
    # estimate ke_cutoff for interaction between GTO(eta) and point charge at
    # the same location so that
    # \sum_{k^2/2 > ke_cutoff} weight*4*pi/k^2 GTO(eta, k) < precision
    # \sum_{k^2/2 > ke_cutoff} weight*4*pi/k^2 GTO(eta, k)
    # ~ \int_kmax^infty 4*pi/k^2 GTO(eta,k) dk^3
    # = (4*pi)^2 *2*eta/kmax^{n-1} e^{-kmax^2/4eta} + ... < precision

    # The magic number 0.2 comes from AFTDF.__init__ and GDF.__init__
    eta = max(eta, 0.2)
    log_k0 = 3 + np.log(eta) / 2
    log_rest = np.log(precision / (32*np.pi**2*eta))
    # The interaction between two s-type density distributions should be
    # enough for the error estimation.  Put lmax here to increate Ecut for
    # slightly better accuracy
    lmax = np.max(cell._bas[:,gto.ANG_OF])
    Ecut = 2*eta * (log_k0*(lmax-1) - log_rest)
    Ecut = max(Ecut, .5)
    return Ecut

def _compensate_nuccell_v0(mydf):
    '''A cell of the compensated Gaussian charges for nucleus'''
    cell = mydf.cell
    nuccell = copy.copy(cell)
    half_sph_norm = .5/np.sqrt(np.pi)
    norm = half_sph_norm/gto.gaussian_int(2, mydf.eta)
    chg_env = [mydf.eta, norm]
    ptr_eta = cell._env.size
    ptr_norm = ptr_eta + 1
    chg_bas = [[ia, 0, 1, 1, 0, ptr_eta, ptr_norm, 0] for ia in range(cell.natm)]
    nuccell._atm = cell._atm
    nuccell._bas = np.asarray(chg_bas, dtype=np.int32)
    nuccell._env = np.hstack((cell._env, chg_env))
    return nuccell


# Since the real-space lattice-sum for nuclear attraction is not implemented,
# use the 3c2e code with steep gaussians to mimic nuclear density
def _fake_nuc_v0(cell):
    fakenuc = copy.copy(cell)
    fakenuc._atm = cell._atm.copy()
    fakenuc._atm[:,gto.PTR_COORD] = np.arange(gto.PTR_ENV_START,
                                                 gto.PTR_ENV_START+cell.natm*3,3)
    _bas = []
    _env = [0]*gto.PTR_ENV_START + [cell.atom_coords().ravel()]
    ptr = gto.PTR_ENV_START + cell.natm * 3
    half_sph_norm = .5/np.sqrt(np.pi)
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb in cell._pseudo:
            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]
            eta = .5 / rloc**2
        else:
            eta = 1e16
        norm = half_sph_norm/gto.gaussian_int(2, eta)
        _env.extend([eta, norm])
        _bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
        ptr += 2
    fakenuc._bas = np.asarray(_bas, dtype=np.int32)
    fakenuc._env = np.asarray(np.hstack(_env), dtype=np.double)
    fakenuc.rcut = cell.rcut
    return fakenuc


def _int_nuc_vloc_atomic_v0(mydf, nuccell, kpts, intor='int3c2e', aosym='s2', comp=1):
    '''Vnuc - Vloc'''
    cell = mydf.cell
    nkpts = len(kpts)

    # Use the 3c2e code with steep s gaussians to mimic nuclear density
    fakenuc = _fake_nuc_v0(cell)
    fakenuc._atm, fakenuc._bas, fakenuc._env = \
            gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                         fakenuc._atm, fakenuc._bas, fakenuc._env)

    kptij_lst = np.hstack((kpts,kpts)).reshape(-1,2,3)
    buf = incore.aux_e2(cell, fakenuc, intor, aosym=aosym, comp=comp,
                        kptij_lst=kptij_lst)

    charge = cell.atom_charges()
    nchg = len(charge)
    charge = np.append(charge, -charge)  # (charge-of-nuccell, charge-of-fakenuc)
    nao = cell.nao_nr()
    nchg2 = len(charge)
    if aosym == 's1':
        nao_pair = nao**2
    else:
        nao_pair = nao*(nao+1)//2
    if comp == 1:
        buf = buf.reshape(nkpts,nao_pair,nchg2)
        mat1 = np.einsum('kxz,z->kzx', buf, charge)
        mat = mat1[:,nchg:,:] + mat1[:,:nchg,:]
    else:
        buf = buf.reshape(nkpts,comp,nao_pair,nchg2)
        mat1 = np.einsum('kczx,z->kczx', buf, charge)
        mat = mat1[:,:,nchg:,:] + mat1[:,:,:nchg,:]

    # vbar is the interaction between the background charge
    # and the compensating function.  0D, 1D, 2D do not have vbar.
    if cell.dimension == 3 and intor in ('int3c2e', 'int3c2e_sph',
                                         'int3c2e_cart'):
        assert(comp == 1)
        charge = -cell.atom_charges()
        
        # TODO check dim, if datatype complex
        nucbar = np.zeros(len(charge), dtype=np.float64)
        nucbar = np.asarray([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
        nucbar *= np.pi/cell.vol

        ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
        for k in range(nkpts):
            if aosym == 's1':
                for i in range(len(charge)):
                    mat[k,i,:] -= nucbar[i] * ovlp[k].reshape(nao_pair)
            else:
                for i in range(len(charge)):
                    mat[k,i,:] -= nucbar[i] * lib.pack_tril(ovlp[k])
    return mat

#--------------------------------------------------------GDF PP new
#
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
        # TODO mat is nkpts x naopair -> nkpts x natm x naopair
        # check comp? sum over halves of z, chrg and -chrg ints 
        if is_zero(kpts):
            mat = np.einsum('k...z,z->k...', bufR, charge)
            mat_at1 = np.einsum('kxz,z->kzx', bufR, charge)
            mat_at  = mat_at1[:,nchg:,:] + mat_at1[:,:nchg,:] 
        else:
            mat = (np.einsum('k...z,z->k...', bufR, charge) +
                   np.einsum('k...z,z->k...', bufI, charge) * 1j)
            mat_at1 = (np.einsum('kxz,z->kzx', bufR, charge) +
                      np.einsum('kxz,z->kzx', bufI, charge) * 1j)
            mat_at  = mat_at1[:,nchg:,:] + mat_at1[:,:nchg,:] 

        # vbar is the interaction between the background charge
        # and the compensating function.  0D, 1D, 2D do not have vbar.
        if cell.dimension == 3 and intor in ('int3c2e', 'int3c2e_sph',
                                             'int3c2e_cart'):
            charge = -cell.atom_charges()

            nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
            nucbar1 = np.asarray([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
            nucbar *= np.pi/cell.vol
            nucbar1 *= np.pi/cell.vol

            # nkpts x nao x nao (ravel -> nao x nao)
            ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
            # TODO mat is nkpts x naopair -> nkpts x natm x naopair
            # inner loop over chrg i: mat[k,i,:] nucbar[i] reshape ovlp to naopair
            for k in range(nkpts):
                if aosym == 's1':
                    for i in range(nchg):
                        mat_at[k,i,:] -= nucbar1[i] * ovlp[k].reshape(nao_pair) 
                    mat[k] -= nucbar * ovlp[k].ravel()
                else:
                    for i in range(nchg):
                        mat_at[k,i,:] -= nucbar1[i] * lib.pack_tril(ovlp[k])
                    mat[k] -= nucbar * lib.pack_tril(ovlp[k])
        return mat, mat_at

    def _get_nuc(self, mesh=None, with_pseudo=False):
        print('my part1')
        ''' I think this is just Vnuc '''
        from pyscf.pbc.df.gdf_builder import _guess_eta
        log = logger.Logger(self.stdout, self.verbose)
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        nao = cell.nao_nr()
        aosym = 's2'
        nao_pair = nao * (nao+1) // 2

        kpt_allow = np.zeros(3)
        eta, mesh, ke_cutoff = _guess_eta(cell, kpts, mesh)

        # Initialize self.supmol
        if self.rs_cell is None:
            self.build()
        self.supmol = supmol = _strip_basis(self.supmol, eta)

        modchg_cell = _compensate_nuccell(cell, eta)
        # TODO vj is n_k x nao_pairs -> n_k x natm x nao_pairs
        # checked, summed over atm gives same
        vj, vj_at = self._int_nuc_vloc(modchg_cell, with_pseudo=with_pseudo,
                                supmol=supmol)

        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        coulG = pyscf_pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv) * kws
        # ngrid x natm
        aoaux = ft_ao.ft_ao(modchg_cell, Gv)
        charges = cell.atom_charges()
        # ngrid -> ngrid x natm TODO change einsum
        # checked, vG vG_at same up to 1e-18
        vG = np.einsum('i,xi->x', -charges, aoaux) * coulG
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
        vGR = vG.real
        vGI = vG.imag
        vG_atR = vG_at.real
        vG_atI = vG_at.imag
        log.debug1('max_memory = %s  Gblksize = %s  ngrids = %s',
                   max_memory, Gblksize, ngrids)

        buf = np.empty((2, nkpts, Gblksize, nao_pair))
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            # shape of Gpq (2, nkpts, nGv, nao_pair), 2 as in R, I
            # Gpq for each ao? TODO check this
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
                # vj[k] choose matrix for k; ji,jx->x where i is n_at, j is gridpoint index TODO fix einsum
                # vj[k] choose matrix for k; ji,jx->ix where i is natm, j is gridpoint index
                vR = np.einsum('k,kx->x', vGR[p0:p1], GpqR)
                vR+= np.einsum('k,kx->x', vGI[p0:p1], GpqI)
                vj[k] += vR
                vR_at  = np.einsum('ji,jx->ix', vG_atR[p0:p1], GpqR)
                vR_at += np.einsum('ji,jx->ix', vG_atI[p0:p1], GpqI)
                vj_at[k] += vR_at
                # if not gamma point
                if not is_zero(kpts[k]):
                    vI = np.einsum('k,kx->x', vGR[p0:p1], GpqI)
                    vI-= np.einsum('k,kx->x', vGI[p0:p1], GpqR)
                    vj[k] += vI * 1j
                    vI_at  = np.einsum('ji,jx->ix', vG_atR[p0:p1], GpqI)
                    vI_at += np.einsum('ji,jx->ix', vG_atI[p0:p1], GpqR)
                    vj_at[k] += vI_at * 1j

        # now there's a triangular matrix for each k (triangular of nao x nao is nao_pairs)
        # unpack here to nao x nao
        # TODO here unpack vj_1atm for each atm in a loop, then append
        # for both gamma and other points
        vj_kpts = []
        vj_kpts_at = []
        for k, kpt in enumerate(kpts):
            if is_zero(kpt):
                vj_1atm_kpts = []
                for i in range(len(charges)):
                    vj_1atm_kpts.append(lib.unpack_tril(vj_at[k,i,:].real))
                vj_kpts_at.append(vj_1atm_kpts)
                #
                vj_kpts.append(lib.unpack_tril(vj[k].real))
            else:
                vj_1atm_kpts = []
                for i in range(len(charges)):
                    vj_1atm_kpts.append(lib.unpack_tril(vj_at[k,i,:]))
                vj_kpts_at.append(vj_1atm_kpts)
                #
                vj_kpts.append(lib.unpack_tril(vj[k]))
        return np.asarray(vj_kpts), np.asarray(vj_kpts_at)

    def get_nuc(self, mesh=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.

        Kwargs:
            mesh: custom mesh grids. By default mesh is determined by the
            function _guess_eta from module pbc.df.gdf_builder.
        '''
        t0 = (logger.process_clock(), logger.perf_counter())
        nuc, nuc_at = self._get_nuc(mesh, with_pseudo=False)
        logger.timer(self, 'get_nuc', *t0)
        return nuc_at

    def get_pp_loc_part1(self, mesh=None):
        return self._get_nuc(mesh, with_pseudo=True)

    def get_pp_loc_part2(self):
        print('part2')
        if self.rs_cell is None:
            self.build()
        cell = self.cell
        supmol = self.supmol
        if supmol.nbas == supmol.bas_mask.size:  # supmol not stripped
            supmol = self.supmol.strip_basis(inplace=False)
        kpts = self.kpts
        nkpts = len(kpts)
        intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
                  'int3c1e_r4_origk', 'int3c1e_r6_origk')
        bufR = 0
        bufI = 0
        for cn in range(1, 5):
            fake_cell = pp_int.fake_cell_vloc(cell, cn)
            if fake_cell.nbas > 0:
                int3c = self.gen_int3c_kernel(intors[cn], 's2', comp=1, j_only=True,
                                              auxcell=fake_cell, supmol=supmol)
                vR, vI = int3c()
                bufR += np.einsum('...i->...', vR)
                if vI is not None:
                    bufI += np.einsum('...i->...', vI)

        if isinstance(bufR, int):
            if any(cell.atom_symbol(ia) in cell._pseudo for ia in range(cell.natm)):
                pass
            else:
                lib.logger.warn(cell, 'cell.pseudo was specified but its elements %s '
                                 'were not found in the system.', cell._pseudo.keys())
            vpploc = [0] * nkpts
        else:
            buf = (bufR + bufI * 1j).reshape(nkpts,-1)
            vpploc = []
            for k, kpt in enumerate(kpts):
                v = lib.unpack_tril(buf[k])
                if abs(kpt).sum() < 1e-9:  # gamma_point:
                    v = v.real
                vpploc.append(v)
        return vpploc

    def get_pp(self, mesh=None):
        '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.

        Kwargs:
            mesh: custom mesh grids. By default mesh is determined by the
            function _guess_eta from module pbc.df.gdf_builder.
        '''
        print('GET_PP')
        t0 = (logger.process_clock(), logger.perf_counter())
        vloc1 = self.get_pp_loc_part1(mesh)
        t1 = logger.timer_debug1(self, 'get_pp_loc_part1', *t0)
        vloc2 = self.get_pp_loc_part2()
        t1 = logger.timer_debug1(self, 'get_pp_loc_part2', *t1)
        vpp = pp_int.get_pp_nl(self.cell, self.kpts)
        nkpts = len(self.kpts)
        for k in range(nkpts):
            vpp[k] += vloc1[k] + vloc2[k]
        t1 = logger.timer_debug1(self, 'get_pp_nl', *t1)
        logger.timer(self, 'get_pp', *t0)
        return vpp

# TODO delete when imported
# Since the real-space lattice-sum for nuclear attraction is not implemented,
# use the 3c2e code with steep gaussians to mimic nuclear density
#def _fake_nuc(cell, with_pseudo=True):
#    fakenuc = copy.copy(cell)
#    fakenuc._atm = cell._atm.copy()
#    fakenuc._atm[:,gto.PTR_COORD] = np.arange(gto.PTR_ENV_START,
#                                              gto.PTR_ENV_START+cell.natm*3,3)
#    _bas = []
#    _env = [0]*gto.PTR_ENV_START + [cell.atom_coords().ravel()]
#    ptr = gto.PTR_ENV_START + cell.natm * 3
#    half_sph_norm = .5/np.sqrt(np.pi)
#    for ia in range(cell.natm):
#        symb = cell.atom_symbol(ia)
#        if with_pseudo and symb in cell._pseudo:
#            pp = cell._pseudo[symb]
#            rloc, nexp, cexp = pp[1:3+1]
#            eta = .5 / rloc**2
#        else:
#            eta = 1e16
#        norm = half_sph_norm/gto.gaussian_int(2, eta)
#        _env.extend([eta, norm])
#        _bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
#        ptr += 2
#    fakenuc._bas = np.asarray(_bas, dtype=np.int32)
#    fakenuc._env = np.asarray(np.hstack(_env), dtype=np.double)
#    fakenuc.rcut = cell.rcut
#    return fakenuc
#
#def _compensate_nuccell(cell, eta):
#    '''A cell of the compensated Gaussian charges for nucleus'''
#    modchg_cell = copy.copy(cell)
#    half_sph_norm = .5/np.sqrt(np.pi)
#    norm = half_sph_norm/gto.gaussian_int(2, eta)
#    chg_env = [eta, norm]
#    ptr_eta = cell._env.size
#    ptr_norm = ptr_eta + 1
#    chg_bas = [[ia, 0, 1, 1, 0, ptr_eta, ptr_norm, 0] for ia in range(cell.natm)]
#    modchg_cell._atm = cell._atm
#    modchg_cell._bas = np.asarray(chg_bas, dtype=np.int32)
#    modchg_cell._env = np.hstack((cell._env, chg_env))
#    return modchg_cell
#
#def _strip_basis(supmol, eta, cutoff=None, inplace=True):
#    rs_cell = supmol.rs_cell
#
#    supmol_exps = np.array([e.min() for e in supmol.bas_exps()])
#    supmol_bas_coords = supmol.atom_coords()[supmol._bas[:,gto.ATOM_OF]]
#
#    dim = rs_cell.dimension
#    if dim == 0:
#        bas_mask = np.ones(supmol.nbas, dtype=bool)
#    else:
#        if cutoff is None:
#            cutoff = supmol.precision
#        # estimation based on 3-center gaussian overlap integrals
#        # ejk = minimize (ai*aj*|ri-rj|^2+ai*eta*ri^2+aj*eta*rj^2)/(ai+aj+eta)
#        ajk = supmol_exps + eta
#        ejk = supmol_exps * eta / ajk
#        rb = np.linalg.norm(supmol_bas_coords[:,:dim], axis=1)
#        a = rs_cell.lattice_vectors()
#        dr = rb - np.linalg.norm(a[:dim])
#        dr[dr < 0] = 0
#        ovlp = ejk * dr**2
#        bas_mask = ovlp < -np.log(cutoff)
#
#    if not inplace:
#        supmol = copy.copy(supmol)
#    if bas_mask.size == supmol.bas_mask.size:
#        supmol.bas_mask = bas_mask.reshape(supmol.bas_mask.shape)
#    else:
#        supmol.bas_mask = supmol.bas_mask.copy()
#        supmol.bas_mask[supmol.bas_mask] = bas_mask
#    supmol._bas = supmol._bas[bas_mask]
#    supmol.sh_loc = supmol.bas_mask_to_sh_loc(rs_cell, supmol.bas_mask)
#    return supmol


#def _int_vnl(cell, fakecell, hl_blocks, kpts):
#    '''Vnuc - Vloc'''
#    rcut = max(cell.rcut, fakecell.rcut)
#    Ls = cell.get_lattice_Ls(rcut=rcut)
#    nimgs = len(Ls)
#    expkL = np.asarray(np.exp(1j*np.dot(kpts, Ls.T)), order='C')
#    nkpts = len(kpts)
#
#    fill = getattr(libpbc, 'PBCnr2c_fill_ks1')
#    intopt = lib.c_null_ptr()
#
#    def int_ket(_bas, intor):
#        if len(_bas) == 0:
#            return []
#        # Str for which int to get
#        intor = cell._add_suffix(intor)
#        # Supposedly:
#        # 1-electron ints from two cells like
#        # < \mu | intor | \nu >, \mu \in cell1, \nu \in cell2
#        # so between real & fakecell (basis f. in cell & sph harm 
#        # in fakecell to make the orbs ~orth to core shells 
#        # represented by pp. I.e. project out the parts of core orbs. 
#        atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
#                                     fakecell._atm, _bas, fakecell._env)
#        atm = np.asarray(atm, dtype=np.int32)
#        # bas : int32 ndarray, libcint integral function argument
#        bas = np.asarray(bas, dtype=np.int32)
#        env = np.asarray(env, dtype=np.double)
#        natm = len(atm)
#        # 2*natm in cell, fakecell 
#        # nbas: nr of shells
#        nbas = len(bas)
#        # The slice is nr of shells in cell/concatenated cell
#        # (for picking which overlap ints to compute, I think)
#        shls_slice = (cell.nbas, nbas, 0, cell.nbas)
#        ao_loc = gto.moleintor.make_loc(bas, intor)
#        ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
#        nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
#
#        out = np.empty((nkpts,ni,nj), dtype=np.complex128)
#        comp = 1
#
#        fintor = getattr(gto.moleintor.libcgto, intor)
#
#        drv = libpbc.PBCnr2c_drv
#        drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
#            ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(nimgs),
#            Ls.ctypes.data_as(ctypes.c_void_p),
#            expkL.ctypes.data_as(ctypes.c_void_p),
#            (ctypes.c_int*4)(*(shls_slice[:4])),
#            ao_loc.ctypes.data_as(ctypes.c_void_p), intopt, lib.c_null_ptr(),
#            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
#            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
#            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
#        print('out returned by _int_vnl_atomic/int_ket ', np.shape(out) )
#        return out
#    
#    # extract how many nl proj. coeff. are there for each atom in fakecell
#    hl_dims = np.asarray([len(hl) for hl in hl_blocks])
#    # _bas: [atom-id,angular-momentum,num-primitive-GTO,num-contracted-GTO,0,ptr-of-exps,
#    # each element reperesents one shell
#    # e.g. diam. prim.fakecell: two lists,  [at_id=0 or 1, ang.mom.=0, nr.primGTOs=1, num.contr.GTOs=1,
#    # 0, ptr-of-exp=6 or 8, ptr.contract.coeff=7 or 9, ..=0 ] 
#
#    # each element in tuple out is ... computed for one shell, l qnr
#    out = (int_ket(fakecell._bas[hl_dims>0], 'int1e_ovlp'),
#           int_ket(fakecell._bas[hl_dims>1], 'int1e_r2_origi'),
#           int_ket(fakecell._bas[hl_dims>2], 'int1e_r4_origi'))
#    return out

