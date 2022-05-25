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
from pyscf.pbc import gto as pbc_gto  
from pyscf.pbc import scf as pbc_scf 
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import incore
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
        nuc = get_pp_loc_part1_atomic(mydf, kpts)
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

    # vloc1 is electron-nucc part in calc. with pseudopotential
    vloc1 = get_pp_loc_part1_atomic(mydf, kpts_lst)
    
    # vloc2 is the electron-pseudopotential part (local)
    vloc2 = get_pp_loc_part2_atomic(cell, kpts_lst)
    
    # vpp is the nonlocal part, i.e. core shells projected out (nonlocal) 
    vpp = get_pp_nl_atomic(cell, kpts_lst)
    
    # TODO see if to leave this out
    # vpp_total = np.zeros(np.shape(vloc1))
    for k in range(nkpts):
        vpp[k] += vloc1[k] + vloc2[k]
        #vpp_total[k] += vloc1[k] + vloc2[k] + vpp[k]

    # never true
    if kpts is None or np.shape(kpts) == (3,):
        vpp = vpp[0]
    #return vpp_total, vloc1, vloc2, vpp
    return vpp


def get_pp_loc_part1_atomic(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    mesh = np.asarray(mydf.mesh)
    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    # nao_pairs (size for saving e.g. upper triangular matrix as array, (i<=j,incl diagonal))
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

        nuccell = _compensate_nuccell(mydf)
        # PP-loc part1 is handled by fakenuc in _int_nuc_vloc
        # TODO why lib.asarray? without also returned as array..
        vj = lib.asarray(_int_nuc_vloc_atomic(mydf, nuccell, kpts_lst))

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

    if kpts is None or np.shape(kpts) == (3,):
        # when only gamma point, the n_k x nao x nao tensor -> nao x nao matrix 
        vj_kpts = vj_kpts[0]
    return np.asarray(vj_kpts)


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
        fakecell = _fake_cell_vloc(cell, cn)
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

    # when only gamma point, the n_k x nao x nao tensor -> nao x nao matrix 
    if kpts is None or np.shape(kpts) == (3,):
        vpploc = vpploc[0]
    return vpploc


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
    fakecell, hl_blocks = _fake_cell_vnl(cell)
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

    if kpts is None or np.shape(kpts) == (3,):
        ppnl = ppnl[0]
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

def _compensate_nuccell(mydf):
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
def _fake_nuc(cell):
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


def _int_nuc_vloc_atomic(mydf, nuccell, kpts, intor='int3c2e', aosym='s2', comp=1):
    '''Vnuc - Vloc'''
    cell = mydf.cell
    nkpts = len(kpts)

    # Use the 3c2e code with steep s gaussians to mimic nuclear density
    fakenuc = _fake_nuc(cell)
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

def _fake_cell_vloc(cell, cn=0):
    '''Generate fake cell for V_{loc}.

    Each term of V_{loc} (erf, C_1, C_2, C_3, C_4) is a gaussian type
    function.  The integral over V_{loc} can be transfered to the 3-center
    integrals, in which the auxiliary basis is given by the fake cell.

    The kwarg cn indiciates which term to generate for the fake cell.
    If cn = 0, the erf term is generated.  C_1,..,C_4 are generated with cn = 1..4
    '''
    fake_env = [cell.atom_coords().ravel()]
    fake_atm = cell._atm.copy()
    fake_atm[:,gto.PTR_COORD] = np.arange(0, cell.natm*3, 3)
    ptr = cell.natm * 3
    fake_bas = []
    half_sph_norm = .5/np.sqrt(np.pi)
    for ia in range(cell.natm):
        if cell.atom_charge(ia) == 0:  # pass ghost atoms
            continue

        symb = cell.atom_symbol(ia)
        if cn == 0:
            if symb in cell._pseudo:
                pp = cell._pseudo[symb]
                rloc, nexp, cexp = pp[1:3+1]
                alpha = .5 / rloc**2
            else:
                alpha = 1e16
            norm = half_sph_norm / gto.gaussian_int(2, alpha)
            fake_env.append([alpha, norm])
            fake_bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
            ptr += 2
        elif symb in cell._pseudo:
            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]
            if cn <= nexp:
                alpha = .5 / rloc**2
                norm = cexp[cn-1]/rloc**(cn*2-2) / half_sph_norm
                fake_env.append([alpha, norm])
                fake_bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
                ptr += 2

    fakecell = copy.copy(cell)
    fakecell._atm = np.asarray(fake_atm, dtype=np.int32)
    fakecell._bas = np.asarray(fake_bas, dtype=np.int32)
    fakecell._env = np.asarray(np.hstack(fake_env), dtype=np.double)
    return fakecell

# sqrt(Gamma(l+1.5)/Gamma(l+2i+1.5))
_PLI_FAC = 1/np.sqrt(np.array((
    (1, 3.75 , 59.0625  ),  # l = 0,
    (1, 8.75 , 216.5625 ),  # l = 1,
    (1, 15.75, 563.0625 ),  # l = 2,
    (1, 24.75, 1206.5625),  # l = 3,
    (1, 35.75, 2279.0625),  # l = 4,
    (1, 48.75, 3936.5625),  # l = 5,
    (1, 63.75, 6359.0625),  # l = 6,
    (1, 80.75, 9750.5625))))# l = 7,

def _fake_cell_vnl(cell):
    '''Generate fake cell for V_{nl}.

    gaussian function p_i^l Y_{lm}
    '''
    # TODO in (nl, rl, hl) hl is the proj part hproj
    '''{ atom: ( (nelec_s, nele_p, nelec_d, ...),
                rloc, nexp, (cexp_1, cexp_2, ..., cexp_nexp),
                nproj_types,
                (r1, nproj1, ( (hproj1[1,1], hproj1[1,2], ..., hproj1[1,nproj1]),
                               (hproj1[2,1], hproj1[2,2], ..., hproj1[2,nproj1]),
                               ...
                               (hproj1[nproj1,1], hproj1[nproj1,2], ...        ) )),
                (r2, nproj2, ( (hproj2[1,1], hproj2[1,2], ..., hproj2[1,nproj1]),
                ... ) )
                )
        ... }]i
    '''
    fake_env = [cell.atom_coords().ravel()]
    fake_atm = cell._atm.copy()
    fake_atm[:,gto.PTR_COORD] = np.arange(0, cell.natm*3, 3)
    ptr = cell.natm * 3
    fake_bas = []
    hl_blocks = []
    for ia in range(cell.natm):
        if cell.atom_charge(ia) == 0:  # pass ghost atoms
            #print('in fake_cell_vnl: ghost atomed passed..')
            continue

        symb = cell.atom_symbol(ia)
        #print('in fake_cell_vnl: ia(#atm), atm symbol', ia, symb)
        if symb in cell._pseudo:
            #print('in fake_cell_vnl: symb was in cell._psuedo')
            pp = cell._pseudo[symb]
            #print('in fake_cell_vnl: pp from cell._pseudo', type(pp), np.shape(pp), pp)
            # nproj_types = pp[4]
            for l, (rl, nl, hl) in enumerate(pp[5:]):
                #print('in fake_cell_vnl: l, (rl, nl, hl) ', l, (rl, nl, hl) )
                if nl > 0:
                    alpha = .5 / rl**2
                    norm = gto.gto_norm(l, alpha)
                    fake_env.append([alpha, norm])
                    fake_bas.append([ia, l, 1, 1, 0, ptr, ptr+1, 0])

#
# Function p_i^l (PRB, 58, 3641 Eq 3) is (r^{2(i-1)})^2 square normalized to 1.
# But here the fake basis is square normalized to 1.  A factor ~ p_i^l / p_1^l
# is attached to h^l_ij (for i>1,j>1) so that (factor * fake-basis * r^{2(i-1)})
# is normalized to 1.  The factor is
#       r_l^{l+(4-1)/2} sqrt(Gamma(l+(4-1)/2))      sqrt(Gamma(l+3/2))
#     ------------------------------------------ = ----------------------------------
#      r_l^{l+(4i-1)/2} sqrt(Gamma(l+(4i-1)/2))     sqrt(Gamma(l+2i-1/2)) r_l^{2i-2}
#
                    fac = np.array([_PLI_FAC[l,i]/rl**(i*2) for i in range(nl)])
                    #print('in fake_cell_vnl: fac', fac)
                    hl = np.einsum('i,ij,j->ij', fac, np.asarray(hl), fac)
                    #print('in fake_cell_vnl: hl contracted with fac', hl)
                    hl_blocks.append(hl)
                    ptr += 2

    fakecell = copy.copy(cell)
    fakecell._atm = np.asarray(fake_atm, dtype=np.int32)
    fakecell._bas = np.asarray(fake_bas, dtype=np.int32)
    fakecell._env = np.asarray(np.hstack(fake_env), dtype=np.double)
    return fakecell, hl_blocks

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

