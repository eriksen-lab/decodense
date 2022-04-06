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
from pyscf import gto
from pyscf import lib
from pyscf import __config__
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc import gto as pbc_gto  
from pyscf.pbc import scf as pbc_scf 
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import incore
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point

PRECISION = getattr(__config__, 'pbc_df_aft_estimate_eta_precision', 1e-8)

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
            vj_1at_kpts = []
            for i in range(len(charges)):
                vj_1at_kpts.append(lib.unpack_tril(vj[k,i,:].real.copy()))
            vj_kpts.append(vj_1at_kpts)
        else:
            vj_1at_kpts = []
            for i in range(len(charges)):
                vj_1at_kpts.append(lib.unpack_tril(vj[k,i,:]))
            vj_kpts.append(vj_1at_kpts)

    if kpts is None or np.shape(kpts) == (3,):
        # when only gamma point, the n_k x nao x nao tensor -> nao x nao matrix 
        vj_kpts = vj_kpts[0]
    return np.asarray(vj_kpts)


# local functions
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

