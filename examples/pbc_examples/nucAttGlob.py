import copy
import numpy as np
from pyscf import gto
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import incore
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.gto import pseudo, estimate_ke_cutoff
# FIXME alt to logger for warnings
from pyscf.lib import logger
from pyscf import __config__

PRECISION = getattr(__config__, 'pbc_df_aft_estimate_eta_precision', 1e-8)

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
            ke_guess = estimate_ke_cutoff_for_eta(cell, mydf.eta, cell.precision)
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
        vj = lib.asarray(mydf._int_nuc_vloc_atomic(nuccell, kpts_lst))

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

def estimate_ke_cutoff_for_eta(cell, eta, precision=PRECISION):
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

# local functions
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

