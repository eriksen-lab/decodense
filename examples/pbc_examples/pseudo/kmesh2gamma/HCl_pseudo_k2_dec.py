#!/usr/bin/env python

import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto, scf
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf.pbc.tools.k2gamma import k2gamma
from pyscf.pbc import tools
from pyscf.pbc.tools.k2gamma import get_phase
from pyscf.pbc.tools.k2gamma import to_supercell_mo_integrals 
#from k2gamma import k2gamma
import decodense
#import pbctools
from decodense import pbctools
from typing import List, Tuple, Dict, Union, Any


# decodense variables
PARAMS = {
    'prop': 'energy',
#    'basis': 'ccpvdz',
#    'xc': 'pbe0',
#    'loc': 'ibo-2',
    'loc': '',
#    'pop': 'iao',
    'part': 'atoms'
}

def check_decomp(cell, mf):
    ''' test which params work for cell '''

    ehf = mf.energy_tot()
    nat = cell.natm
    res_all = []
    #for i in ['', 'fb', 'pm', 'ibo-2', 'ibo-4']:
    for i in ['pm', 'ibo-2' ]:
        for j in ['mulliken', 'iao']:
            decomp = decodense.DecompCls(prop='energy', part='atoms', loc=i, pop=j)
            res = decodense.main(cell, decomp, mf)
            print('Decodense res for cell, loc: {}, pop: {}'.format(i,j))
            for k, v in res.items():
                print(k, v)
            print()
            print('E_hf_pyscf - E_hf_dec = ', ehf - (np.sum(res['kin']) + np.sum(res['coul']) + np.sum(res['exch']) + np.sum(res['nuc_att_glob']) + np.sum(res['nuc_att_loc']) + np.sum(res['struct'])) )
            print('---------------------------------')
            print()
            #res_all.append(res)
    return print('Done!')
    
def check_k2gamma_ovlps(cell, scell, phase, kmesh, kmf, mf):
    # 1. transform ao overlap ints and check if it's sensible
    print('Checking the k2gamma transformation of AO ints')
    NR, Nk = phase.shape
    nao = cell.nao
    # TODO note that if kpts not give only returns Gamma-p. ints
    s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    s = scell.pbc_intor('int1e_ovlp')
    s1 = np.einsum('Rk,kij,Sk->RiSj', phase, s_k, phase.conj())
    print('Difference between S for scell and S_k transformed to real space')
    print(abs(s-s1.reshape(s.shape)).max())
    s = scell.pbc_intor('int1e_ovlp').reshape(NR,nao,NR,nao)
    s1 = np.einsum('Rk,RiSj,Sk->kij', phase.conj(), s, phase)
    print(abs(s1-s_k).max())

    # 2. The following is to check whether the MO is correctly coverted:
    print("Supercell gamma MO in AO basis from conversion:")
    scell = tools.super_cell(cell, kmesh)
    # FIXME df here will prob be different than for RKS calc...
    mf_sc = scf.RHF(scell).density_fit()
    print('mf_sc type df')
    print(mf_sc.with_df)
    c_g_ao = mf.mo_coeff
    s = mf_sc.get_ovlp()
    mf_sc.run()
    sc_mo = mf_sc.mo_coeff
    nocc = scell.nelectron // 2
    print("Supercell gamma MO from direct calculation:")
    print(np.linalg.det(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc])))
    print(np.linalg.svd(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc]))[1])
    return


def _h_core(mol: Union[gto.Cell, mgto.Mole], mf=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the kin and nuc attraction ints
        """
        # kin ints
        if (isinstance(mol, gto.Cell) and isinstance(mf, scf.hf.RHF)):
            kin = mol.pbc_intor('int1e_kin')
            # individual atomic potentials
            mydf = mf.with_df
            if mol.pseudo:
                if isinstance(mydf, df.df.DF):
                    sub_nuc_tot, sub_nuc_vloc, sub_nuc_vnl = pbctools.get_pp_atomic_df(mydf, kpts=np.zeros(3))
                    sub_nuc = (sub_nuc_tot, sub_nuc_vloc, sub_nuc_vnl)
                    # total nuclear potential
                    nuc = np.sum(sub_nuc_tot, axis=0)
                elif isinstance(mydf, df.fft.FFTDF):
                    sub_nuc_tot, sub_nuc_vloc, sub_nuc_vnl = pbctools.get_pp_atomic_fftdf(mydf, kpts=np.zeros(3))
                    sub_nuc = (sub_nuc_tot, sub_nuc_vloc, sub_nuc_vnl)
                    # total nuclear potential
                    nuc = np.sum(sub_nuc_tot, axis=0)
                else:
                    warnings.warn('Decodense code for %s object is not implemented yet. ', mydf)
            else:
                if isinstance(mydf, df.df.DF):
                    sub_nuc = pbctools.get_nuc_atomic_df(mydf, kpts=np.zeros(3))
                    # total nuclear potential
                    nuc = np.sum(sub_nuc, axis=0)
                elif isinstance(mydf, df.fft.FFTDF):
                    sub_nuc = pbctools.get_nuc_atomic_fftdf(mydf, kpts=np.zeros(3))
                    # total nuclear potential
                    nuc = np.sum(sub_nuc, axis=0)
                else:
                    warnings.warn('Decodense code for %s object is not implemented yet. ', mydf)

        elif isinstance(mol, mgto.Mole): 
            kin = mol.intor_symmetric('int1e_kin')
            # coordinates and charges of nuclei
            coords = mol.atom_coords()
            charges = mol.atom_charges()
            # individual atomic potentials
            sub_nuc = np.zeros([mol.natm, mol.nao_nr(), mol.nao_nr()], dtype=np.float64)
            for k in range(mol.natm):
                with mol.with_rinv_origin(coords[k]):
                    sub_nuc[k] = -1. * mol.intor('int1e_rinv') * charges[k]
            # total nuclear potential
            nuc = np.sum(sub_nuc, axis=0)
        else:
            print('Wrong object passed to _h_core pbc')
        return kin, nuc, sub_nuc 


##########################################
######### CELL OBJECT FOR TESTING ########
##########################################
#
# mesh: The numbers of grid points in the FFT-mesh in each direction
#FFTDF uses plane waves (PWs) as the auxiliary basis, whose size is determined by FFTDF.mesh, which is set to Cell.mesh upon initialization. Cell.mesh is a 1d array-like object of three integer numbers, [nx, ny, nz], that defines the number of PWs (or the real-space grid points in the unit cell) in the x, y and z directions, respectively. 
#To use a PW basis of a different size, the user can either overwrite FFTDF.mesh directly or change it by specifying Cell.ke_cutoff. 
# set automatically based on value of precision (of integral accuracy, default 1e-8)
# ke_cutoff: Kinetic energy cutoff of the plane waves in FFT-DF
# rcut: Cutoff radius (in Bohr) of the lattice summation in the integral evaluation
# cell
cell = gto.Cell()
cell.atom = '''
 H   0.686524  1.000000  0.686524
 Cl  0.981476  1.000000  0.981476
'''
#cell.basis = 'sto3g'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 4.18274
cell.a[1, 0] = -0.23273
cell.a[1, 1] = 4.17626
cell.a[2, 0] = -1.92719
cell.a[2, 1] = -2.13335
cell.a[2, 2] = 3.03810
cell.build()

#2 k-points for each axis, 2^3=8 kpts in total
#kmesh = [2,2,2]  
kmesh = [2,2,2]  
#default: shifted Monkhorst-Pack mesh centered at Gamma-p.
#to get non-shifted: with_gamma_point=False
#to get centered at specific p.(units of lattice vectors): scaled_center=[0.,0.25,0.25]
#mesh: The numbers of grid points in the FFT-mesh in each direction
kpts = cell.make_kpts(kmesh)

# TODO maybe make it return the original df obj and not default?
kmf = scf.KRHF(cell, kpts).density_fit().newton()
print('kmf df type')
print(kmf.with_df)
ehf = kmf.kernel()
kdm = kmf.make_rdm1()
print("HF energy (per unit cell) = %.17g" % ehf)

# transform the kmf object to mf obj for a supercell
mf = k2gamma(kmf) 
print('k2gamma transf. mf df type', mf.with_df)
scell, phase = get_phase(cell, kpts)
mydf = df.df.DF(scell)
mf.with_df = mydf
print('mf type', mf.with_df)
dm = (mf.make_rdm1()).real
# check sanity
check_k2gamma_ovlps(cell, scell, phase, kmesh, kmf, mf)


# J, K int
print('Computing J, K ints')
J_int, K_int = mf.get_jk()
J_int *= .5
K_int *= -0.25
e_j = np.einsum('ij,ij', J_int, dm)
e_k = np.einsum('ij,ij', K_int, dm)

# kin, nuc atrraction 
# in decodense: glob>trace(sub_nuc_i, rdm1_tot), loc>trace(nuc,rdm1_atom_i)
print('Computing kinetic, nuc ints')
kinetic, nuc, subnuc = _h_core(scell, mf)
sub_nuc, sub_nuc_loc, sub_nuc_nl = subnuc
e_kin = np.einsum('ij,ij', kinetic, dm)
#
e_nuc_att_glob = np.einsum('ij,ij', nuc, dm)
#nuc_att_glob *= .5
e_nuc_att_loc = np.einsum('xij,ij->x', sub_nuc, dm)
#nuc_att_loc *= .5

e_struct = pbctools.ewald_e_nuc(cell)

E_total_cell = e_kin + e_j + e_k + np.sum(e_nuc_att_loc) + np.sum(e_struct)
##########################################
##########################################
## printing, debugging, etc.
print('')
print('TEST')
#print('mf hcore and dm', dm.dtype, dm) 
print('difference hcore: ', np.einsum('ij,ij', mf.get_hcore(), dm) - (e_kin + np.sum(e_nuc_att_loc)) )
print('e_nuc - e_struct ',  cell.energy_nuc() - np.sum(e_struct) )
print('')
print()
print('scell')
print('energy_tot', mf.energy_tot())
#print('energy_elec', mf.energy_elec())
print()

print(type(kmf) )
print(type(mf) )


check_decomp(scell, mf)
