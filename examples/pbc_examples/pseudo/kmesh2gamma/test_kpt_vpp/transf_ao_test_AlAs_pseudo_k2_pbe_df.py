#!/usr/bin/env python

import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto, scf, dft
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf.pbc.tools.k2gamma import k2gamma
from pyscf.pbc import tools
from pyscf.pbc.dft import multigrid
from pyscf.pbc.tools.k2gamma import get_phase
from pyscf.pbc.tools.k2gamma import to_supercell_ao_integrals 
from pyscf.pbc.tools.k2gamma import to_supercell_mo_integrals 
#from k2gamma import k2gamma
import decodense
#import pbctools
from decodense import pbctools
from typing import List, Tuple, Dict, Union, Any

import time
import sys
import json
#from pyscf.pbc import gto


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
    print('ehf cell', ehf)
    nat = cell.natm
    res_all = []
    #for i in ['', 'fb', 'pm', 'ibo-2', 'ibo-4']:
    #for i in ['pm', 'ibo-2' ]:
    for i in [ 'ibo-2' ]:
        #for j in ['mulliken', 'iao']:
        for j in ['iao']:
            decomp = decodense.DecompCls(prop='energy', part='atoms', loc=i, pop=j)
            res = decodense.main(cell, decomp, mf)
            print('Decodense res for cell, loc: {}, pop: {}'.format(i,j))
            for k, v in res.items():
                print(k, v)
            print()
            ehf_dec_tot = np.sum(res['el']+res['struct'])
            print('ehf_dec_tot', ehf_dec_tot)
            print('E_hf_pyscf - E_hf_dec_tot = ', ehf - ehf_dec_tot )
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
 Al  0.000000  0.000000  0.000000
 As  6.081570  3.511190  2.482790
'''
#cell.basis = 'sto3g'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 4.05438
cell.a[1, 0], cell.a[2, 0] = 2.02719, 2.02719
cell.a[1, 1] = 3.51119
cell.a[2, 1] = 1.17040
cell.a[2, 2] = 3.31039
cell.exp_to_discard = 0.1
cell.build()
# cell

#2 k-points for each axis, 2^3=8 kpts in total
#kmesh = [2,2,2]  
kmesh = [2,2,2]  
#default: shifted Monkhorst-Pack mesh centered at Gamma-p.
#to get non-shifted: with_gamma_point=False
#to get centered at specific p.(units of lattice vectors): scaled_center=[0.,0.25,0.25]
#mesh: The numbers of grid points in the FFT-mesh in each direction
kpts = cell.make_kpts(kmesh)

kmf = dft.KRKS(cell, kpts).newton()
kmf.xc = 'pbe'
kmf.with_df = df.df.DF(cell, kpts)
edft = kmf.kernel()
kdm = kmf.make_rdm1()
print("DFT energy (per unit cell) = %.17g" % edft)
print('kmf._numint', kmf._numint, type(kmf._numint))
# TODO is rdm1 always hermitian?
#hermi = 1
#_, _, vxc = kmf._numint.nr_rks(cell, kmf.grids, kmf.xc, kdm, hermi, kpts, kpts_band)
# TODO give max mem? max_memory=2000 
_, exc, vxc = kmf._numint.nr_rks(cell, kmf.grids, kmf.xc, kdm, kpts=kpts, get_exc=True)
print('vxc, exc', np.shape(vxc), type(vxc), np.shape(exc), type(exc) )
data0 = np.load(f'AlAs_df_k2.npz')
vxc0 = data0['vxc']
print('vxc from numint and calc: same? ', np.allclose(vxc, vxc0) )
print('vxc - vxc0, max ', np.max(abs(vxc - vxc0)) )

# vxc+vj test
veff = kmf.get_veff()
vj, _ = kmf.get_jk(with_k=False)
print('veff from numint and calc: same? ', np.allclose(veff, vxc+vj) )
print('veff - veff_mine, max ', np.max(abs(veff - vxc - vj)) )
print('energy elec', kmf.energy_elec() )
print('vhf kdm', np.einsum('kij,kij->k', vxc+vj, kdm) )


# transform the kmf object to mf obj for a supercell
# get the supercell and its mo coeff., occupations, en 
mf_scf = k2gamma(kmf)
supcell = mf_scf.mol
mo_coeff, mo_occ, e_mo = mf_scf.mo_coeff, mf_scf.mo_occ, mf_scf.mo_energy
# 
# save npz data
name = sys.argv[0]
name = name[:-3]
np.savez(f'{name}.npz', mo_coeff=mo_coeff, mo_occ=mo_occ, e_mo=e_mo)
data = np.load(f'{name}.npz')
coeff, occ, en = data['mo_coeff'], data['mo_occ'], data['e_mo']
#print()
# make the mf obj match kmf obj
mf = dft.RKS(supcell)
mf.xc = 'pbe'
mf.with_df = df.df.DF(supcell)
mf.mo_coeff = coeff
mf.mo_energy = en
mf.mo_occ = occ
#print('grids', mf.grids, np.shape(mf.grids))
#
print(type(kmf) )
print(type(mf) )

# now get the supcell
# test if supcell inherited all the attributes
#print('supcell.basis', supcell.basis)
#print('supcell.pseudo', supcell.pseudo)
#print('supcell.a', supcell.a)
#print('supcell.atom', supcell.atom)
#
# dump cell obj to json str obj
supcell_json_str = gto.cell.dumps(supcell)
#print('supcell json', supcell_json_str)
#
# write json cell file and read it again
print('types: json supcell, saved and loaded')
print(type(supcell_json_str))
# write cell obj to json file
with open(f'{name}.json', "w") as outfile:
    json.dump(supcell_json_str, outfile)
# read json file
with open(f'{name}.json', 'r') as openfile:
    json_object = json.load(openfile)
    supcell2 = gto.cell.loads(json_object)

#start_vpp = time.process_time()
#vpp0 = kmf.get_nuc_att(kpt=kpts)
#print(f'CPU time when computing kmf vpp: ', time.process_time() - start_vpp)
#print('shape vpp from kmf: ', np.shape(vpp0) )
#print()
