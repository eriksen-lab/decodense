#!/usr/bin/env python

import numpy as np
from pyscf.pbc import df, dft
from pyscf.pbc import gto, scf
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf.pbc.tools.k2gamma import k2gamma
from pyscf.pbc import tools
from pyscf.pbc.tools.k2gamma import get_phase
from pyscf.pbc.tools.k2gamma import to_supercell_ao_integrals 
#from k2gamma import k2gamma
import decodense
#import pbctools
from decodense import pbctools
from typing import List, Tuple, Dict, Union, Any
import sys


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
    #for i in ['pm', 'ibo-2' ]:
    #    for j in ['mulliken', 'iao']:
    for i in [ 'ibo-2' ]:
        for j in [ 'iao']:
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
    
def k2gamma_supcell(kmf, xc=None, df_type=None):
    ''' Transform KRKS object to RKS object (gamma p.),
        return cor. supercell, too. Inherit df, xc.  '''

    # transform the kmf object to mf obj for a supercell
    # get the supercell and its mo coeff., occupations, en 
    mf_scf = k2gamma(kmf)
    supcell = mf_scf.mol
    mo_coeff, mo_occ, e_mo = mf_scf.mo_coeff, mf_scf.mo_occ, mf_scf.mo_energy

    return mo_coeff, mo_occ, e_mo, supcell

def make_rks_obj_supcell(mo_coeff, mo_occ, e_mo, supcell, xc=None, df_type=None):
    ''' Make a RKS object (gamma p.) from transformed KRKS data '''

    if df_type == 'GDF':
        mf = dft.RKS(supcell).density_fit().apply(mscf.addons.remove_linear_dep_)
        mf.xc = xc
        print('RKS df obj. of type: ', mf.with_df)
        # overwrite df object to GDF
        mf.with_df = df.df.DF(supcell)
        mf.with_df.auxbasis = "weigend"
        print('RKS df obj. overwritten, of type: ', mf.with_df)
    elif df_type == 'FFTDF':
        mf = dft.RKS(supcell).apply(mscf.addons.remove_linear_dep_)
        mf.xc = xc
        print('RKS df obj. of type: ', mf.with_df)
        # overwrite df object to GDF
        mf.with_df = df.FFTDF(supcell)
        print('RKS df obj. overwritten, of type: ', mf.with_df)
    else:
        print('%s DF object not supported', mf.with_df)
        sys.exit()
    # write mo coeff, occupations, en
    mf.mo_coeff = mo_coeff
    mf.mo_energy = e_mo
    mf.mo_occ = mo_occ
    mf.nlc = ''
    mf.initialize_grids()

    return mf



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
kmesh = [4,4,4]  
#default: shifted Monkhorst-Pack mesh centered at Gamma-p.
#to get non-shifted: with_gamma_point=False
#to get centered at specific p.(units of lattice vectors): scaled_center=[0.,0.25,0.25]
#mesh: The numbers of grid points in the FFT-mesh in each direction
#kpts = cell.make_kpts(kmesh)
kpts = cell.make_kpts(kmesh,
                      space_group_symmetry=True, 
                      time_reversal_symmetry=True)

kmf = dft.KRKS(cell, kpts).density_fit()
print('kmf df type')
print(kmf.with_df)
kmf.xc = 'pbe'
kmf = kmf.newton()
kmf.with_df.auxbasis = "weigend"
ehf = kmf.kernel()
print('mo coeff kmf', np.shape(kmf.mo_coeff) )
#kdm = kmf.make_rdm1()
print("DFF energy (per unit cell) = %.17g" % ehf)
sys.exit

# transform back to nonsymnm. kmf obj.
#kmf = kmf.to_khf()
#kmf.kernel(kmf.make_rdm1())

# transform the kmf object to mf obj for a supercell
mo_coeff, mo_occ, e_mo, supcell = k2gamma_supcell(kmf, xc='pbe', df_type='GDF')
print('mo_coeff shape transformed', np.shape(mo_coeff))
j_int, _ = kmf.get_jk(with_k=False)
print('vj shape', np.shape(j_int))
j_int = to_supercell_ao_integrals(cell, kpts.kpts, j_int)
print('vj shape transformed', np.shape(j_int))

vpp = kmf.with_df.get_pp(kpts.kpts)
print('vpp shape', np.shape(vpp))
vpp  = to_supercell_ao_integrals(cell, kpts.kpts, vpp)
print('vpp transformed shape', np.shape(vpp))

# make the dft obj
mf = make_rks_obj_supcell(mo_coeff, mo_occ, e_mo, supcell, xc='pbe', df_type='GDF')
mf.vj = j_int
mf.vpp = vpp

check_decomp(supcell, mf)
