#!/usr/bin/env python

import numpy as np
import pandas as pd
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf.pbc import gto, scf
from pyscf.pbc.tools.pbc import super_cell, cell_plus_imgs
from typing import List, Tuple, Dict, Union, Any

import decodense
#import nucAttGlob


# decodense variables
PARAMS = {
    'prop': 'energy',
#    'basis': 'ccpvdz',
#    'xc': 'pbe0',
    'loc': 'ibo-2',
#    'loc': '',
    'pop': 'iao',
    'part': 'atoms'
}

def check_decomp(cell, mf):
    ''' test which params work for cell '''

    ehf = mf.energy_tot()
    nat = cell.natm
    res_all = []
    for i in ['', 'fb', 'pm', 'ibo-2', 'ibo-4']:
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
    


##########################################
######### CELL OBJECT FOR TESTING ########
##########################################
#
# cell
cell = gto.Cell()
cell.atom = '''
 H   0.81252   1.47613   2.81966
 H   1.18600   1.19690   0.22918
 F   0.11649   1.99653   3.20061
 F   1.88203   0.67651   0.61013
'''
cell.basis = 'sto3g'
cell.a = np.eye(3) * 2.78686
cell.a[1, 1] = 2.67303
cell.a[1, 0] = -0.78834
cell.a[2, 2] = 5.18096
cell.build()

mf = scf.RHF(cell).density_fit()
ehf = mf.kernel()
dm = mf.make_rdm1()
print("HF energy (per unit cell) = %.17g" % ehf)


## init decomp object for cell
#decomp = decodense.DecompCls(**PARAMS)
#res = decodense.main(cell, decomp, mf)

# specify replicated units (total nr=product)
#nmp = [1, 1, 1]   # in this case it's plus one in each dir, i.e. 27 cells in total
#supcell = cell_plus_imgs(cell, nmp)
nmp = [2, 2, 2]
# make super cell
supcell = super_cell(cell, nmp)
print(supcell.atom_charges(), len(supcell.atom_charges()) )
# run molecular HF, but using ints between periodic gaussians
sup_mf = scf.RHF(supcell).density_fit()
sup_ehf = sup_mf.kernel()
sup_dm = sup_mf.make_rdm1()
print()
print('supcell energy: ', sup_mf.energy_tot(), np.prod(nmp), np.prod(nmp)*ehf)
print('supcell energy per cell: ', sup_mf.energy_tot()/np.prod(nmp) )
print(supcell.atom_charges(), len(supcell.atom_charges()) )
print()
## init decomp object for cell
decomp = decodense.DecompCls(**PARAMS)
res = decodense.main(supcell, decomp, sup_mf)
# save decodense results, atoms as rows
print(decodense.results(supcell, 'title', dump_res = True, suffix = f'_HF_supcell_2', **res))
dframe = pd.read_csv('res_HF_supcell_2.csv')
pd.options.display.max_columns = len(dframe.columns)
print(dframe)


## J, K int
#J_int, K_int = mf.get_jk()
#J_int *= .5
#K_int *= -0.25
#e_j = np.einsum('ij,ij', J_int, dm)
#e_k = np.einsum('ij,ij', K_int, dm)
#
## kin, nuc atrraction 
## in decodense: glob>trace(sub_nuc_i, rdm1_tot), loc>trace(nuc,rdm1_atom_i)
#kinetic, nuc, sub_nuc = _h_core(cell, mf)
#e_kin = np.einsum('ij,ij', kinetic, dm)
##
#nuc_att_glob = np.einsum('ij,ij', nuc, dm)
#nuc_att_glob *= .5
#nuc_att_loc = np.einsum('xij,ij->x', sub_nuc, dm)
#nuc_att_loc *= .5

##########################################
######### MOL OBJECT FOR TESTING #########
##########################################
#
#
mol = mgto.Mole()
mol.atom = '''
 H                  0.00000000    0.00000000    0.00000000
 F                  0.00000000    0.00000000    0.94900000
'''
mol.basis = 'sto3g'
mol.build()
mol_E_nuc = mol.energy_nuc()
#

mol_mf = mscf.RHF(mol).density_fit()
mol_ehf = mol_mf.kernel()
mol_dm = mol_mf.make_rdm1()
print("HF energy (molecule) = %.17g" % mol_ehf)


# init decomp object for mol
decomp_mol = decodense.DecompCls(**PARAMS)
#print('decomp', dir(decomp))
#####
res_mol = decodense.main(mol, decomp_mol, mol_mf)

## J, K int
#mol_J_int, mol_K_int = mol_mf.get_jk()
#mol_J_int *= .5
#mol_K_int *= -0.25
#mol_e_j = np.einsum('ij,ij', mol_J_int, mol_dm)
#mol_e_k = np.einsum('ij,ij', mol_K_int, mol_dm)
#
## kin, nuc atrraction 
## in decodense: glob>trace(sub_nuc_i, rdm1_tot), loc>trace(nuc,rdm1_atom_i)
#mol_kinetic, mol_nuc, mol_sub_nuc = _h_core(mol)
#mol_e_kin = np.einsum('ij,ij', mol_kinetic, mol_dm)
##
#mol_nuc_att_glob = np.einsum('ij,ij', mol_nuc, mol_dm)
#mol_nuc_att_glob *= .5
##print('shape of glob, loc', np.shape(mol_nuc), np.shape(mol_sub_nuc),np.shape(mol_dm) )
#mol_nuc_att_loc = np.einsum('xij,ij->x', mol_sub_nuc, mol_dm)
#mol_nuc_att_loc *= .5

##########################################
##########################################
## printing, debugging, etc.
print()
print('Decodense res for cell')
for k, v in res.items():
    print(k, v)
print()
print('Decodense res for mol')
for k, v in res_mol.items():
    print(k, v)
print()
###

print('mol')
print('energy_tot', mol_mf.energy_tot())
print('energy_elec', mol_mf.energy_elec())
print('cell')
print('energy_tot', mf.energy_tot())
print('energy_elec', mf.energy_elec())
print('supercell e_hf dec and pyscf difference', sup_mf.energy_tot() - np.sum(res['struct']) - np.sum(res['el']) )
print()



#### from decodense
###print('MOLECULE')
###print('e_nuc from decodense', np.sum(res_mol['struct']) )
###print('e_kin from decodense (molecule): ', np.sum(res_mol['kin']) )
###print('e_kin as trace of T and D matrices (molecule): ', mol_e_kin )
###print('difference: ', mol_e_kin - np.sum(res_mol['kin']) )
###print('e_coul as trace of J and D matrices (molecule): ', mol_e_j )
###print('difference: ', mol_e_j - np.sum(res_mol['coul']) )
###print('e_exch as trace of K and D matrices (molecule): ', mol_e_k )
###print('difference: ', mol_e_k - np.sum(res_mol['exch']) )
####
###print('nuc_att_glob from decodense (molecule): ', np.sum(res_mol['nuc_att_glob']) )
###print('mol_nuc_att_glob as trace of sub_nuc and D: ', mol_nuc_att_glob )
###print('difference: ', mol_nuc_att_glob - np.sum(res_mol['nuc_att_glob']) )
###print('nuc_att_loc from decodense (molecule): ', np.sum(res_mol['nuc_att_loc']) )
###print('mol_nuc_att_loc as trace of nuc and D: ', np.sum(mol_nuc_att_loc) )
###print('local from decodense, computed here:')
###print(res_mol['nuc_att_loc'])
###print(mol_nuc_att_loc)
###print('difference: ', mol_nuc_att_loc - res_mol['nuc_att_loc'] )
###E_total_mol = mol_e_kin + mol_e_j + mol_e_k + 2.*mol_nuc_att_glob + np.sum(res_mol['struct'])
###print('Mol e_kin + e_nuc + e_jk + e_nuc_att_glob ', E_total_mol)
###print( E_total_mol - (np.sum(res_mol['struct']) + np.sum(res_mol['coul']) + np.sum(res_mol['kin']) + np.sum(res_mol['exch']) + np.sum(res_mol['nuc_att_glob']+res_mol['nuc_att_loc']) ) )
###print(mol_mf.energy_tot() - E_total_mol)
###print()
####
###print('CELL')
###print('e_nuc from decodense', np.sum(res['struct']) )
###print('e_kin as trace of T and D matrices (cell): ', e_kin) 
####
#### other terms
###print('e_coul as trace of J and D matrices (cell): ', e_j)
###print('e_exch as trace of K and D matrices (cell): ', e_k)
####
###print('nuc_att_glob as trace of (what would correspond to) sub_nuc and D: ', cell_nuc_att_atomic, np.einsum('z->', cell_nuc_att_atomic) )
####print('nuc_att_loc as trace of nuc and d's: ', 2*np.sum(nuc_att_loc) )
###print('nuc_att as trace of nuc from pyscf and D: ', cell_nuc_att )
####print('local for cell computed here:')
####print(nuc_att_loc)
####
####E_total_cell = e_kin + e_j + e_k + 2.*nuc_att_glob + np.sum(res['struct'])
####print('e_kin + e_nuc + e_jk + e_nuc_att_glob ', E_total_cell)
###E_total_cell = e_kin + e_j + e_k + cell_nuc_att + np.sum(res['struct'])
###print('e_kin + e_nuc + e_jk + e_nuc_att_glob ', E_total_cell)
###print('PBC E_tot (here) - E_tot (pyscf) = ', E_total_cell - mf.energy_tot() )
####
###print('from hcore', np.einsum('ij,ij', mf.get_hcore(), dm))
###
#check_decomp(cell, mf)

