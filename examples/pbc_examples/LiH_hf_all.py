#!/usr/bin/env python

import numpy as np
from pyscf.pbc import gto, scf
from pyscf import gto as mgto
from pyscf import scf as mscf
import decodense
from typing import List, Tuple, Dict, Union, Any


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


def _h_core(mol: Union[gto.Cell, mgto.Mole]) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the kin and nuc attraction ints
        """
        # kin ints
        if isinstance(mol, gto.Cell):
            kin = mol.pbc_intor('int1e_kin')
        else: 
            kin = mol.intor_symmetric('int1e_kin')
        # coordinates and charges of nuclei
        coords = mol.atom_coords()
        charges = mol.atom_charges()
        # individual atomic potentials
        sub_nuc = np.zeros([mol.natm, mol.nao_nr(), mol.nao_nr()], dtype=np.float64)
        for k in range(mol.natm):
            with mol.with_rinv_origin(coords[k]):
                if isinstance(mol, gto.Cell):
                    sub_nuc[k] = -1. * mol.pbc_intor('int1e_rinv') * charges[k]
                else:
                    sub_nuc[k] = -1. * mol.intor('int1e_rinv') * charges[k]
        # total nuclear potential
        nuc = np.sum(sub_nuc, axis=0)
        return kin, nuc, sub_nuc 


##########################################
######### CELL OBJECT FOR TESTING ########
##########################################
#
# cell
#L = 4
#cell = gto.Cell()
#cell.atom = f'Li 0 0 0; H {L/2} {L/2} {L/2}'
#cell.a = (np.ones([3, 3]) - np.eye(3)) * L / 2
#cell.unit = 'A'
#cell.exp_to_discard = 0.1
#cell.cart = True
##cell.verbose=4
#cell.basis='sto3g'
#cell.build()
cell = gto.Cell()
cell.atom = '''
    Li   0.00000   0.00000   0.00000
    H    2.82855   1.63306   1.15475
'''
cell.a =  np.eye(3) * 2.82855
cell.a[1,0], cell.a[2,0] = 1.41428, 1.41428
cell.a[1,1] = 2.44960
cell.a[2,1] = 0.81653
cell.a[2,2] = 2.30950
cell.exp_to_discard = 0.1
#cell.verbose=4
cell.basis='sto3g'
cell.build()

# JK terms diverge with default density_fit
#mf = scf.RHF(cell).density_fit()
mf = scf.KRHF(cell).mix_density_fit()
ehf = mf.kernel()
dm = mf.make_rdm1()
print("HF energy (per unit cell) = %.17g" % ehf)


# init decomp object for cell
decomp = decodense.DecompCls(**PARAMS)
#print('decomp', dir(decomp))
#####
res = decodense.main(cell, decomp, mf)

# J, K int
J_int, K_int = mf.get_jk()
J_int *= .5
K_int *= -0.25
#print('size dm', np.shape(dm))
#print('size J', np.shape(J_int))
#e_j = np.einsum('ij,ij', J_int, dm)
#e_k = np.einsum('ij,ij', K_int, dm)
e_j = np.einsum('ij,ij', J_int[0], dm[0])
e_k = np.einsum('ij,ij', K_int[0], dm[0])

# kin, nuc atrraction 
# in decodense: glob>trace(sub_nuc_i, rdm1_tot), loc>trace(nuc,rdm1_atom_i)
kinetic, nuc, sub_nuc = _h_core(cell)
#print('shape kinetic', np.shape(kinetic), np.shape(nuc), np.shape(sub_nuc) )
#e_kin = np.einsum('ij,ij', kinetic, dm)
e_kin = np.einsum('ij,ij', kinetic, dm[0])
#
#nuc_att_glob = np.einsum('ij,ij', nuc, dm)
nuc_att_glob = np.einsum('ij,ij', nuc, dm[0])
nuc_att_glob *= .5
#nuc_att_loc = np.einsum('xij,ij->x', sub_nuc, dm)
nuc_att_loc = np.einsum('xij,ij->x', sub_nuc, dm[0])
nuc_att_loc *= .5

##########################################
######### MOL OBJECT FOR TESTING #########
##########################################
#
#
mol = mgto.Mole()
mol.atom = '''
 Li                 0.00000000    0.00000000    0.00000000
 H                  0.00000000    0.00000000    1.41427500
'''
 #H                  0.00000000    0.00000000    1.59490000
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
mol_J_int, mol_K_int = mol_mf.get_jk()
mol_J_int *= .5
mol_K_int *= -0.25
mol_e_j = np.einsum('ij,ij', mol_J_int, mol_dm)
mol_e_k = np.einsum('ij,ij', mol_K_int, mol_dm)

# kin, nuc atrraction 
# in decodense: glob>trace(sub_nuc_i, rdm1_tot), loc>trace(nuc,rdm1_atom_i)
mol_kinetic, mol_nuc, mol_sub_nuc = _h_core(mol)
mol_e_kin = np.einsum('ij,ij', mol_kinetic, mol_dm)
#
mol_nuc_att_glob = np.einsum('ij,ij', mol_nuc, mol_dm)
mol_nuc_att_glob *= .5
#print('shape of glob, loc', np.shape(mol_nuc), np.shape(mol_sub_nuc),np.shape(mol_dm) )
mol_nuc_att_loc = np.einsum('xij,ij->x', mol_sub_nuc, mol_dm)
mol_nuc_att_loc *= .5

##########################################
##########################################
## printing, debugging, etc.
nuc_att_ints, nuc_att_ints_atomic = mf.get_nuc_att()
print('nuc_att_ints shape', np.shape(nuc_att_ints), np.shape(nuc_att_ints_atomic) )
#cell_nuc_att = np.einsum('ij,ji', nuc_att_ints, dm)
#cell_nuc_att_atomic = np.einsum('zij,ji->z', nuc_att_ints_atomic, dm)
cell_nuc_att = np.einsum('ij,ji', nuc_att_ints, dm[0])
cell_nuc_att_atomic = np.einsum('zij,ji->z', nuc_att_ints_atomic, dm[0])
print('CELL_NUC_ATT ', cell_nuc_att)
print('CELL_NUC_ATT_ATOMIC ', cell_nuc_att_atomic)
print('Their difference ', cell_nuc_att - np.einsum('z->', cell_nuc_att_atomic) )


####print results
#print(dir(res))
print()
print('Decodense res for cell')
for k, v in res.items():
    print(k, v)
print()
print('Decodense res for mol')
for k, v in res_mol.items():
    print(k, v)
print()

print('E_nuc from atomwise calculation: ', np.sum(res['struct']) )
print('E_nuc from pyscf: ', cell.energy_nuc())
print('E_nuc from pyscf (molecule): ', mol_E_nuc)
print()

# mf print
#print('mf')
#print(dir(mol_mf))
#print('cell mf')
#print(dir(mf))
#
print('mol')
print('energy_tot', mol_mf.energy_tot())
# same as:
#print('mol el and struct', np.sum(res_mol['el']) + np.sum(res_mol['struct']) )
print('energy_elec', mol_mf.energy_elec())
# same as:
#print('mol el ', np.sum(res_mol['el']) )
#print('mol j and k ', np.sum(res_mol['coul']) + np.sum(res_mol['exch']) )
print('cell')
print('energy_tot', mf.energy_tot())
print('energy_elec', mf.energy_elec())
print()



#
# the kinetic energy term for molecule
# from decodense
print('MOLECULE')
print('e_nuc from decodense', np.sum(res_mol['struct']) )
print('e_kin from decodense (molecule): ', np.sum(res_mol['kin']) )
print('e_kin as trace of T and D matrices (molecule): ', mol_e_kin )
print('difference: ', mol_e_kin - np.sum(res_mol['kin']) )
print('e_coul as trace of J and D matrices (molecule): ', mol_e_j )
print('difference: ', mol_e_j - np.sum(res_mol['coul']) )
print('e_exch as trace of K and D matrices (molecule): ', mol_e_k )
print('difference: ', mol_e_k - np.sum(res_mol['exch']) )
#
print('nuc_att_glob from decodense (molecule): ', np.sum(res_mol['nuc_att_glob']) )
print('mol_nuc_att_glob as trace of sub_nuc and D: ', mol_nuc_att_glob )
print('difference: ', mol_nuc_att_glob - np.sum(res_mol['nuc_att_glob']) )
print('nuc_att_loc from decodense (molecule): ', np.sum(res_mol['nuc_att_loc']) )
print('mol_nuc_att_loc as trace of nuc and D: ', np.sum(mol_nuc_att_loc) )
print('local from decodense, computed here:')
print(res_mol['nuc_att_loc'])
print(mol_nuc_att_loc)
print('difference: ', mol_nuc_att_loc - res_mol['nuc_att_loc'] )
E_total_mol = mol_e_kin + mol_e_j + mol_e_k + 2.*mol_nuc_att_glob + np.sum(res_mol['struct'])
print('Mol e_kin + e_nuc + e_jk + e_nuc_att_glob ', E_total_mol)
print( E_total_mol - (np.sum(res_mol['struct']) + np.sum(res_mol['coul']) + np.sum(res_mol['kin']) + np.sum(res_mol['exch']) + np.sum(res_mol['nuc_att_glob']+res_mol['nuc_att_loc']) ) )
print(mol_mf.energy_tot() - E_total_mol)
print()
#
# the kinetic energy term for cell
print('CELL')
print('e_nuc from decodense', np.sum(res['struct']) )
print('e_kin as trace of T and D matrices (cell): ', e_kin) 
#
# other terms
print('e_coul as trace of J and D matrices (cell): ', e_j)
print('e_exch as trace of K and D matrices (cell): ', e_k)
#
print('nuc_att_glob as trace of (what would correspond to) sub_nuc and D: ', cell_nuc_att_atomic, np.einsum('z->', cell_nuc_att_atomic) )
#print('nuc_att_loc as trace of nuc and d's: ', 2*np.sum(nuc_att_loc) )
print('nuc_att as trace of nuc from pyscf and D: ', cell_nuc_att )
#print('local for cell computed here:')
#print(nuc_att_loc)
#
#E_total_cell = e_kin + e_j + e_k + 2.*nuc_att_glob + np.sum(res['struct'])
#print('e_kin + e_nuc + e_jk + e_nuc_att_glob ', E_total_cell)
E_total_cell = e_kin + e_j + e_k + cell_nuc_att + np.sum(res['struct'])
print('e_kin + e_nuc + e_jk + e_nuc_att_glob ', E_total_cell)
print('PBC E_tot (here) - E_tot (pyscf) = ', E_total_cell - mf.energy_tot() )
#
#print('from hcore', np.einsum('ij,ij', mf.get_hcore(), dm))
#print(dir(mf))

