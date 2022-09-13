#!/usr/bin/env python

import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto, scf
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf.pbc import tools
import decodense
import pbctools
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
B_to_AA = 0.529177249

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
    

def _h_core(mol: Union[gto.Cell, mgto.Mole], mf=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the kin and nuc attraction ints
        """
        # kin ints
        if (isinstance(mol, gto.Cell) and isinstance(mf, scf.hf.RHF)):
            kin = mol.pbc_intor('int1e_kin')
            # individual atomic potentials
            mydf = mf.with_df
            #mydf = df.FFTDF(mol)
            sub_nuc = pbctools.get_nuc_atomic(mydf, kpts=np.zeros(3)) 
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
        else:
            print('Wrong object passed to _h_core pbc')
        # total nuclear potential
        nuc = np.sum(sub_nuc, axis=0)
        return kin, nuc, sub_nuc 


##########################################
######### CELL OBJECT FOR TESTING ########
##########################################
#
# cell
L = 5.13151
L2 = 1
cell = gto.Cell()
cell.atom = '''
N   3.3573057889  4.2135928819  3.9285014595   
N   0.7915508066  0.0330209489  5.7610402611   
C   0.7916534368  1.1746563936  5.9760437189   
C   3.3574084191  3.5979510359  4.9136629148   
C   0.7917150149  2.6234979540  6.2111099344   
C   3.3574699972  2.7888549525  6.1383104235   
C   4.6404552501  3.0510749075  6.9618241765   
C   2.0747002678  3.0513103070  6.9620498765   
H   0.7916483053  3.1164158610  5.2279388610   
H   3.3574032876  1.7318581628  5.8344170061   
H   4.7247454328  4.1148665350  7.1799212422   
H   2.1589904504  2.4907011897  7.8920694324   
H   4.5561548044  2.4898459043  7.8916360597   
H   1.9903998221  4.1153693553  7.1795503917   
'''
cell.a = [[L,0,0],[0,L2,0],[0,0,L2]] 
cell.basis = 'sto3g'
cell.dimension = 1
cell.build()


#cell.build(unit = 'B',
#           a = [[L,0,0],[0,1,0],[0,0,1]],
#           mesh = [10,20,20],
#           atom = 'H 0 0 0; H 0 0 1.8',
#           dimension=1,
#           basis='sto3g')

print('cell coord Bohr', cell.atom_coords())

print('supcell')
supmol = tools.super_cell(cell, [2,1,1]).to_mol()
print('size', np.shape(supmol.atom_coords()) )
for i in range(28):
#for i in range(len(supmol.atom_coords()[0])):
    print(supmol.atom_coords()[i, :]*B_to_AA)

#mf = scf.RHF(cell).density_fit()
#ehf = mf.kernel()
#dm = mf.make_rdm1()
#print("HF energy (per unit cell) = %.17g" % ehf)
#print('coords', cell.atom_coords())



#print()
#print('cell')
#print('energy_tot', mf.energy_tot())
#print('energy_elec', mf.energy_elec())
#print()


check_decomp(cell, mf)

