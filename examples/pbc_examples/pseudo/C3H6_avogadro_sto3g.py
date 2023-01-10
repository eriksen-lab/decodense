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
L = 2.5002804443
L2 = 1
cell = gto.Cell()
cell.atom = '''
C  1.2722441715      2.1226883399      0.7117079804     
C  1.2498908961      0.8975060736     -0.1949367806     
C  0.0000000000      0.0000000000      0.0000000000     
H  0.0776028298     -0.7986900713     -0.7522495405     
H  0.0616735521     -0.5044411888      0.9728386733     
H  2.1992449970      2.6906543209      0.6014709503     
H  0.4793191219      2.8223730544      0.4375971452     
H  1.1585639040      1.8454394358      1.7640706871     
H  1.2508362063      1.2546985408     -1.2328696564     
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
for i in range(18):
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

