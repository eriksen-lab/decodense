#!/usr/bin/env python

import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto, scf
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf.pbc.tools.k2gamma import k2gamma
from pyscf.pbc.tools.k2gamma import get_phase
#from k2gamma import k2gamma
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
L = 4
cell = gto.Cell()
cell.build(unit = 'B',
           a = [[L,0,0],[0,1,0],[0,0,1]],
           mesh = [10,20,20],
           atom = 'H 0 0 0; H 0 0 1.8',
           dimension=1,
#           basis='gth-dzvp',
#           pseudo='gth-hf-rev')
           basis='sto3g')

nk = [2,2,2]  # 2 k-points for each axis, 2^3=8 kpts in total
kpts = cell.make_kpts(nk)

kmf = scf.KRHF(cell, kpts).density_fit()
ehf = kmf.kernel()
kdm = kmf.make_rdm1()
print("HF energy (per unit cell) = %.17g" % ehf)
print('kdm', np.shape(kdm), kdm.dtype )

# transform the kmf object to mf obj for a supercell
mf = k2gamma(kmf) 
cell, phase = get_phase(cell, kpts)
dm = mf.make_rdm1()

## init decomp object for cell
#decomp = decodense.DecompCls(**PARAMS)
##print('decomp', dir(decomp))
######
#res = decodense.main(cell, decomp, mf)

# J, K int
J_int, K_int = mf.get_jk()
J_int *= .5
K_int *= -0.25
e_j = np.einsum('ij,ij', J_int, dm)
e_k = np.einsum('ij,ij', K_int, dm)

# kin, nuc atrraction 
# in decodense: glob>trace(sub_nuc_i, rdm1_tot), loc>trace(nuc,rdm1_atom_i)
kinetic, nuc, sub_nuc = _h_core(cell, mf)
e_kin = np.einsum('ij,ij', kinetic, dm)
#
nuc_att_glob = np.einsum('ij,ij', nuc, dm)
nuc_att_glob *= .5
nuc_att_loc = np.einsum('xij,ij->x', sub_nuc, dm)
nuc_att_loc *= .5

##########################################
##########################################
## printing, debugging, etc.
nuc_att_ints, nuc_att_ints_atomic = mf.get_nuc_att()
cell_nuc_att = np.einsum('ij,ji', nuc_att_ints, dm)
cell_nuc_att_atomic = np.einsum('zij,ji->z', nuc_att_ints_atomic, dm)
print('cell_nuc_att_atomic ints ', np.shape(nuc_att_ints_atomic) )
print('CELL_NUC_ATT ', cell_nuc_att)
print('CELL_NUC_ATT_ATOMIC ', cell_nuc_att_atomic)
print('Their difference ', cell_nuc_att - np.einsum('z->', cell_nuc_att_atomic) )
###
###
#######print results
#####print(dir(res))
#print()
#print('Decodense res for cell')
#for k, v in res.items():
#    print(k, v)
print()
print('cell')
print('energy_tot', mf.energy_tot())
print('energy_elec', mf.energy_elec())
print()



#
# the kinetic energy term for cell
print('CELL')
#print('e_nuc from decodense', np.sum(res['struct']) )
e_struct = pbctools.ewald_e_nuc(cell)
print('e_kin as trace of T and D matrices (cell): ', e_kin) 
#
# other terms
print('e_coul as trace of J and D matrices (cell): ', e_j)
print('e_exch as trace of K and D matrices (cell): ', e_k)
#
#print('nuc_att_glob as trace of (what would correspond to) sub_nuc and D: ', cell_nuc_att_atomic, np.einsum('z->', cell_nuc_att_atomic) )
#print('nuc_att_loc as trace of nuc and d's: ', 2*np.sum(nuc_att_loc) )
print('nuc_att as trace of nuc from pyscf and D: ', cell_nuc_att )
#print('local for cell computed here:')
#print(nuc_att_loc)
#
#E_total_cell = e_kin + e_j + e_k + 2.*nuc_att_glob + np.sum(res['struct'])
#print('e_kin + e_nuc + e_jk + e_nuc_att_glob ', E_total_cell)
#E_total_cell = e_kin + e_j + e_k + cell_nuc_att + np.sum(res['struct'])
E_total_cell = e_kin + e_j + e_k + cell_nuc_att + np.sum(e_struct)
print('e_kin + e_nuc + e_jk + e_nuc_att_glob + e_struct', E_total_cell)
print('PBC E_tot (here) - E_tot (pyscf) = ', E_total_cell - mf.energy_tot() )
#
print('TEST')
print('from hcore', np.einsum('ij,ij', mf.get_hcore(), dm))
print('my kin+nuc_att ', e_kin + cell_nuc_att )
print('difference hcore: ', np.einsum('ij,ij', mf.get_hcore(), dm) - (e_kin + cell_nuc_att) )
print('e_struct ', e_struct)
print('e_nuc ', cell.energy_nuc() )
print('e_nuc - e_struct ',  cell.energy_nuc() - np.sum(e_struct) )
#print(dir(mf))
#
print('e_nuc_att term test')
vpp, vpp_atomic = mf.get_nuc_att()
print('vpp shape', np.shape(vpp) )
print('vpp_atomic shape', np.shape(vpp_atomic) )
e_nuc_att_pp = np.einsum('ij,ij', vpp, dm)
e_nuc_att_pp_atomic = np.einsum('zij,ij->z', vpp_atomic, dm)
print('e_nuc_att_pp', e_nuc_att_pp )
print('e_nuc_att_pp_atomic', np.sum(e_nuc_att_pp_atomic), e_nuc_att_pp_atomic )
print('e_nuc_att_pp - e_nuc_att_pp_atomic', e_nuc_att_pp - np.einsum('z->', e_nuc_att_pp_atomic) )

check_decomp(cell, mf)

