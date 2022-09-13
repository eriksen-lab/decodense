#!/usr/bin/env python

import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto, scf, dft
from pyscf import gto as mgto
from pyscf import scf as mscf
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
    for i in ['']:#, 'fb', 'pm', 'ibo-2', 'ibo-4']:
        for j in ['mulliken']:#, 'iao']:
            decomp = decodense.DecompCls(prop='energy', part='atoms', loc=i, pop=j)
            res = decodense.main(cell, decomp, mf)
            print('Decodense res for cell, loc: {}, pop: {}'.format(i,j))
            for k, v in res.items():
                print(k, v)
            print()
            print('E_hf_pyscf - E_hf_dec = ', ehf - (np.sum(res['kin']) + np.sum(res['coul']) + np.sum(res['exch']) + np.sum(res['nuc_att_glob']) + np.sum(res['nuc_att_loc']) + np.sum(res['struct']) + np.sum(res['xc']) ) )
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
            if mol.pseudo:
                sub_nuc, sub_nuc_vloc1, sub_nuc_vloc2, sub_nuc_vpp = pbctools.get_pp_atomic(mydf, kpts=np.zeros(3))
            else:
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


def _vk_dft(mol: Union[gto.Cell, mgto.Mole], mf: dft.rks.KohnShamDFT, \
            xc_func: str, rdm1: np.ndarray, vk: np.ndarray) -> np.ndarray:
        """
        this function returns the appropriate dft exchange operator
        """
        # range-separated and exact exchange parameters
        ks_omega, ks_alpha, ks_hyb = mf._numint.rsh_and_hybrid_coeff(xc_func)
        print()
        print('ks_hyb', ks_hyb)
        print()
        # scale amount of exact exchange
        vk *= ks_hyb
        # range separated coulomb operator
        if abs(ks_omega) > 1e-10:
            vk_lr = mf.get_k(mol, rdm1, omega=ks_omega)
            vk_lr *= (ks_alpha - ks_hyb)
            vk += vk_lr
        return vk



##########################################
######### CELL OBJECT FOR TESTING ########
##########################################
#
# cell
# initialize a cell object
# the (1/4, 1/4, 1/4) basis atoms are: #2, i think?
cell = gto.Cell()
cell.atom = '''
 C   3.79049   2.18844   1.54746
 C   2.52699   1.45896   1.03164
'''
#cell.basis = 'sto3g'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
# .a is a matrix for lattice vectors.  Each row of .a is a primitive vector.
cell.a = np.eye(3)*2.52699
cell.a[1:,0] = 1.26350
cell.a[1,1], cell.a[2,2], cell.a[2,1] = 2.18844, 2.06328, 0.72948
cell.verbose = 3
cell.build()


mf = dft.RKS(cell).density_fit()
mf.xc='wB97M_V'
mf.nlc='VV10'
edft = mf.kernel()
print("DFT energy (per unit cell) = %.17g" % edft)
dm = mf.make_rdm1()
#mf = scf.RHF(cell).density_fit()
#ehf = mf.kernel()
#dm = mf.make_rdm1()
#print("HF energy (per unit cell) = %.17g" % ehf)

#print(dir(mf))


## init decomp object for cell
#decomp = decodense.DecompCls(**PARAMS)
##print('decomp', dir(decomp))
######
#res = decodense.main(cell, decomp, mf)

# J, K int
J_int, K_int = mf.get_jk()
#print('ek before hybrid scaling', np.einsum('ij,ij', K_int, dm))
K_int = _vk_dft(cell, mf, 'm06,m06', dm, K_int)
#print('ek after hybrid scaling', np.einsum('ij,ij', K_int, dm))
e_j = np.einsum('ij,ij', J_int, dm)
e_k = np.einsum('ij,ij', K_int, dm)
e_j *= 0.5
e_k *= -0.25

# kin, nuc atrraction 
# in decodense: glob>trace(sub_nuc_i, rdm1_tot), loc>trace(nuc,rdm1_atom_i)
kinetic, nuc, sub_nuc = _h_core(cell, mf)
e_kin = np.einsum('ij,ij', kinetic, dm)
#
nuc_att_glob = np.einsum('ij,ij', nuc, dm)
#nuc_att_glob *= .5
nuc_att_loc = np.einsum('xij,ij->x', sub_nuc, dm)
#nuc_att_loc *= .5
print('CELL_NUC_ATT_ATT', nuc_att_glob)
print('CELL_NUC_ATT_ATOMIC ', nuc_att_loc)
print('Their difference ', nuc_att_glob - np.einsum('z->', nuc_att_loc) )

##########################################
##########################################
## printing, debugging, etc.
#nuc_att_ints, nuc_att_ints_atomic = mf.get_nuc_att()
#cell_nuc_att = np.einsum('ij,ji', nuc_att_ints, dm)
#cell_nuc_att_atomic = np.einsum('zij,ji->z', nuc_att_ints_atomic, dm)
#print('cell_nuc_att_atomic ints ', np.shape(nuc_att_ints_atomic) )
#print('CELL_NUC_ATT ', cell_nuc_att)
#print('CELL_NUC_ATT_ATOMIC ', cell_nuc_att_atomic)
#print('Their difference ', cell_nuc_att - np.einsum('z->', cell_nuc_att_atomic) )
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
###### other terms
#####geteff = mf.get_veff()
#####getj, getk = mf.get_jk()
#####getk = _vk_dft(cell, mf, 'm06,m06', dm, getk)
#####vxc = geteff - getj + (.5*getk) 
#####e_getvff = np.einsum('ij,ij', geteff, dm) 
#####e_vxc = np.einsum('ij,ij', vxc, dm) 
#####print('e_coul as trace of  0.5J and D matrices (cell): ', e_j)
#####print('e_exch as trace of -0.25K and D matrices (cell): ', e_k)
#####print('e_veff, e_vxc as mf.get_veff', e_getvff, e_vxc)
######
print('energy_elec: (e_kin+e_jk_xc+e_nucatt), e_jk_xc', mf.energy_elec())
print('e_xc = energy_elec_jk_xc - ej -ek', mf.energy_elec()[1] - (e_j + e_k) )
print('e_xc = energy_elec - ej - ek - e_kin -e_nucatt', mf.energy_elec()[0] - (e_j + e_k + nuc_att_glob + e_kin) )
#
#print()
#print('my pyscf func')
#veff_tot, vxc1, vj1, vk1 = mf.get_veff1()
#vefftot1 = vxc1 + vj1 - 0.5*vk1
#print('total veff, veff from arrays', np.einsum('ij,ij', veff_tot, dm), np.einsum('ij,ij', vefftot1, dm))
#print('vj1, vk1 from arrays', np.einsum('ij,ij', vj1, dm), np.einsum('ij,ij', vk1, dm))
#print('vxc from arrays', np.einsum('ij,ij', vxc1, dm))
print()
#print('nuc_att_glob as trace of (what would correspond to) sub_nuc and D: ', cell_nuc_att_atomic, np.einsum('z->', cell_nuc_att_atomic) )
#print('nuc_att_loc as trace of nuc and d's: ', 2*np.sum(nuc_att_loc) )
#print('nuc_att as trace of nuc from pyscf and D: ', cell_nuc_att )
#print('local for cell computed here:')
#print(nuc_att_loc)
#
#E_total_cell = e_kin + e_j + e_k + 2.*nuc_att_glob + np.sum(res['struct'])
#print('e_kin + e_nuc + e_jk + e_nuc_att_glob ', E_total_cell)
#E_total_cell = e_kin + e_j + e_k + cell_nuc_att + np.sum(res['struct'])
#E_total_cell = e_kin + e_j + e_k + cell_nuc_att + np.sum(e_struct)
E_total_cell = e_kin + e_j + e_k + .5*nuc_att_glob + .5*np.einsum('z->', nuc_att_loc) + np.sum(e_struct)
print('e_kin + e_jk + 0.5e_nuc_att_glob + 0.5nuc_att_loc + e_struct', E_total_cell)
print('mf.energy_tot()', mf.energy_tot())
print('PBC E_tot (here, -e_xc) - E_tot (pyscf) = ', E_total_cell - mf.energy_tot() )
#
print('TEST')
print('from hcore', np.einsum('ij,ij', mf.get_hcore(), dm))
#print('my kin+nuc_att ', e_kin + cell_nuc_att )
#print('difference hcore: ', np.einsum('ij,ij', mf.get_hcore(), dm) - (e_kin + cell_nuc_att) )
print('my kin+nuc_att ', e_kin + nuc_att_glob )
print('difference hcore: ', np.einsum('ij,ij', mf.get_hcore(), dm) - (e_kin + nuc_att_glob) )
print('e_struct ', e_struct)
print('e_nuc ', cell.energy_nuc() )
print('e_nuc - e_struct ',  cell.energy_nuc() - np.sum(e_struct) )
#print(dir(mf))
#
#print('e_nuc_att term test')
#vpp, vpp_atomic = mf.get_nuc_att()
#print('vpp shape', np.shape(vpp) )
#print('vpp_atomic shape', np.shape(vpp_atomic) )
#e_nuc_att_pp = np.einsum('ij,ij', vpp, dm)
#e_nuc_att_pp_atomic = np.einsum('zij,ij->z', vpp_atomic, dm)
#print('e_nuc_att_pp', e_nuc_att_pp )
#print('e_nuc_att_pp_atomic', np.sum(e_nuc_att_pp_atomic), e_nuc_att_pp_atomic )
#print('e_nuc_att_pp - e_nuc_att_pp_atomic', e_nuc_att_pp - np.einsum('z->', e_nuc_att_pp_atomic) )

check_decomp(cell, mf)

