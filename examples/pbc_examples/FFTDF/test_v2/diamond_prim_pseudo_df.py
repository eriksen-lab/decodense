#!/usr/bin/env python

import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto, scf
from pyscf import gto as mgto
from pyscf import scf as mscf
import decodense
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

def print_mesh(mesh):
    print("mesh = [%d, %d, %d]  (%d PWs)" % (*mesh, np.prod(mesh)))

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


mf = scf.RHF(cell).density_fit()
ehf = mf.kernel()
dm = mf.make_rdm1()
print("HF energy (per unit cell) = %.17g" % ehf)


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
kinetic, nuc, subnuc = _h_core(cell, mf)
sub_nuc, sub_nuc_loc, sub_nuc_nl = subnuc
e_kin = np.einsum('ij,ij', kinetic, dm)
#
e_nuc_att_glob = np.einsum('ij,ij', nuc, dm)
# 0.5 factor if doing e_nuc_att = 1/2glob + 1/2loc in decodense
#e_nuc_att_glob *= .5
e_nuc_att_loc = np.einsum('xij,ij->x', sub_nuc, dm)
#e_nuc_att_loc *= .5

e_struct = pbctools.ewald_e_nuc(cell)



E_total_cell = e_kin + e_j + e_k + np.sum(e_nuc_att_loc) + np.sum(e_struct)
##########################################
##########################################
## printing, debugging, etc.
vpp = mf.get_nuc_att()
e_nuc_att_pyscf = np.einsum('ij,ji', vpp, dm)
print()
print('PYSCF')
print('nuc_att pyscf', e_nuc_att_pyscf)
print('from hcore', np.einsum('ij,ij', mf.get_hcore(), dm))
print('energy_tot', mf.energy_tot())
print('energy_elec', mf.energy_elec())
print()
print('MINE')
print('e_kin as trace of T and D matrices (cell): ', e_kin) 
print('e_coul as trace of J and D matrices (cell): ', e_j)
print('e_exch as trace of K and D matrices (cell): ', e_k)
print('e_nuc_att_loc: ', np.sum(e_nuc_att_loc))
print('e_struct ', np.sum(e_struct))
print('e_nuc_att_glob + kin (hcore)', np.sum(e_nuc_att_loc) + e_kin)
print('Total: e_kin + e_nuc + e_jk + e_nuc_att_loc + e_struct', E_total_cell)
print()
print('PBC E_tot (here) - E_tot (pyscf) = ', E_total_cell - mf.energy_tot() )
#

print('')
print('TEST')
print('difference hcore: ', np.einsum('ij,ij', mf.get_hcore(), dm) - (e_kin + np.sum(e_nuc_att_loc)) )
print('e_nuc - e_struct ',  cell.energy_nuc() - np.sum(e_struct) )
#print(dir(mf))
print('')
#
print('Vpp parts test')
print('')
#vpp_fft_at = pbctools.get_pp_fftdf(mydf)
mydf = mf.with_df
print('df object: ', type(mydf) )
vpp_gdf_at, vpp_loc_at, vpp_nl_at = pbctools.get_pp_atomic_df(mydf)
vpp_gdf = np.einsum('zab->ab', vpp_gdf_at)
print('all close pyscf vpp and vpp_gdf?', np.allclose(vpp, vpp_gdf, atol=1e-08) )
#check_decomp(cell, mf)
