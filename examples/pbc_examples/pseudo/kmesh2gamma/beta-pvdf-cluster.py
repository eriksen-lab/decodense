import importlib
import sys
import h5py
import numpy as np
import pandas as pd
from pyscf import scf as mol_scf
from pyscf.pbc import gto, scf
from pyscf.pbc import df, dft, scf, tools
from pyscf.pbc.dft import multigrid
from pyscf.pbc.tools.k2gamma import k2gamma
import decodense

# nr of images in 1D ring
nkpt = 2

# a function that loops over all decodense params
# the results are written to separate csv files for each
# combination of params (beta: jobbeta_pop_loc_nkpts.csv) 

def run_all_decomp(cell, mf, nkpts=1):
    ''' run decodense with all params implemented for cell obj'''

    ehf = mf.energy_tot()
    print('mf en, kmf en', ehf, edft)
    print('mf en ', cell.energy_nuc())
    print('mf en, kmf en elec', kmf.energy_elec(), mf.energy_elec())
    J, K = mf.get_jk()
    J *= 0.5
    K *= -0.25
    dm = mf.make_rdm1()
    e_j = np.einsum('ij,ij', J, dm)
    e_k = np.einsum('ij,ij', K, dm)
    print('mf en jk', e_j+e_k)

    nat = cell.natm
    res_all = []
    for i in ['pm']:
        for j in ['mulliken', 'iao']:
            decomp = decodense.DecompCls(prop='energy', part='atoms', loc=i, pop=j)
            res = decodense.main(cell, decomp, mf)
            print()
            print(decodense.results(cell, f'HF energy (localization procedure: {i}, population: {j})', dump_res = True, suffix = f'_beta_PVDF_k{nkpts}_{i}_{j}', **res))
            dframe = pd.read_csv(f'res_beta_PVDF_k{nkpts}_{i}_{j}.csv')
            pd.options.display.max_columns = len(dframe.columns)
            print()
            print('---------------------------------')
            ehf_dec = dframe['tot'].values
            ehf_dec_per_cell = np.sum(ehf_dec) / nkpts
            print('E_hf_dec per cell = ', ehf_dec_per_cell)
            print('E_hf_pyscf - E_hf_dec = ', ehf - np.sum(ehf_dec) )
            print('---------------------------------')
            print()
    return print('Done decomposing!')

def run_iao_decomp(cell, mf, nkpts=1):
    ''' run decodense with all params implemented for cell obj'''

    ehf = mf.energy_tot()
    nat = cell.natm
    res_all = []
    for i in ['ibo-2']:#'', 'fb', 'pm', 'ibo-2', 'ibo-4']:
        for j in ['iao']:#'mulliken', 'iao']:
            decomp = decodense.DecompCls(prop='energy', part='atoms', loc=i, pop=j)
            res = decodense.main(cell, decomp, mf)
            print()
            print(decodense.results(cell, f'HF energy (localization procedure: {i}, population: {j})', dump_res = True, suffix = f'_beta_PVDF_k{nkpts}_{i}_{j}', **res))
            dframe = pd.read_csv(f'res_beta_PVDF_k{nkpts}_{i}_{j}.csv')
            pd.options.display.max_columns = len(dframe.columns)
            print()
            print('---------------------------------')
            ehf_dec = dframe['tot'].values
            ehf_dec_per_cell = np.sum(ehf_dec) / nkpts
            print('E_hf_dec per cell = ', ehf_dec_per_cell)
            print('E_hf_pyscf - E_hf_dec = ', ehf - np.sum(ehf_dec) )
            print('---------------------------------')
            print()
    return print('Done decomposing!')

def run_eda_decomp(cell, mf, nkpts=1):
    ''' run decodense with eda for cell obj'''

    ehf = mf.energy_tot()
    nat = cell.natm
    res_all = []
    for i in ['']:
        for j in ['mulliken']:
            decomp = decodense.DecompCls(prop='energy', part='eda', loc=i, pop=j)
            res = decodense.main(cell, decomp, mf)
            print()
            print(decodense.results(cell, f'HF energy (localization procedure: {i}, population: {j})', dump_res = True, suffix = f'_beta_PVDF_k{nkpts}_eda_{i}_{j}', **res))
            dframe = pd.read_csv(f'res_beta_PVDF_k{nkpts}_eda_{i}_{j}.csv')
            pd.options.display.max_columns = len(dframe.columns)
            print()
            print('---------------------------------')
            ehf_dec = dframe['tot'].values
            ehf_dec_per_cell = np.sum(ehf_dec) / nkpts
            print('E_hf_dec per cell = ', ehf_dec_per_cell)
            print('E_hf_pyscf - E_hf_dec = ', ehf - np.sum(ehf_dec) )
            print('---------------------------------')
            print()
    return print('Done decomposing!')


# cell 1D
# lattice parameter for the periodicity axis (x)
L = 2.53619
# param for the other two axes, ignored by PySCF
L2 = 6
cell = gto.Cell()
cell.atom = '''
F 1.2680950000  4.9820185705  3.7343103562  
F 1.2680950000  2.7518014295  3.7343103562  
C 1.2680950000  3.8669100000  2.8564186956  
C 0.0000000000  3.8669100000  2.0435967139  
H 0.0000000000  4.7440721306  1.4088379422  
H 0.0000000000  2.9897478694  1.4088379422  
'''
cell.a = [[L,0,0],[0,L2,0],[0,0,L2]]
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-hf-rev'
cell.dimension = 1
#cell.verbose = 5
cell.build()


# make repetitions along x-axis
kmesh = [nkpt,1,1]
kpts = cell.make_kpts(kmesh)

# run KRKS calculation
#kmf = dft.KRKS(cell, kpts).density_fit().newton().apply(mol_scf.addons.remove_linear_dep_)
kmf = dft.KRKS(cell, kpts).density_fit()#.apply(mol_scf.addons.remove_linear_dep_)
kmf.xc = 'b3lyp'
#kmf.with_df = multigrid.MultiGridFFTDF(cell, kpts) #no exch!
edft = kmf.kernel()
print("DFT energy (per unit cell) = %.17g" % edft)

# transform the kmf object to mf obj for a supercell
# get the supercell and its mo coeff., occupations, en 
mf_scf = k2gamma(kmf)
supcell = mf_scf.mol
#mf_scf.with_df = df.df.DF(supcell)
mo_coeff, mo_occ, e_mo = mf_scf.mo_coeff, mf_scf.mo_occ, mf_scf.mo_energy
# make the mf obj match kmf obj
mf = dft.RKS(supcell).density_fit()
mf.xc = 'b3lyp'
# write mo coeff, occupations, en
mf.mo_coeff = mo_coeff
mf.mo_energy = e_mo
mf.mo_occ = mo_occ
print('df obj. of type: ', mf.with_df)
# overwrite df object to GDF
mydf = df.df.DF(supcell)
mf.with_df = mydf
print('df obj. of type: ', mf.with_df)
#dm = (mf.make_rdm1()).real
# check sanity
#check_k2gamma_ovlps(cell, supcell, phase, kmesh, kmf, mf, 'pbe')

# decompose the results
run_iao_decomp(supcell, mf, nkpt)
#run_all_decomp(supcell, mf, nkpt)
#run_eda_decomp(supcell, mf, nkpt)
