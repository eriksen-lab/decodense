import importlib
import sys
import h5py
import numpy as np
import pandas as pd
from pyscf.pbc import scf
from pyscf import scf as mol_scf
from pyscf.pbc.tools.pbc import super_cell#, cell_plus_imgs
import decodense

name = sys.argv[0]
structure = sys.argv[1]
nim = int(sys.argv[2])
# make the nimgs list (to amke a supercell with) 
nimgs = [nim, nim, nim]
# remove .py
name, structure = name[:-3], structure[:-3]
name = name.split('/')[-1]
chkfile_name = sys.argv[3]

# a function that loops over all decodense params
# the results are written to separate csv files for each
# combination of params (name: jobname_pop_loc_nimgs.csv) 
#def run_all_decomp(cell, mf, nimgs=1):
#    ''' run decodense with all params implemented for cell obj'''
#
#    ehf = mf.energy_tot()
#    nat = cell.natm
#    n_im = nimgs[0]
#    res_all = []
#    for i in ['', 'fb', 'pm', 'ibo-2', 'ibo-4']:
#        for j in ['mulliken', 'iao']:
#            decomp = decodense.DecompCls(prop='energy', part='atoms', loc=i, pop=j)
#            res = decodense.main(cell, decomp, mf)
#            print()
#            #print(decodense.results(cell, f'HF energy (localization procedure: {i}, population: {j})', dump_res = True, suffix = f'_{name}_{structure}_n{n_im}_{i}_{j}', **res))
#            #dframe = pd.read_csv(f'res_{name}_{structure}_n{n_im}_{i}_{j}.csv')
#            pd.options.display.max_columns = len(dframe.columns)
#            print()
#            print('---------------------------------')
#            ehf_dec = dframe['tot'].values 
#            ehf_dec_per_cell = np.sum(ehf_dec) / np.prod(nimgs)
#            print('E_hf_dec per cell = ', ehf_dec_per_cell)
#            print('E_hf_pyscf - E_hf_dec = ', ehf - ehf_dec_per_cell)
#            print('---------------------------------')
#            print()
#    return print('Done decomposing!')

def pick_center_atom(cell):
    ''' Find an atom closest to the center of the cell, return it's index  '''
    center_point = np.sum(cell.a, axis=0) *.5
    coords = cell.atom_coords() - center_point
    dist = np.einsum('ij,ij->i', coords, coords)
    dist = np.sqrt(dist)
    center_atm_idx = np.argmin(abs(dist))     
    return center_atm_idx, coords[center_atm_idx,:]


def save_scf_iteration(envs):
    cycle = envs['cycle']
    info = {'fock'    : envs['fock'],
            'dm'      : envs['dm'],
            'mo_coeff': envs['mo_coeff'],
            'mo_energy':envs['mo_energy'],
            'e_tot'   : envs['e_tot']}
    scf.chkfile.save(mf.chkfile, 'HF-iteration/%d' % cycle, info)

# import cell from a given file
struct_mod = importlib.import_module(str(structure), package=None) 
#from struct_mod import cell
cell = struct_mod.cell
# make a super cell
supcell = super_cell(cell, nimgs)
coords = supcell.atom_coords()

#print('loop')
#for i in range(16):
#    print('atom i', i, supcell.atom_symbol(i))
#    print('atom i coord', supcell.atom_coord(i))

# compute the centerpoint
b, b_coord = pick_center_atom(supcell)
print('b ', b)
print('b_coord ', b_coord, np.sqrt(b_coord.dot(b_coord)) )
print('Supercell lattice vectors')
print(supcell.a)
print('Center atom is: ', b)
pd.DataFrame(coords).to_csv(f'coords_{name}_{structure}_n{nim}.csv', index=None, header=['x', 'y', 'z'])

# run HF calculation
mf = scf.RHF(supcell).density_fit().apply(mol_scf.addons.remove_linear_dep_)
mf.chkfile = chkfile_name
mf.callback = save_scf_iteration
mf.init_guess = 'chkfile'
ehf = mf.kernel()
print("HF energy (per unit cell) = %.17g" % ehf)
with h5py.File(chkfile_name, 'r') as f:
    print(f['HF-iteration'].keys())
    print(f['HF-iteration/1'].keys())


# decompose the results
#decomp = decodense.DecompCls(prop='energy', part='atoms', loc='', pop='mulliken')
#res = decodense.main(supcell, decomp, mf)
#print(decodense.results(cell, f'HF energy (localization procedure: none, population: mulliken)', dump_res = True, suffix = f'_RHF_diamond_prim_n2__mulliken', **res))
