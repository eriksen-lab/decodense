import numpy as np
from pyscf import gto, scf
import decodense

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


mol = gto.Mole()
mol.verbose = 2
#mol.output = 'out_h2o'
mol.atom = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587'''
mol.basis = 'ccpvdz'
#mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf.kernel()

#print('rhf', dir(mf))
#print('e_nuc', mol.energy_nuc())

# init decomp object
decomp = decodense.DecompCls(**PARAMS)
#print('decomp', dir(decomp))
#####
res = decodense.main(mol, decomp, mf)

####print results
#print(dir(res))
for k, v in res.items():
    print(k, v)
