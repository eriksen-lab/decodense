#!/usr/bin/env python

import numpy as np
from pyscf.pbc import gto, scf
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

##########################################
######### CELL OBJECT FOR TESTING ########
##########################################
#
# cell
cell = gto.Cell()
cell.atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
#
# Note the extra attribute ".a" in the "cell" initialization.
# .a is a matrix for lattice vectors.  Each row of .a is a primitive vector.
#
cell.a = np.eye(3)*3.5668
cell.build()

## all-electron calculation
#cell = gto.M(
#    a = np.eye(3)*3.5668,
#    atom = '''C     0.      0.      0.    
#              C     0.8917  0.8917  0.8917
#              C     1.7834  1.7834  0.    
#              C     2.6751  2.6751  0.8917
#              C     1.7834  0.      1.7834
#              C     2.6751  0.8917  2.6751
#              C     0.      1.7834  1.7834
#              C     0.8917  2.6751  2.6751''',
#    #basis = '6-31g',
#    basis = 'sto3g',
#    verbose = 2,
#)
#


mf = scf.RHF(cell)
ehf = mf.kernel()
print("HF energy (per unit cell) = %.17g" % ehf)


# init decomp object
decomp = decodense.DecompCls(**PARAMS)
#print('decomp', dir(decomp))
#####
res = decodense.main(cell, decomp, mf)

####print results
#print(dir(res))
for k, v in res.items():
    print(k, v)

print('E_nuc from atomwise calculation: ', np.sum(res['struct']) )
print('E_nuc from pyscf: ', cell.energy_nuc())

#cell1 = cell
#print('cell ', cell.energy_nuc())
#print('cell1 ', cell1.energy_nuc())
#cell1.a = np.eye(3)*2.*3.5668
#print('test')
#print('cell ', cell.energy_nuc())
#print('cell1 ', cell1.energy_nuc())
