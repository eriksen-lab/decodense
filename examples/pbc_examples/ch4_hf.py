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
cell.atom = '''
 C                  3.17500000    3.17500000    3.17500000
 H                  2.54626556    2.54626556    2.54626556
 H                  3.80373444    3.80373444    2.54626556
 H                  2.54626556    3.80373444    3.80373444
 H                  3.80373444    2.54626556    3.80373444
'''
cell.basis = 'sto3g'
cell.a = np.eye(3) * 6.35
cell.build()
#

# buffer too small if not using mixed DF
# TypeError: buffer is too small for requested array
# WARN: FFTDF integrals are found in all-electron calculation.  It often causes huge error.
# Recommended methods are DF or MDF.
mf = scf.RHF(cell).mix_density_fit()
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

cell1 = cell
print('cell ', cell.energy_nuc())
print('cell1 ', cell1.energy_nuc())
cell1.a = np.eye(3)*2.*6.35
print('test')
print('cell ', cell.energy_nuc())
print('cell1 ', cell1.energy_nuc())
