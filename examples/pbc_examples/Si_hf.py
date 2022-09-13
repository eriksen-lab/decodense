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
cell.atom = [['Si',[0.0,0.0,0.0]], ['Si',[1.35775, 1.35775, 1.35775]]]
cell.a = [[0.0, 2.7155, 2.7155], [2.7155, 0.0, 2.7155], [2.7155, 2.7155, 0.0]]
#cell.a = [[0.0, 22.7155, 22.7155], [22.7155, 0.0, 22.7155], [22.7155, 22.7155, 0.0]]
#cell.a = [[0.0, 152.7155, 152.7155], [152.7155, 0.0, 152.7155], [152.7155, 152.7155, 0.0]]
cell.basis = 'sto3g'
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

