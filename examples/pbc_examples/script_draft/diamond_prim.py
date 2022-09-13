import numpy as np
from pyscf.pbc import gto, scf

# initialize a cell object
# the (1/4, 1/4, 1/4) basis atoms are: #2, i think?
cell = gto.Cell()
cell.atom = '''
 C   3.79049   2.18844   1.54746
 C   2.52699   1.45896   1.03164
'''
cell.basis = 'sto3g'
# .a is a matrix for lattice vectors.  Each row of .a is a primitive vector.
cell.a = np.eye(3)*2.52699
cell.a[1:,0] = 1.26350
cell.a[1,1], cell.a[2,2], cell.a[2,1] = 2.18844, 2.06328, 0.72948
cell.verbose = 3
#cell.verbose = 5
cell.build()


