
import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto, scf
from pyscf import gto as mgto
from pyscf import scf as mscf
from pyscf.pbc.tools.k2gamma import k2gamma
from pyscf.pbc import tools
from pyscf.pbc.tools.k2gamma import get_phase

# optional check if transformation went fine
def check_k2gamma_ovlps(cell, scell, phase, kmesh, kmf, mf):
    # 1. transform ao overlap ints and check if it's sensible
    print('Checking the k2gamma transformation of AO ints')
    NR, Nk = phase.shape
    nao = cell.nao
    # TODO note that if kpts not give only returns Gamma-p. ints
    s_k = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    s = scell.pbc_intor('int1e_ovlp')
    s1 = np.einsum('Rk,kij,Sk->RiSj', phase, s_k, phase.conj())
    print('Difference between S for scell and S_k transformed to real space')
    print(abs(s-s1.reshape(s.shape)).max())
    s = scell.pbc_intor('int1e_ovlp').reshape(NR,nao,NR,nao)
    s1 = np.einsum('Rk,RiSj,Sk->kij', phase.conj(), s, phase)
    print(abs(s1-s_k).max())

    # 2. The following is to check whether the MO is correctly coverted:
    print("Supercell gamma MO in AO basis from conversion:")
    scell = tools.super_cell(cell, kmesh)
    # FIXME df here will prob be different than for RKS calc...
    mf_sc = scf.RHF(scell).density_fit()
    print('mf_sc type df')
    print(mf_sc.with_df)
    c_g_ao = mf.mo_coeff
    s = mf_sc.get_ovlp()
    mf_sc.run()
    sc_mo = mf_sc.mo_coeff
    nocc = scell.nelectron // 2
    print("Supercell gamma MO from direct calculation:")
    print(np.linalg.det(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc])))
    print(np.linalg.svd(c_g_ao[:,:nocc].T.conj().dot(s).dot(sc_mo[:,:nocc]))[1])
    return



##########################################################
# after cell is loaded
##########################################################

#default: shifted Monkhorst-Pack mesh centered at Gamma-p.
#to get non-shifted: with_gamma_point=False
#to get centered at specific p.(units of lattice vectors): scaled_center=[0.,0.25,0.25]
#mesh: The numbers of grid points in the FFT-mesh in each direction
# this could be sys.arg like n_cellnr
kmesh = [2,2,2]
kpts = cell.make_kpts(kmesh)
kmf = scf.KRHF(cell, kpts).density_fit()#.newton() for RHF
ehf = kmf.kernel()
kdm = kmf.make_rdm1()
print("HF energy (per unit cell) = %.17g" % ehf)

# transform the kmf object to mf obj for a supercell
# overwrite df object to GDF
mf = k2gamma(kmf) 
scell, phase = get_phase(cell, kpts)
mydf = df.df.DF(scell)
mf.with_df = mydf
print('df obj. of type: ', mf.with_df)
dm = (mf.make_rdm1()).real
# check sanity
#check_k2gamma_ovlps(cell, scell, phase, kmesh, kmf, mf)

# pass supercell and its mf obj to decodense
check_decomp(scell, mf)
