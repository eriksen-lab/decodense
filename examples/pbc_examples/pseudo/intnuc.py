
def _int_nuc_vloc(mydf, nuccell, kpts, intor='int3c2e', aosym='s2', comp=1):
    '''Vnuc - Vloc'''
    cell = mydf.cell
    nkpts = len(kpts)

    # Use the 3c2e code with steep s gaussians to mimic nuclear density
    fakenuc = _fake_nuc(cell)
    fakenuc._atm, fakenuc._bas, fakenuc._env = \
            gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                         fakenuc._atm, fakenuc._bas, fakenuc._env)

    kptij_lst = numpy.hstack((kpts,kpts)).reshape(-1,2,3)
    buf = incore.aux_e2(cell, fakenuc, intor, aosym=aosym, comp=comp,
                        kptij_lst=kptij_lst)

    charge = cell.atom_charges()
    charge = numpy.append(charge, -charge)  # (charge-of-nuccell, charge-of-fakenuc)
    nao = cell.nao_nr()
    nchg = len(charge)
    if aosym == 's1':
        nao_pair = nao**2
    else:
        nao_pair = nao*(nao+1)//2
    if comp == 1:
        buf = buf.reshape(nkpts,nao_pair,nchg)
        mat = numpy.einsum('kxz,z->kx', buf, charge)
    else:
        buf = buf.reshape(nkpts,comp,nao_pair,nchg)
        mat = numpy.einsum('kcxz,z->kcx', buf, charge)

    # vbar is the interaction between the background charge
    # and the compensating function.  0D, 1D, 2D do not have vbar.
    if cell.dimension == 3 and intor in ('int3c2e', 'int3c2e_sph',
                                         'int3c2e_cart'):
        assert(comp == 1)
        charge = -cell.atom_charges()

        nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
        nucbar *= numpy.pi/cell.vol

        ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
        for k in range(nkpts):
            if aosym == 's1':
                mat[k] -= nucbar * ovlp[k].reshape(nao_pair)
            else:
                mat[k] -= nucbar * lib.pack_tril(ovlp[k])
    return mat
