
def _int_nuc_vloc_atomic(mydf, nuccell, kpts, intor='int3c2e', aosym='s2', comp=1):
    '''Vnuc - Vloc'''
    cell = mydf.cell
    nkpts = len(kpts)

    # Use the 3c2e code with steep s gaussians to mimic nuclear density
    fakenuc = _fake_nuc(cell)
    fakenuc._atm, fakenuc._bas, fakenuc._env = \
            gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                         fakenuc._atm, fakenuc._bas, fakenuc._env)

    kptij_lst = np.hstack((kpts,kpts)).reshape(-1,2,3)
    buf = incore.aux_e2(cell, fakenuc, intor, aosym=aosym, comp=comp,
                        kptij_lst=kptij_lst)

    charge = cell.atom_charges()
    nchg = len(charge)
    charge = np.append(charge, -charge)  # (charge-of-nuccell, charge-of-fakenuc)
    nao = cell.nao_nr()
    nchg2 = len(charge)
    if aosym == 's1':
        nao_pair = nao**2
    else:
        nao_pair = nao*(nao+1)//2
    if comp == 1:
        buf = buf.reshape(nkpts,nao_pair,nchg2)
        mat1 = np.einsum('kxz,z->kzx', buf, charge)
        mat = mat1[:,nchg:,:] + mat1[:,:nchg,:]
    else:
        buf = buf.reshape(nkpts,comp,nao_pair,nchg2)
        mat1 = np.einsum('kczx,z->kczx', buf, charge)
        mat = mat1[:,:,nchg:,:] + mat1[:,:,:nchg,:]

    # vbar is the interaction between the background charge
    # and the compensating function.  0D, 1D, 2D do not have vbar.
    if cell.dimension == 3 and intor in ('int3c2e', 'int3c2e_sph',
                                         'int3c2e_cart'):
        assert(comp == 1)
        charge = -cell.atom_charges()
        
        # TODO check dim, if datatype complex
        nucbar = np.zeros(len(charge), dtype=np.float64)
        nucbar = np.asarray([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
        nucbar *= np.pi/cell.vol

        ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
        for k in range(nkpts):
            if aosym == 's1':
                for i in range(len(charge)):
                    mat[k,i,:] -= nucbar[i] * ovlp[k].reshape(nao_pair)
            else:
                for i in range(len(charge)):
                    mat[k,i,:] -= nucbar[i] * lib.pack_tril(ovlp[k])
    return mat
