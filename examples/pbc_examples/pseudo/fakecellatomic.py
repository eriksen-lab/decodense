
def _fake_cell_vloc(cell, cn=0):
    '''Generate fake cell for V_{loc}.

    Each term of V_{loc} (erf, C_1, C_2, C_3, C_4) is a gaussian type
    function.  The integral over V_{loc} can be transfered to the 3-center
    integrals, in which the auxiliary basis is given by the fake cell.

    The kwarg cn indiciates which term to generate for the fake cell.
    If cn = 0, the erf term is generated.  C_1,..,C_4 are generated with cn = 1..4
    '''
    fake_env = [cell.atom_coords().ravel()]
    fake_atm = cell._atm.copy()
    fake_atm[:,gto.PTR_COORD] = np.arange(0, cell.natm*3, 3)
    ptr = cell.natm * 3
    fake_bas = []
    half_sph_norm = .5/np.sqrt(np.pi)
    for ia in range(cell.natm):
        if cell.atom_charge(ia) == 0:  # pass ghost atoms
            continue

        symb = cell.atom_symbol(ia)
        if cn == 0:
            if symb in cell._pseudo:
                pp = cell._pseudo[symb]
                rloc, nexp, cexp = pp[1:3+1]
                alpha = .5 / rloc**2
            else:
                alpha = 1e16
            norm = half_sph_norm / gto.gaussian_int(2, alpha)
            fake_env.append([alpha, norm])
            fake_bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
            ptr += 2
        elif symb in cell._pseudo:
            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]
            if cn <= nexp:
                alpha = .5 / rloc**2
                norm = cexp[cn-1]/rloc**(cn*2-2) / half_sph_norm
                fake_env.append([alpha, norm])
                fake_bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
                ptr += 2

    fakecell = copy.copy(cell)
    fakecell._atm = np.asarray(fake_atm, dtype=np.int32)
    fakecell._bas = np.asarray(fake_bas, dtype=np.int32)
    fakecell._env = np.asarray(np.hstack(fake_env), dtype=np.double)
    return fakecell

# sqrt(Gamma(l+1.5)/Gamma(l+2i+1.5))
_PLI_FAC = 1/np.sqrt(np.array((
    (1, 3.75 , 59.0625  ),  # l = 0,
    (1, 8.75 , 216.5625 ),  # l = 1,
    (1, 15.75, 563.0625 ),  # l = 2,
    (1, 24.75, 1206.5625),  # l = 3,
    (1, 35.75, 2279.0625),  # l = 4,
    (1, 48.75, 3936.5625),  # l = 5,
    (1, 63.75, 6359.0625),  # l = 6,
    (1, 80.75, 9750.5625))))# l = 7,

def _fake_cell_vnl(cell):
    '''Generate fake cell for V_{nl}.

    gaussian function p_i^l Y_{lm}
    '''
    # TODO in (nl, rl, hl) hl is the proj part hproj
    '''{ atom: ( (nelec_s, nele_p, nelec_d, ...),
                rloc, nexp, (cexp_1, cexp_2, ..., cexp_nexp),
                nproj_types,
                (r1, nproj1, ( (hproj1[1,1], hproj1[1,2], ..., hproj1[1,nproj1]),
                               (hproj1[2,1], hproj1[2,2], ..., hproj1[2,nproj1]),
                               ...
                               (hproj1[nproj1,1], hproj1[nproj1,2], ...        ) )),
                (r2, nproj2, ( (hproj2[1,1], hproj2[1,2], ..., hproj2[1,nproj1]),
                ... ) )
                )
        ... }]i
    '''
    fake_env = [cell.atom_coords().ravel()]
    fake_atm = cell._atm.copy()
    fake_atm[:,gto.PTR_COORD] = np.arange(0, cell.natm*3, 3)
    ptr = cell.natm * 3
    fake_bas = []
    hl_blocks = []
    for ia in range(cell.natm):
        if cell.atom_charge(ia) == 0:  # pass ghost atoms
            #print('in fake_cell_vnl: ghost atomed passed..')
            continue

        symb = cell.atom_symbol(ia)
        #print('in fake_cell_vnl: ia(#atm), atm symbol', ia, symb)
        if symb in cell._pseudo:
            #print('in fake_cell_vnl: symb was in cell._psuedo')
            pp = cell._pseudo[symb]
            #print('in fake_cell_vnl: pp from cell._pseudo', type(pp), np.shape(pp), pp)
            # nproj_types = pp[4]
            for l, (rl, nl, hl) in enumerate(pp[5:]):
                #print('in fake_cell_vnl: l, (rl, nl, hl) ', l, (rl, nl, hl) )
                if nl > 0:
                    alpha = .5 / rl**2
                    norm = gto.gto_norm(l, alpha)
                    fake_env.append([alpha, norm])
                    fake_bas.append([ia, l, 1, 1, 0, ptr, ptr+1, 0])

#
# Function p_i^l (PRB, 58, 3641 Eq 3) is (r^{2(i-1)})^2 square normalized to 1.
# But here the fake basis is square normalized to 1.  A factor ~ p_i^l / p_1^l
# is attached to h^l_ij (for i>1,j>1) so that (factor * fake-basis * r^{2(i-1)})
# is normalized to 1.  The factor is
#       r_l^{l+(4-1)/2} sqrt(Gamma(l+(4-1)/2))      sqrt(Gamma(l+3/2))
#     ------------------------------------------ = ----------------------------------
#      r_l^{l+(4i-1)/2} sqrt(Gamma(l+(4i-1)/2))     sqrt(Gamma(l+2i-1/2)) r_l^{2i-2}
#
                    fac = np.array([_PLI_FAC[l,i]/rl**(i*2) for i in range(nl)])
                    #print('in fake_cell_vnl: fac', fac)
                    hl = np.einsum('i,ij,j->ij', fac, np.asarray(hl), fac)
                    #print('in fake_cell_vnl: hl contracted with fac', hl)
                    hl_blocks.append(hl)
                    ptr += 2

    fakecell = copy.copy(cell)
    fakecell._atm = np.asarray(fake_atm, dtype=np.int32)
    fakecell._bas = np.asarray(fake_bas, dtype=np.int32)
    fakecell._env = np.asarray(np.hstack(fake_env), dtype=np.double)
    return fakecell, hl_blocks
