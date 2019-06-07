#!/usr/bin/env python
#
# Author: Janus Juul Eriksen <januseriksen@gmail.com>
#

import sys
import numpy as np
from pyscf import gto, scf, dft, lo, symm


def e_elec(h_core, vj, vk, rdm1):
    """
    this function returns a contribution to a mean-field energy from rdm1:
    E(rdm1) = 2. * Tr[h * rdm1] + Tr[v_eff(rdm1_tot) * rdm1]

    :param h_core: core hamiltonian. numpy array of shape (n_orb, n_orb)
    :param vj: coulumb potential. numpy array of shape (n_orb, n_orb)
    :param vk: exchange potential. numpy array of shape (n_orb, n_orb)
    :param rdm1: orbital specific rdm1. numpy array of shape (n_orb, n_orb)
    :return: scalar
    """
    # contribution from core hamiltonian
    e_core = np.einsum('ij,ji', h_core, rdm1) * 2.

    # contribution from effective potential
    e_veff = np.einsum('ij,ji', vj, rdm1)
    if vk is not None:
        e_veff -= np.einsum('ij,ji', vk * .5, rdm1)

    return e_core + e_veff


def e_tot(mol, mf, s, mo_coeff, dft=False):
    """
    this function returns a sorted orbital-decomposed mean-field energy for a given orbital variant

    :param mol: pyscf mol object
    :param mf: pyscf mean-field object
    :param s: overlap matrix. numpy array of shape (n_orb, n_orb)
    :param mo_coeff: mo coefficients. numpy array of shape (n_orb, n_orb)
    :param dft: dft logical. bool
    :return: numpy array of shape (nocc,)
    """
    # compute 1-RDM (in AO representation)
    rdm1 = mf.make_rdm1(mo_coeff, mf.mo_occ)

    # core hamiltonian
    h_core = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')

    # mean-field effective potential
    if dft:

        v_dft = mf.get_veff(mol, rdm1)
        vj, vk = v_dft.vj, v_dft.vk

    else:

        vj, vk = mf.get_jk(mol, rdm1)

    # init orbital energy array
    e_orb = np.zeros(mol.nocc, dtype=np.float64)
    # init charge_centres list
    centres = []

    # loop over orbitals
    for orb in range(mol.nocc):

        # orbital-specific 1rdm
        rdm1_orb = np.einsum('ip,jp->ij', mo_coeff[:, [orb]], mo_coeff[:, [orb]])

        # charge centres of orbital
        centres.append(charge_centres(mol, s, rdm1_orb))

        # energy from individual orbitals
        e_orb[orb] = e_elec(h_core, vj, vk, rdm1_orb)

    # convert centres to array
    centres = np.array(centres)

    # sort arrays wrt e_orb
    centres = centres[np.argsort(e_orb)]
    e_orb = np.sort(e_orb)

    return e_orb, centres


def loc_orbs(mol, mf, s, variant):
    """
    this function returns a set of localized MOs of a specific variant

    :param mol: pyscf mol object
    :param mf: pyscf mf object
    :param s: overlap matrix. numpy array of shape (n_orb, n_orb)
    :param variant: localization variant. string
    :return: numpy array of shape (n_orb, n_orb)
    """
    # copy MOs from mean-field object
    mo_coeff = np.copy(mf.mo_coeff)

    # init localizer
    if variant == 'boys':

        # Foster-Boys procedure
        loc = lo.Boys(mol, mo_coeff[:, :mol.nocc])

    elif variant == 'pm':

        # Pipek-Mezey procedure
        loc = lo.PM(mol, mo_coeff[:, :mol.nocc])

    elif variant == 'er':

        # Edmiston-Ruedenberg procedure
        loc = lo.ER(mol, mo_coeff[:, :mol.nocc])

    elif variant == 'ibo':

        # compute IAOs
        a = lo.iao.iao(mol, mo_coeff[:, :mol.nocc])

        # orthogonalize IAOs
        a = lo.vec_lowdin(a, s)

        # IBOs via Pipek-Mezey procedure
        loc = lo.ibo.PM(mol, mo_coeff[:, :mol.nocc], a)

    else:

        raise RuntimeError('unknown localization procedure')

    # convergence threshold
    loc.conv_tol = 1.0e-10

    # localize occupied orbitals
    mo_coeff[:, :mol.nocc] = loc.kernel()

    return mo_coeff


def set_ncore(mol):
    """
    this function returns number of core orbitals

    :param mol: pyscf mol object
    :return: integer
    """
    # init ncore
    ncore = 0

    # loop over atoms
    for i in range(mol.natm):

        if mol.atom_charge(i) > 2: ncore += 1
        if mol.atom_charge(i) > 12: ncore += 4
        if mol.atom_charge(i) > 20: ncore += 4
        if mol.atom_charge(i) > 30: ncore += 6

    return ncore


def energy_nuc(mol):
    """
    this function returns the nuclear repulsion energy for all atoms of the system

    :param mol: pyscf mol object
    :return: numpy array of shape (natm,)
    """
    # charges
    charges = mol.atom_charges()

    # coordinates
    coords = mol.atom_coords()

    # init e_nuc
    e_nuc = np.zeros(mol.natm)

    # loop over atoms
    for j in range(mol.natm):

        # charge and coordinates of atom_j
        q_j = charges[j]
        r_j = coords[j]

        # loop over atoms < atom_j
        for i in range(j):

            # charge and coordinates of atom_i
            q_i = charges[i]
            r_i = coords[i]

            # distance between atom_j & atom_i
            r = np.linalg.norm(r_i - r_j)

            # repulsion energy
            e_nuc[j] += (q_i * q_j) / r

    return e_nuc


def charge_centres(mol, s, rdm1):
    """
    this function returns the mulliken charges on the individual atoms

    :param mol: pyscf mol object
    :param s: overlap matrix. numpy array of shape (n_orb, n_orb)
    :param rdm1: orbital specific rdm1. numpy array of shape (n_orb, n_orb)
    :return: numpy array of shape (natm,)
    """
    # mulliken population matrix
    pop = np.einsum('ij,ji->i', rdm1, s).real

    # init charges
    charges = np.zeros(mol.natm)

    # loop over AOs
    for i, s in enumerate(mol.ao_labels(fmt=None)):

        charges[s[0]] += pop[i]

    # get sorted indices
    max_idx = np.argsort(charges)[::-1]

    if np.abs(charges[max_idx[0]]) / np.abs((charges[max_idx[0]] + charges[max_idx[1]])) > 0.95:

        # core orbital
        return [mol.atom_symbol(max_idx[0]), mol.atom_symbol(max_idx[0])]

    else:

        # valence orbitals
        return [mol.atom_symbol(max_idx[0]), mol.atom_symbol(max_idx[1])]


def main():
    """ main program """

    # read in molecule argument
    if len(sys.argv) != 4:
        sys.exit('\n missing or too many arguments: python orb_decomp.py molecule xc_functional localization_procedure\n')

    # set molecule
    molecule = sys.argv[1]
    xc_func = sys.argv[2]
    loc_proc = sys.argv[3]


    # init molecule
    mol = gto.Mole()
    mol.build(
    verbose = 0,
    output = None,
    atom = open('../structures/'+molecule+'.xyz').read(),
    basis = '631g',
    symmetry = True,
    )


    # singlet check
    assert mol.spin == 0, 'decomposition scheme only implemented for singlet states'


    # molecular dimensions
    mol.ncore = set_ncore(mol)
    mol.nocc = mol.nelectron // 2


    # nuclear repulsion energy
    e_nuc = np.sum(energy_nuc(mol))
    # overlap matrix
    s = mol.intor_symmetric('int1e_ovlp')


    # init and run HF calc
    mf_hf = scf.RHF(mol)
    mf_hf.conv_tol = 1.0e-12
    mf_hf.run()
    assert mf_hf.converged, 'HF not converged'

    # init and run DFT (B3LYP) calc
    mf_dft = dft.RKS(mol)
    mf_dft.xc = xc_func
    mf_dft.conv_tol = 1.0e-12
    mf_dft.run()
    assert mf_hf.converged, 'DFT not converged'


    # energy of XC functional evaluated on a grid
    e_xc = mf_dft._numint.nr_rks(mol, mf_dft.grids, mf_dft.xc, \
                                 mf_dft.make_rdm1(mf_dft.mo_coeff, mf_dft.mo_occ))[1]


    # decompose HF energy by means of canonical orbitals
    mo_coeff = mf_hf.mo_coeff
    e_hf = e_tot(mol, mf_hf, s, mo_coeff)[0]

    # decompose HF energy by means of localized MOs
    mo_coeff = loc_orbs(mol, mf_hf, s, loc_proc)
    e_hf_loc, centres_hf = e_tot(mol, mf_hf, s, mo_coeff)

    # decompose DFT energy by means of canonical orbitals
    mo_coeff = mf_dft.mo_coeff
    e_dft = e_tot(mol, mf_dft, s, mo_coeff, dft=True)[0]

    # decompose DFT energy by means of localized MOs
    mo_coeff = loc_orbs(mol, mf_dft, s, loc_proc)
    e_dft_loc, centres_dft = e_tot(mol, mf_dft, s, mo_coeff, dft=True)


    # print results
    print('\n\n results for: {:} with localization procedure: {:}'.format(molecule, loc_proc))


    print('\n\n hartree-fock\n')
    print('  MO  |   canonical   |   localized   |     atom(s)    |    bond length')
    print('-------------------------------------------------------------------------')
    for i in range(mol.nocc):
        print('  {:>2d}  | {:>10.3f}    | {:>10.3f}    |    {:^10s}  | {:>10.3f}'. \
                format(i, e_hf[i], e_hf_loc[i], \
                       centres_hf[i, 0] if centres_hf[i, 0] == centres_hf[i, 1] else \
                       '{:s} & {:s}'.format(*centres_hf[i]), \
                       0.0))
    print('-------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------')
    print('  sum | {:>10.3f}    | {:>10.3f}    |'. \
            format(np.sum(e_hf), np.sum(e_hf_loc)))
    print('-------------------------------------------------------------------------')
    print('  nuc | {:>+10.3f}    | {:>+10.3f}    |'. \
            format(e_nuc, e_nuc))
    print('-------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------')
    print('  sum | {:>12.5f}  | {:>12.5f}  |'. \
            format(np.sum(e_hf) + e_nuc, np.sum(e_hf_loc) + e_nuc))

    print('\n *** HF reference energy  = {:.5f}'. \
            format(mf_hf.e_tot))


    print('\n\n dft ({:s})\n'.format(xc_func))
    print('  MO  |   canonical   |   localized   |     atom(s)    |    bond length')
    print('-------------------------------------------------------------------------')
    for i in range(mol.nocc):
        print('  {:>2d}  | {:>10.3f}    | {:>10.3f}    |    {:^10s}  | {:>10.3f}'. \
                format(i, e_dft[i], e_dft_loc[i], \
                       centres_dft[i, 0] if centres_dft[i, 0] == centres_dft[i, 1] else \
                       '{:s} & {:s}'.format(*centres_dft[i]), \
                       0.0))
    print('-------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------')
    print('  sum | {:>10.3f}    | {:>10.3f}    |'. \
            format(np.sum(e_dft), np.sum(e_dft_loc)))
    print('-------------------------------------------------------------------------')
    print('  nuc | {:>+10.3f}    | {:>+10.3f}    |'. \
            format(e_nuc, e_nuc))
    print('  xc  | {:>+10.3f}    | {:>+10.3f}    |'. \
            format(e_xc, e_xc))
    print('-------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------')
    print('  sum | {:>12.5f}  | {:>12.5f}  |'. \
            format(np.sum(e_dft) + e_nuc + e_xc, np.sum(e_dft_loc) + e_nuc + e_xc))

    print('\n *** DFT reference energy = {:.5f}\n\n'. \
            format(mf_dft.e_tot))


if __name__ == '__main__':
    main()


