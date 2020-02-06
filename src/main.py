#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main mf_decomp program
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import os.path
import shutil
import numpy as np
from pyscf import gto, scf, dft, lib

import system
import orbitals
import energy
import results
import tools


def main():
    """
    main program
    """
    # setup calculation
    decomp = _setup()

    # init molecule
    mol = gto.Mole()
    mol.build(verbose = 0, output = None, atom = decomp.atom, \
                basis = decomp.param['basis'], symmetry = True)

    # singlet check
    assert mol.spin == 0, 'decomposition scheme only implemented for singlet states'

    # overlap matrix
    s = mol.intor_symmetric('int1e_ovlp')
    # ao dipole integrals with gauge origin at (0.0, 0.0, 0.0)
    with mol.with_common_origin([0.0, 0.0, 0.0]):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    # nuclear repulsion energy
    e_nuc = mol.energy_nuc()
    # nuclear dipole moment
    dip_nuc = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    # inter-atomic distance array
    rr = gto.mole.inter_distance(mol) * lib.param.BOHR

    # hf calc
    mf_hf = scf.RHF(mol)
    mf_hf.conv_tol = 1.0e-12
    mf_hf.kernel()
    assert mf_hf.converged, 'HF not converged'

    # molecular dimensions
    mol.ncore = orbitals.set_ncore(mol)
    mol.nocc = np.where(mf_hf.mo_occ > 0.)[0].size
    mol.nvirt = np.where(mf_hf.mo_occ == 0.)[0].size
    mol.norb = mol.nocc + mol.nvirt

    # print result header
    print(results.main(mol, decomp))

    # decompose hf energy by means of canonical orbitals
    rep_idx, mo_hf_can = np.arange(mol.nocc), mf_hf.mo_coeff
    e_hf, dip_hf = energy.e_tot(mol, mf_hf, 'hf_can', ao_dip, \
                                    mo_hf_can[:, :mol.nocc], rep_idx, decomp.param['cube'])
    # decompose hf energy by means of localized MOs
    mo_hf_loc = orbitals.loc_orbs(mol, mf_hf.mo_coeff, s, decomp.param['loc'])
    rep_idx, centres_hf = orbitals.reorder(mol, s, mo_hf_loc, decomp.param['pop'], decomp.param['thres'])
    e_hf_loc, dip_hf_loc = energy.e_tot(mol, mf_hf, 'hf_loc', ao_dip, \
                                            mo_hf_loc[:, :mol.nocc], rep_idx, decomp.param['cube'])
    # sort results
    e_hf, dip_hf = results.sort(mol, 'hf_can', e_hf, dip_hf, decomp.param['cube'])[:2]
    e_hf_loc, dip_hf_loc, centres_hf = results.sort(mol, 'hf_loc', e_hf_loc, dip_hf_loc, \
                                                    decomp.param['cube'], centres=centres_hf)

    # print hf results
    print('\n\n ** hartree-fock')
    print(' ---------------\n')
    print(results.energy(mol, e_hf, e_hf_loc, e_nuc, mf_hf.e_tot, centres_hf, rr))
    print(results.dipole(mol, dip_hf, dip_hf_loc, dip_nuc, \
                            scf.hf.dip_moment(mol, mf_hf.make_rdm1(), unit='au', verbose=0), centres_hf, rr))

    # delete mf_hf
    del mf_hf

    if decomp.param['dft']:

        # dft calc
        mf_dft = dft.RKS(mol)
        mf_dft.xc = decomp.param['xc']
        mf_dft.conv_tol = 1.0e-12
        mf_dft.kernel()
        assert mf_dft.converged, 'DFT not converged'

        # decompose dft energy by means of canonical orbitals
        rep_idx, mo_dft_can = np.arange(mol.nocc), mf_dft.mo_coeff
        e_dft, dip_dft = energy.e_tot(mol, mf_dft, 'dft_can', ao_dip, \
                                        mo_dft_can[:, :mol.nocc], rep_idx, decomp.param['cube'])
        # decompose dft energy by means of localized MOs
        mo_dft_loc = orbitals.loc_orbs(mol, mf_dft.mo_coeff, s, decomp.param['loc'])
        rep_idx, centres_dft = orbitals.reorder(mol, s, mo_dft_loc, decomp.param['pop'], decomp.param['thres'])
        e_dft_loc, dip_dft_loc = energy.e_tot(mol, mf_dft, 'dft_loc', ao_dip, \
                                                mo_dft_loc[:, :mol.nocc], rep_idx, decomp.param['cube'])
        # sort results
        e_dft, dip_dft = results.sort(mol, 'dft_can', e_dft, dip_dft, decomp.param['cube'])[:2]
        e_dft_loc, dip_dft_loc, centres_dft = results.sort(mol, 'dft_loc', e_dft_loc, dip_dft_loc, \
                                                           decomp.param['cube'], centres=centres_dft)

        # print dft results
        print('\n\n ** dft')
        print(' ------\n')
        print(results.energy(mol, e_dft, e_dft_loc, e_nuc, mf_dft.e_tot, centres_dft, rr))
        print(results.dipole(mol, dip_dft, dip_dft_loc, dip_nuc, \
                                scf.hf.dip_moment(mol, mf_dft.make_rdm1(), unit='au', verbose=0), centres_dft, rr))

        # delete mf_dft
        del mf_dft


def _setup() -> system.DecompCls:
    """
    set decomp info
    """
    # decomp object
    decomp = system.DecompCls()
    decomp.atom, decomp.param = system.set_param(decomp.param)
    if 'xc' in decomp.param.keys():
        decomp.param['dft'] = True

    # rm out dir if present
    if os.path.isdir(results.OUT):
        shutil.rmtree(results.OUT, ignore_errors=True)

    # make main out dir
    os.mkdir(results.OUT)
    if decomp.param['cube']:
        # make hf out dirs
        os.mkdir(results.OUT + '/hf_can')
        os.mkdir(results.OUT + '/hf_loc')
        # make dft out dirs
        if decomp.param['dft']:
            os.mkdir(results.OUT + '/dft_can')
            os.mkdir(results.OUT + '/dft_loc')

    # init logger
    sys.stdout = tools.Logger(results.RES_FILE) # type: ignore

    return decomp


if __name__ == '__main__':
    main()


