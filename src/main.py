#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main mf_decomp program

usage:
python main.py `struc_path` `mol` `basis` `loc_proc` `pop_scheme` `xc_func` `cube`
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import numpy as np
from pyscf import gto, scf, dft

import orbitals
import energy
import results
import tools


def main():
    """ main program """

    # read in molecule argument
    if len(sys.argv) != 8:
        raise SyntaxError('missing or too many arguments:\n'
                          'python main.py `struc_path` `mol` `basis` `loc_proc` `pop_scheme` `xc_func` `cube`')

    # set system info
    system = {}
    system['struc_path'] = sys.argv[1]
    if system['struc_path'][-1] != '/':
        system['struc_path'] += '/'
    system['molecule'] = sys.argv[2]
    system['basis'] = sys.argv[3]
    system['loc_proc'] = sys.argv[4]
    system['pop_scheme'] = sys.argv[5]
    system['xc_func'] = sys.argv[6]
    if system['xc_func'].lower() == 'none':
        system['dft'] = False
    else:
        system['dft'] = True
    system['cube'] = sys.argv[7].lower() == 'true'


    # output path
    system['out_path'] = os.getcwd() + '/' + \
                         system['molecule'] + '_' + \
                         system['basis'] + '_' + \
                         system['loc_proc'] + '_' + \
                         system['pop_scheme'] + '_' + \
                         system['xc_func']


    # make out dir
    os.mkdir(system['out_path'])
    # make hf out dirs
    system['out_hf_can_path'] = system['out_path'] + '/hf_can'
    os.mkdir(system['out_hf_can_path'])
    system['out_hf_loc_path'] = system['out_path'] + '/hf_loc'
    os.mkdir(system['out_hf_loc_path'])
    # make dft out dirs
    if system['dft']:
        system['out_dft_can_path'] = system['out_path'] + '/dft_can'
        os.mkdir(system['out_dft_can_path'])
        system['out_dft_loc_path'] = system['out_path'] + '/dft_loc'
        os.mkdir(system['out_dft_loc_path'])


    # init logger
    sys.stdout = tools.Logger(system['out_path']+'/output.out')


    # init molecule
    mol = gto.Mole()
    mol.build(
    verbose = 0,
    output = None,
    atom = open(system['struc_path']+system['molecule']+'.xyz').read(),
    basis = system['basis'],
    symmetry = True,
    )

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


    # init and run hf calc
    mf_hf = scf.RHF(mol)
    mf_hf.conv_tol = 1.0e-12
    mf_hf.kernel()
    assert mf_hf.converged, 'HF not converged'
    e_hf_tot = mf_hf.e_tot
    dip_hf_tot = scf.hf.dip_moment(mol, mf_hf.make_rdm1(), unit='au', verbose=0)


    # molecular dimensions
    mol.ncore = orbitals.set_ncore(mol)
    mol.nocc = np.where(mf_hf.mo_occ > 0.)[0].size
    mol.nvirt = np.where(mf_hf.mo_occ == 0.)[0].size
    mol.norb = mol.nocc + mol.nvirt


    # init and run dft calc
    if system['dft']:

        mf_dft = dft.RKS(mol)
        mf_dft.xc = system['xc_func']
        mf_dft.conv_tol = 1.0e-12
        mf_dft.kernel()
        assert mf_dft.converged, 'DFT not converged'
        e_dft_tot = mf_dft.e_tot
        dip_dft_tot = scf.hf.dip_moment(mol, mf_dft.make_rdm1(), unit='au', verbose=0)

        # energy of xc functional evaluated on a grid
        e_xc = mf_dft._numint.nr_rks(mol, mf_dft.grids, mf_dft.xc, \
                                     mf_dft.make_rdm1(mf_dft.mo_coeff, mf_dft.mo_occ))[1]

    else:

        e_dft_tot = dip_dft_tot = e_xc = None


    # decompose hf energy by means of canonical orbitals
    rep_idx, mo_hf_can = np.arange(mol.nocc), mf_hf.mo_coeff
    e_hf, dip_hf = energy.e_tot(mol, system, 'hf_can', ao_dip, mo_hf_can[:, :mol.nocc], rep_idx)

    # decompose hf energy by means of localized MOs
    mo_hf_loc = orbitals.loc_orbs(mol, mf_hf.mo_coeff, s, system['loc_proc'])
    rep_idx, centres_hf = orbitals.reorder(mol, mf_hf, s, mo_hf_loc, pop=system['pop_scheme'])
    e_hf_loc, dip_hf_loc = energy.e_tot(mol, system, 'hf_loc', ao_dip, mo_hf_loc[:, :mol.nocc], rep_idx)

    # decompose dft energy by means of canonical orbitals
    if system['dft']:

        rep_idx, mo_dft_can = np.arange(mol.nocc), mf_dft.mo_coeff
        e_dft, dip_dft = energy.e_tot(mol, system, 'dft_can', ao_dip, mo_dft_can[:, :mol.nocc], rep_idx, \
                                      alpha=dft.libxc.hybrid_coeff(system['xc_func']))

    else:

        e_dft = dip_dft = None

    # decompose dft energy by means of localized MOs
    if system['dft']:

        mo_dft_loc = orbitals.loc_orbs(mol, mf_dft.mo_coeff, s, system['loc_proc'])
        rep_idx, centres_dft = orbitals.reorder(mol, mf_dft, s, mo_dft_loc, pop=system['pop_scheme'])
        e_dft_loc, dip_dft_loc = energy.e_tot(mol, system, 'dft_loc', ao_dip, mo_dft_loc[:, :mol.nocc], rep_idx, \
                                              alpha=dft.libxc.hybrid_coeff(system['xc_func']))

    else:

        e_dft_loc = dip_dft_loc = centres_dft = None


    # sort results
    e_hf, dip_hf = results.sort_results(mol, system, 'hf_can', e_hf, dip_hf)[:2]
    e_hf_loc, dip_hf_loc, centres_hf = results.sort_results(mol, system, 'hf_loc', \
                                                            e_hf_loc, dip_hf_loc, centres=centres_hf)
    if system['dft']:
        e_dft, dip_dft = results.sort_results(mol, system, 'dft_can', e_dft, dip_dft)[:2]
        e_dft_loc, dip_dft_loc, centres_dft = results.sort_results(mol, system, 'dft_loc', \
                                                                   e_dft_loc, dip_dft_loc, centres=centres_dft)


    # print results
    results.print_results(mol, system, e_hf, dip_hf, e_hf_loc, dip_hf_loc, \
                          e_dft, dip_dft, e_dft_loc, dip_dft_loc, \
                          centres_hf, centres_dft, e_nuc, dip_nuc, e_xc, \
                          e_hf_tot, dip_hf_tot, e_dft_tot, dip_dft_tot)


if __name__ == '__main__':
    main()


