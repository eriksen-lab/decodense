#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
results module containing all functions related to printing the results of an mf_decomp calculation
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import os
import numpy as np
import math
from pyscf import gto, lib
from pyscf import tools as pyscf_tools

import tools


def sort_results(mol, mo_can, mo_loc, e_can, e_loc, dip_can, dip_loc, orb_type, system, centres_loc):
    """
    this function returns sorted results for unique bonds only

    :param mol: pyscf mol object
    :param mo_can: canonical mo coefficients. numpy array of shape (n_orb, n_orb)
    :param mo_loc: localized mo coefficients. numpy array of shape (n_orb, n_orb)
    :param e_can: decomposed mean-field canonical energy contributions. numpy array of (nocc,)
    :param e_loc: decomposed mean-field local energy contributions. numpy array of (nocc,)
    :param dip_can: decomposed mean-field canonical dipole moment contributions. numpy array of (nocc, 3)
    :param dip_loc: decomposed mean-field local dipole moment contributions. numpy array of (nocc, 3)
    :param orb_type: type of orbitals. string
    :param system: system information. dict
    :param centres_loc: corresponding charge centres. numpy array of shape (nocc, 2)
    :return: numpy array of shape (nunique,) [e_orb_unique],
             numpy array of shape (nunique, 3) [dip_orb_unique],
             numpy array of shape (nunique, 2) [centres_unique]
    """
    # set output paths
    if system['cube']:
        if orb_type == 'hf':
            out_can_path = system['out_hf_can_path']
            out_loc_path = system['out_hf_loc_path']
        else:
            out_can_path = system['out_dft_can_path']
            out_loc_path = system['out_dft_loc_path']

    # sort canonical results
    sort_can = np.argsort(e_can)
    dip_can = dip_can[sort_can]
    e_can = np.sort(e_can)

    # rdm1 canonical cube files
    if system['cube']:
        for i in range(mol.nocc):

            # get orbital
            orb = mo_can[:, i].reshape(mol.norb, 1)

            # orbital-specific rdm1
            rdm1_orb = np.einsum('ip,jp->ij', orb, orb) * 2.

            # write cube file
            pyscf_tools.cubegen.density(mol, out_can_path + '/rdm1_{:}_can_{:}_tmp.cube'.format(orb_type, i), rdm1_orb)

        # sort canonical cube files
        for i, j in enumerate(sort_can):
            os.rename(out_can_path + '/rdm1_{:}_can_{:}_tmp.cube'.format(orb_type, j), \
                      out_can_path + '/rdm1_{:}_can_{:}.cube'.format(orb_type, i))

    # search for the unique centres for local results
    centres_loc_unique = np.unique(centres_loc, axis=0)

    # repetitive centres
    rep_idx = [np.where((centres_loc == i).all(axis=1))[0] for i in centres_loc_unique]

    # e_loc_unique
    e_loc_unique = np.array([np.sum(e_loc[rep_idx[i]]) for i in range(len(rep_idx))], dtype=np.float64)
    # dip_loc_unique
    dip_loc_unique = np.array([np.sum(dip_loc[np.asarray(rep_idx[i])[None, :]], axis=1).reshape(-1) \
                                for i in range(len(rep_idx))], dtype=np.float64)

    # sort arrays wrt e_orb_unique
    sort_loc = np.argsort(e_loc_unique)
    centres_loc_unique = centres_loc_unique[sort_loc]
    dip_loc_unique = dip_loc_unique[sort_loc]
    e_loc_unique = np.sort(e_loc_unique)

    # rdm1 local cube files
    if system['cube']:
        for i, j in enumerate(rep_idx):

            # get orbital
            orb = mo_loc[:, j].reshape(mol.norb, -1)

            # orbital-specific rdm1
            rdm1_orb = np.einsum('ip,jp->ij', orb, orb) * 2.

            # write cube file
            pyscf_tools.cubegen.density(mol, out_loc_path + '/rdm1_{:}_loc_{:}_tmp.cube'.format(orb_type, i), rdm1_orb)

        # sort local cube files
        for i, j in enumerate(sort_loc):
            os.rename(out_loc_path + '/rdm1_{:}_loc_{:}_tmp.cube'.format(orb_type, j), \
                      out_loc_path + '/rdm1_{:}_loc_{:}.cube'.format(orb_type, i))

    return e_can, e_loc_unique, dip_can, dip_loc_unique, centres_loc_unique


def print_results(mol, system, e_hf, dip_hf, e_hf_loc, dip_hf_loc, \
                  e_dft, dip_dft, e_dft_loc, dip_dft_loc, \
                  centres_hf, centres_dft, e_nuc, dip_nuc, e_xc, \
                  e_hf_ref, dip_hf_ref, e_dft_ref, dip_dft_ref):
    """
    this function prints the results of an mf_decomp calculation

    :param mol: pyscf mol object
    :param system: system information. dict
    :param e_hf: canonical hf decomposed energy results. numpy array of shape (nocc,)
    :param dip_hf: canonical hf decomposed dipole results. numpy array of shape (nocc, 3)
    :param e_hf_loc: localized hf decomposed energy results. numpy array of shape (nocc,)
    :param dip_hf_loc: localized hf decomposed dipole results. numpy array of shape (nocc, 3)
    :param e_dft: canonical dft decomposed energy results. numpy array of shape (nocc,)
    :param dip_dft: canonical dft decomposed dipole results. numpy array of shape (nocc, 3)
    :param e_dft_loc: localized dft decomposed results. numpy array of shape (nocc,)
    :param dip_dft_loc: localized dft decomposed dipole results. numpy array of shape (nocc, 3)
    :param centres_hf: centre assignments for localized hf results. numpy array of shape (nocc, 2) [*strings]
    :param centres_dft: centre assignments for localized dft results. numpy array of shape (nocc, 2) [*strings]
    :param e_nuc: nuclear repulsion energy. scalar
    :param dip_nuc: nuclear dipole moment. numpy array of shape (3,)
    :param e_xc: exchange-correlation energy. scalar
    :param e_hf_ref: reference hf energy. scalar
    :param dip_hf_ref: reference hf dipole moment. numpy array of shape (3,)
    :param e_dft_ref: reference dft energy. scalar
    :param dip_dft_ref: reference dft dipole moment. numpy array of shape (3,)
    """
    # system info
    print('\n\n system info:')
    print(' ------------')
    print(' molecule          = {:}'.format(system['molecule']))
    print(' point group       = {:}'.format(mol.groupname))
    print(' basis set         = {:}'.format(system['basis']))
    print('\n localization      = {:}'.format(system['loc_proc']))
    print(' assignment        = {:}'.format(system['pop_scheme']))
    if system['dft']:
        print(' xc functional     = {:}'.format(system['xc_func']))
    print('\n electrons         = {:}'.format(mol.nelectron))
    print(' occupied orbitals = {:}'.format(mol.nocc))
    print(' virtual orbitals  = {:}'.format(mol.nvirt))
    print(' total orbitals    = {:}'.format(mol.norb))


    # print git version
    print('\n git version: {:}\n\n'.format(tools.git_version()))


    # get inter-atomic distance array
    rr = gto.mole.inter_distance(mol) * lib.param.BOHR


    # print hf results
    print('\n\n ** hartree-fock')
    print(' ---------------\n')

    # energy
    print('------------------------------------------------------------------------')
    print('{:^70}'.format('ground-state energy'))
    print('------------------------------------------------------------------------')
    print('  MO  |   canonical   |   localized   |     atom(s)   |   bond length')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    for i in range(mol.nocc):

        if i < e_hf_loc.size:

           # core or valence orbital(s)
            core = centres_hf[i, 0] == centres_hf[i, 1]

            print('  {:>2d}  | {:>10.3f}    | {:>10.3f}    |{:^15s}| {:>10s}'. \
                    format(i, e_hf[i], e_hf_loc[i], \
                           mol.atom_symbol(centres_hf[i, 0]) if core else '{:s} & {:s}'. \
                           format(mol.atom_symbol(centres_hf[i, 0]), mol.atom_symbol(centres_hf[i, 1])), \
                           '' if core else '{:>.3f}'. \
                            format(rr[centres_hf[i, 0], centres_hf[i, 1]])))

        else:

            print('  {:>2d}  | {:>10.3f}    |'. \
                    format(i, e_hf[i]))

    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('  sum | {:>10.3f}    | {:>10.3f}    |'. \
            format(np.sum(e_hf), np.sum(e_hf_loc)))
    print('------------------------------------------------------------------------')
    print('  nuc | {:>+10.3f}    | {:>+10.3f}    |'. \
            format(e_nuc, e_nuc))
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('  tot | {:>12.5f}  | {:>12.5f}  |'. \
            format(np.sum(e_hf) + e_nuc, np.sum(e_hf_loc) + e_nuc))
    print('\n *** HF reference energy = {:.5f}\n\n'. \
            format(e_hf_ref))

    # dipole moment
    print('----------------------------------------------------------------------------------------------------------------')
    print('{:^100}'.format('ground-state dipole moment'))
    print('----------------------------------------------------------------------------------------------------------------')
    print('  MO  |             canonical             |            localized              |     atom(s)   |   bond length')
    print('----------------------------------------------------------------------------------------------------------------')
    print('      |     x     /     y     /     z     |     x     /     y     /     z     |')
    print('----------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------------------')
    for i in range(mol.nocc):

        if i < dip_hf_loc.shape[0]:

           # core or valence orbital(s)
            core = centres_hf[i, 0] == centres_hf[i, 1]

            print('  {:>2d}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |{:^15s}| {:>10s}'. \
                    format(i, *dip_hf[i] + 1.0e-10, *dip_hf_loc[i] + 1.0e-10, \
                           mol.atom_symbol(centres_hf[i, 0]) if core else '{:s} & {:s}'. \
                           format(mol.atom_symbol(centres_hf[i, 0]), mol.atom_symbol(centres_hf[i, 1])), \
                           '' if core else '{:>.3f}'. \
                            format(rr[centres_hf[i, 0], centres_hf[i, 1]])))

        else:

            print('  {:>2d}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |'. \
                    format(i, *dip_hf[i] + 1.0e-10))

    print('----------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------------------')
    print('  sum | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |'. \
            format(*np.fromiter(map(math.fsum, dip_hf.T), dtype=dip_hf.dtype, count=dip_hf.shape[1]) + 1.0e-10, \
                   *np.fromiter(map(math.fsum, dip_hf_loc.T), dtype=dip_hf_loc.dtype, count=dip_hf_loc.shape[1]) + 1.0e-10))
    print('----------------------------------------------------------------------------------------------------------------')
    print('  nuc | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |'. \
            format(*dip_nuc + 1.0e-10, *dip_nuc + 1.0e-10))
    print('----------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------------------------------')
    print('  tot | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |'. \
            format(*(dip_nuc - np.fromiter(map(math.fsum, dip_hf.T), dtype=dip_hf.dtype, count=dip_hf.shape[1])) + 1.0e-10, \
                   *(dip_nuc - np.fromiter(map(math.fsum, dip_hf_loc.T), dtype=dip_hf_loc.dtype, count=dip_hf_loc.shape[1])) + 1.0e-10))
    print('\n *** HF reference dipole moment = {:>8.3f}  / {:>8.3f}  / {:>8.3f}\n\n'. \
            format(*dip_hf_ref + 1.0e-10))


    # print dft results
    if system['dft']:

        # dipole moment
        print('\n\n ** dft')
        print(' ------\n')
        print('------------------------------------------------------------------------')
        print('{:^70}'.format('ground-state energy'))
        print('------------------------------------------------------------------------')
        print('  MO  |   canonical   |   localized   |     atom(s)   |   bond length')
        print('------------------------------------------------------------------------')
        print('------------------------------------------------------------------------')
        for i in range(mol.nocc):

            if i < e_dft_loc.size:

                # core or valence orbital(s)
                core = centres_dft[i, 0] == centres_dft[i, 1]

                print('  {:>2d}  | {:>10.3f}    | {:>10.3f}    |{:^15s}| {:>10s}'. \
                        format(i, e_dft[i], e_dft_loc[i], \
                               mol.atom_symbol(centres_dft[i, 0]) if core else '{:s} & {:s}'. \
                               format(mol.atom_symbol(centres_dft[i, 0]), mol.atom_symbol(centres_dft[i, 1])), \
                               '' if core else '{:>.3f}'. \
                                format(rr[centres_dft[i, 0], centres_dft[i, 1]])))

            else:

                print('  {:>2d}  | {:>10.3f}    |'. \
                        format(i, e_dft[i]))

        print('------------------------------------------------------------------------')
        print('------------------------------------------------------------------------')
        print('  sum | {:>10.3f}    | {:>10.3f}    |'. \
                format(np.sum(e_dft), np.sum(e_dft_loc)))
        print('------------------------------------------------------------------------')
        print('  nuc | {:>+10.3f}    | {:>+10.3f}    |'. \
                format(e_nuc, e_nuc))
        print('  xc  | {:>+10.3f}    | {:>+10.3f}    |'. \
                format(e_xc, e_xc))
        print('------------------------------------------------------------------------')
        print('------------------------------------------------------------------------')
        print('  tot | {:>12.5f}  | {:>12.5f}  |'. \
                format(np.sum(e_dft) + e_nuc + e_xc, np.sum(e_dft_loc) + e_nuc + e_xc))
        print('\n *** DFT reference energy = {:.5f}\n\n'. \
                format(e_dft_ref))

        # dipole moment
        print('----------------------------------------------------------------------------------------------------------------')
        print('{:^100}'.format('ground-state dipole moment'))
        print('----------------------------------------------------------------------------------------------------------------')
        print('  MO  |             canonical             |            localized              |     atom(s)   |   bond length')
        print('----------------------------------------------------------------------------------------------------------------')
        print('      |     x     /     y     /     z     |     x     /     y     /     z     |')
        print('----------------------------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------------------------')
        for i in range(mol.nocc):

            if i < dip_dft_loc.shape[0]:

               # core or valence orbital(s)
                core = centres_dft[i, 0] == centres_dft[i, 1]

                print('  {:>2d}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |{:^15s}| {:>10s}'. \
                        format(i, *dip_dft[i] + 1.0e-10, *dip_dft_loc[i] + 1.0e-10, \
                               mol.atom_symbol(centres_dft[i, 0]) if core else '{:s} & {:s}'. \
                               format(mol.atom_symbol(centres_dft[i, 0]), mol.atom_symbol(centres_dft[i, 1])), \
                               '' if core else '{:>.3f}'. \
                                format(rr[centres_dft[i, 0], centres_dft[i, 1]])))

            else:

                print('  {:>2d}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |'. \
                        format(i, *dip_dft[i] + 1.0e-10))

        print('----------------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------------')
        print('  sum | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |'. \
                format(*np.fromiter(map(math.fsum, dip_dft.T), dtype=dip_dft.dtype, count=dip_dft.shape[1]) + 1.0e-10, \
                       *np.fromiter(map(math.fsum, dip_dft_loc.T), dtype=dip_dft_loc.dtype, count=dip_dft_loc.shape[1]) + 1.0e-10))
        print('----------------------------------------------------------------------------------------------------')
        print('  nuc | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |'. \
                format(*dip_nuc + 1.0e-10, *dip_nuc + 1.0e-10))
        print('----------------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------------')
        print('  tot | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  | {:>8.3f}  / {:>8.3f}  / {:>8.3f}  |'. \
                format(*(dip_nuc - np.fromiter(map(math.fsum, dip_dft.T), dtype=dip_dft.dtype, count=dip_dft.shape[1])) + 1.0e-10, \
                       *(dip_nuc - np.fromiter(map(math.fsum, dip_dft_loc.T), dtype=dip_dft_loc.dtype, count=dip_dft_loc.shape[1])) + 1.0e-10))
        print('\n *** DFT reference dipole moment = {:>8.3f}  / {:>8.3f}  / {:>8.3f}\n\n'. \
                format(*dip_dft_ref + 1.0e-10))


