#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
main mf_decomp program

usage:
python main.py `molecule` `xc_functional` `localization_procedure`
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import numpy as np
from pyscf import gto, scf, dft

import orbitals
import energy
import results


def main():
    """ main program """

    # read in molecule argument
    if len(sys.argv) != 4:
        raise SyntaxError('\n missing or too many arguments: python main.py `molecule` `xc_functional` `localization_procedure`\n')

    # set system info
    system = {}
    system['molecule'] = sys.argv[1]
    system['xc_func'] = sys.argv[2]
    system['loc_proc'] = sys.argv[3]


    # init molecule
    mol = gto.Mole()
    mol.build(
    verbose = 0,
    output = None,
    atom = open('../structures/'+system['molecule']+'.xyz').read(),
    basis = '631g',
    symmetry = True,
    )

    # singlet check
    assert mol.spin == 0, 'decomposition scheme only implemented for singlet states'


    # molecular dimensions
    mol.ncore = orbitals.set_ncore(mol)
    mol.nocc = mol.nelectron // 2


    # overlap matrix
    s = mol.intor_symmetric('int1e_ovlp')


    # init and run HF calc
    mf_hf = scf.RHF(mol)
    mf_hf.conv_tol = 1.0e-12
    mf_hf.run()
    assert mf_hf.converged, 'HF not converged'

    # init and run DFT (B3LYP) calc
    mf_dft = dft.RKS(mol)
    mf_dft.xc = system['xc_func']
    mf_dft.conv_tol = 1.0e-12
    mf_dft.run()
    assert mf_hf.converged, 'DFT not converged'


    # nuclear repulsion energy
    e_nuc = np.sum(energy.energy_nuc(mol))
    # energy of XC functional evaluated on a grid
    e_xc = mf_dft._numint.nr_rks(mol, mf_dft.grids, mf_dft.xc, \
                                 mf_dft.make_rdm1(mf_dft.mo_coeff, mf_dft.mo_occ))[1]


    # decompose HF energy by means of canonical orbitals
    mo_coeff = mf_hf.mo_coeff
    e_hf = energy.e_tot(mol, mf_hf, s, mo_coeff)[0]

    # decompose HF energy by means of localized MOs
    mo_coeff = orbitals.loc_orbs(mol, mf_hf, s, system['loc_proc'])
    e_hf_loc, centres_hf = energy.e_tot(mol, mf_hf, s, mo_coeff)
    e_hf_loc, centres_hf = results.sort_results(e_hf_loc, centres_hf)

    # decompose DFT energy by means of canonical orbitals
    mo_coeff = mf_dft.mo_coeff
    e_dft = energy.e_tot(mol, mf_dft, s, mo_coeff, dft=True)[0]

    # decompose DFT energy by means of localized MOs
    mo_coeff = orbitals.loc_orbs(mol, mf_dft, s, system['loc_proc'])
    e_dft_loc, centres_dft = energy.e_tot(mol, mf_dft, s, mo_coeff, dft=True)
    e_dft_loc, centres_dft = results.sort_results(e_dft_loc, centres_dft)

    results.print_results(system, mol, e_hf, e_hf_loc, e_dft, e_dft_loc, centres_hf, centres_dft, \
                          e_nuc, e_xc, mf_hf.e_tot, mf_dft.e_tot)


if __name__ == '__main__':
    main()


