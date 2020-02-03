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
import re
import shutil
import ast
import numpy as np
from pyscf import gto, scf, dft
from typing import List, Dict, Union, Any

import orbitals
import energy
import results
import tools


def main():
    """ main program """
    # set system info
    system = SystemCls()
    system.atom, system.param = _set_param(system.param)
    if 'xc_func' not in system.param.keys():
        system.param['dft'] = False
    if 'cube' not in system.param.keys():
        system.param['cube'] = False

    # rm out dir if present
    if os.path.isdir(results.OUT):
        shutil.rmtree(results.OUT, ignore_errors=True)

    # make main out dir
    os.mkdir(results.OUT)
    if system.param['cube']:
        # make hf out dirs
        os.mkdir(results.OUT + '/hf_can')
        os.mkdir(results.OUT + '/hf_loc')
        # make dft out dirs
        if system['dft']:
            os.mkdir(results.OUT + '/dft_can')
            os.mkdir(results.OUT + '/dft_loc')

    # init logger
    sys.stdout = tools.Logger(results.RES_FILE)

    # init molecule
    mol = gto.Mole()
    mol.build(verbose = 0, output = None, atom = system.atom, basis = system.param['basis'], symmetry = True)

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
    if system.param['dft']:

        mf_dft = dft.RKS(mol)
        mf_dft.xc = system.param['xc_func']
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
    e_hf, dip_hf = energy.e_tot(mol, 'hf_can', ao_dip, mo_hf_can[:, :mol.nocc], rep_idx, system.param['cube'])

    # decompose hf energy by means of localized MOs
    mo_hf_loc = orbitals.loc_orbs(mol, mf_hf.mo_coeff, s, system.param['loc_proc'])
    rep_idx, centres_hf = orbitals.reorder(mol, mf_hf, s, mo_hf_loc, pop=system.param['pop_scheme'])
    e_hf_loc, dip_hf_loc = energy.e_tot(mol, 'hf_loc', ao_dip, mo_hf_loc[:, :mol.nocc], rep_idx, system.param['cube'])

    # decompose dft energy by means of canonical orbitals
    if system.param['dft']:

        rep_idx, mo_dft_can = np.arange(mol.nocc), mf_dft.mo_coeff
        e_dft, dip_dft = energy.e_tot(mol, 'dft_can', ao_dip, mo_dft_can[:, :mol.nocc], rep_idx, system.param['cube'], \
                                      alpha=dft.libxc.hybrid_coeff(system.param['xc_func']))

    else:

        e_dft = dip_dft = None

    # decompose dft energy by means of localized MOs
    if system.param['dft']:

        mo_dft_loc = orbitals.loc_orbs(mol, mf_dft.mo_coeff, s, system.param['loc_proc'])
        rep_idx, centres_dft = orbitals.reorder(mol, mf_dft, s, mo_dft_loc, pop=system.param['pop_scheme'])
        e_dft_loc, dip_dft_loc = energy.e_tot(mol, system, 'dft_loc', ao_dip, mo_dft_loc[:, :mol.nocc], rep_idx, system.param['cube'], \
                                              alpha=dft.libxc.hybrid_coeff(system.param['xc_func']))

    else:

        e_dft_loc = dip_dft_loc = centres_dft = None


    # sort results
    e_hf, dip_hf = results.sort(mol, 'hf_can', e_hf, dip_hf, system.param['cube'])[:2]
    e_hf_loc, dip_hf_loc, centres_hf = results.sort(mol, 'hf_loc', e_hf_loc, dip_hf_loc, \
                                                    system.param['cube'], centres=centres_hf)
    if system.param['dft']:
        e_dft, dip_dft = results.sort(mol, 'dft_can', e_dft, dip_dft, system.param['cube'])[:2]
        e_dft_loc, dip_dft_loc, centres_dft = results.sort(mol, 'dft_loc', e_dft_loc, dip_dft_loc, \
                                                           system.param['cube'], centres=centres_dft)


    # print results
    results.main(mol, system, e_hf, dip_hf, e_hf_loc, dip_hf_loc, \
                 e_dft, dip_dft, e_dft_loc, dip_dft_loc, \
                 centres_hf, centres_dft, e_nuc, dip_nuc, e_xc, \
                 e_hf_tot, dip_hf_tot, e_dft_tot, dip_dft_tot)


class SystemCls(object):
        """
        this class contains all system attributes
        """
        def __init__(self) -> None:
                """
                init molecule attributes
                """
                # set defaults
                self.atom: Union[List[str], str] = ''
                self.param: Dict[str, Any] = {'basis': 'sto-3g', 'loc_proc': 'pm', 'pop_scheme': 'mulliken'}


def _set_param(param):
        """
        this function sets system parameter attributes from input file
        """
        # read input file
        try:
            with open(os.getcwd()+'/input') as f:
                content = f.readlines()
                for i in range(len(content)):
                    if content[i].strip():
                        if content[i].split()[0][0] == '#':
                            continue
                        elif re.split('=',content[i])[0].strip() == 'atom':
                            atom = ''
                            for j in range(i+1, len(content)):
                                if content[j][:3] == "'''" or content[j][:3] == '"""':
                                    break
                                else:
                                    atom += content[j]
                        elif re.split('=',content[i])[0].strip() == 'param':
                            try:
                                inp = ast.literal_eval(re.split('=',content[i])[1].strip())
                            except ValueError:
                                raise ValueError('wrong input -- error in reading in param dictionary')
                            # update system
                            param = {**param, **inp}
        except IOError:
            sys.stderr.write('\nIOError : input file not found\n\n')
            raise
        return atom, param


if __name__ == '__main__':
    main()


