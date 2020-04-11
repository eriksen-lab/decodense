#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
import os.path
import re
import scipy.io as sio
import numpy as np
from mpi4py import MPI
from pyscf import gto

import decodense

# input / output
INPUT = os.getcwd() + '/qm7.mat'
OUTPUT = os.getcwd() + '/output/'
UNIT = 'bohr'

# decodense variables
PARAMS = {
    'basis': 'ccpvdz',
    'xc': 'pbe0',
    'loc': 'ibo-2',
    'pop': 'iao',
    'prop': 'energy',
}

def main():
        """
        main program
        """
        # mpi attributes
        comm = MPI.COMM_WORLD
        stat = MPI.Status()
        rank = comm.Get_rank()
        size = comm.Get_size()
        assert 1 < size, 'run.py must be run in parallel: `mpiexec -np N run.py`'

        # init decomp object
        decomp = decodense.DecompCls(**PARAMS)

        # master
        if rank == 0:

            # write MPI parameters
            print('\n MPI global size = {:}\n'.format(size))

            # make output dir
            if not os.path.isdir(OUTPUT):
                restart = False
                os.mkdir(OUTPUT)
            else:
                restart = True
            # load in dataset
            data = sio.loadmat(INPUT)
            # number of slaves and tasks
            n_slaves = size - 1
            n_tasks = data['R'].shape[0]

            # start_idx
            if restart:
                results = np.array([int(i) for j in sorted(os.listdir(OUTPUT)) for i in re.findall('(\d+)', j)])
                assert results.size % 3 == 0, 'restart error: invalid number of *_el.npy, *_tot.npy, and *_atom.npy files'
                start_idx = np.argmax(np.ediff1d(np.unique(results))) + 1
                if start_idx == 1:
                    start_idx += np.max(np.unique(results))
            else:
                start_idx = 0

            # loop over molecules in data set
            for mol_idx, mol_geo in enumerate(data['R'][start_idx:], start_idx):

                # probe for available slaves
                comm.Probe(source=MPI.ANY_SOURCE, tag=1, status=stat)
                # receive slave results
                res = comm.recv(source=stat.source, tag=1)
                # save results
                if res is not None:
                    np.save(OUTPUT + str(res['idx']) + '_el', res['prop_el'])
                    np.save(OUTPUT + str(res['idx']) + '_tot', res['prop_tot'])
                    if PARAMS['prop'] == 'energy':
                        np.save(OUTPUT + str(res['idx']) + '_atom', res['prop_atom'])

                # send mol_dict to slave
                comm.send({'idx': mol_idx, \
                           'struct': [[int(z), mol_geo[i]] for i, z in enumerate(data['Z'][mol_idx]) if 0. < z]}, \
                          dest=stat.source, tag=2)

                # print status
                prog = (mol_idx + 1) / n_tasks
                status = int(round(50 * prog))
                remainder = (50 - status)
                print(' STATUS:   [{:}]   ---  {:>6.2f} %'.format('#' * status + '-' * remainder, prog * 100.))

            # done with all tasks
            while n_slaves > 0:

                # probe for available slaves
                comm.Probe(source=MPI.ANY_SOURCE, tag=1, status=stat)
                # receive slave results
                res = comm.recv(source=stat.source, tag=1)
                # save results
                if res is not None:
                    np.save(OUTPUT + str(res['idx']) + '_el', res['prop_el'])
                    np.save(OUTPUT + str(res['idx']) + '_tot', res['prop_tot'])
                    if PARAMS['prop'] == 'energy':
                        np.save(OUTPUT + str(res['idx']) + '_atom', res['prop_atom'])

                # send exit signal to slave
                comm.send(None, dest=stat.source, tag=2)
                # remove slave
                n_slaves -= 1

            # write final info
            with open(OUTPUT + 'info.txt', 'w') as f_info:
                f_info.write(decodense.info(decomp))

        else: # slaves

            # send availability to master
            comm.send(None, dest=0, tag=1)

            # receive work from master
            while True:

                # receive mol_dict
                mol_dict = comm.recv(source=0, tag=2)
                # perform task
                if mol_dict is not None:
                    # init molecule
                    mol = gto.M(verbose = 0, output = None, unit = UNIT, basis = PARAMS['basis'], atom = mol_dict['struct'])
                    # decodense calc
                    e_calc = decodense.main(mol, decomp)
                    # send results to master
                    if PARAMS['prop'] == 'energy':
                        # atomic energies
                        e_atom = np.array([decodense.atom_energies[PARAMS['xc'].upper()] \
                                                                  [PARAMS['basis'].upper()] \
                                                                  [mol.atom_pure_symbol(atom)] for atom in range(mol.natm)])
                        comm.send({'idx': mol_dict['idx'], 'prop_el': e_calc['prop_el'], \
                                   'prop_tot': e_calc['prop_tot'], 'prop_atom': e_calc['prop_tot'] - e_atom}, dest=0, tag=1)
                    else:
                        comm.send({'idx': mol_dict['idx'], 'prop_el': e_calc['prop_el'], \
                                   'prop_tot': e_calc['prop_tot']}, dest=0, tag=1)
                else:
                    # exit
                    break


if __name__ == '__main__':
    main()


