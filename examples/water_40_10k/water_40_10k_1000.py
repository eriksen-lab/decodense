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

# decodense variables
PARAMS = {
    'prop': 'energy',
    'basis': 'ccpvdz',
    'xc': '',
    'loc': 'ibo-2',
    'pop': 'iao',
    'part': 'atoms'
}
UNIT = 'ang'

# input / output
INPUT = os.getcwd() + '/water_40_10k.mat'
NUMBER = 1000
OUTPUT = os.getcwd() + '/{:}_{:}_{:}_{:}_{:}_{:}/'.format(PARAMS['prop'], PARAMS['xc'] if PARAMS['xc'] != '' else 'hf', \
                                                          PARAMS['basis'], PARAMS['loc'] if PARAMS['loc'] != '' else 'can', \
                                                          PARAMS['pop'], PARAMS['part'], PARAMS['prop'])

def randomize(a, seed):
    np.random.seed(seed)
    return a[np.random.permutation(a.shape[0])]

def main():
        """
        main program
        """
        # mpi attributes
        comm = MPI.COMM_WORLD
        stat = MPI.Status()
        rank = comm.Get_rank()
        size = comm.Get_size()
        assert 1 < size, 'script must be run in parallel: `mpiexec -np N ...`'

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
            # randomize geometries
            geometries = randomize(data['R'], 0)
            # number of slaves and tasks
            n_slaves = size - 1
            n_tasks = NUMBER

            # start_idx
            if restart:
                results = np.array([int(i) for j in sorted(os.listdir(OUTPUT)) for i in re.findall('(\d+)', j)])
                assert results.size % 2 == 0, 'restart error: invalid number of *_el.npy and *_tot.npy files'
                start_idx = np.argmax(np.ediff1d(np.unique(results))) + 1
                if start_idx == 1:
                    start_idx += np.max(np.unique(results))
            else:
                start_idx = 0

            # loop over molecules in data set
            for mol_idx, mol_geo in enumerate(geometries[start_idx:NUMBER], start_idx):

                # probe for available slaves
                comm.Probe(source=MPI.ANY_SOURCE, tag=1, status=stat)
                # receive slave results
                res = comm.recv(source=stat.source, tag=1)
                # save results
                if res is not None:
                    np.save(OUTPUT + str(res['idx']) + '_el', res['prop_el'])
                    np.save(OUTPUT + str(res['idx']) + '_tot', res['prop_tot'])

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
                    mol = gto.M(verbose = 0, output = None, unit = UNIT, \
                                basis = PARAMS['basis'], atom = mol_dict['struct'])
                    # decodense calc
                    e_calc = decodense.main(mol, decomp)
                    # send results to master
                    if PARAMS['prop'] == 'energy':
                        comm.send({'idx': mol_dict['idx'], 'prop_el': e_calc['prop_el'], \
                                   'prop_tot': e_calc['prop_nuc'] + e_calc['prop_el']}, dest=0, tag=1)
                    else:
                        comm.send({'idx': mol_dict['idx'], 'prop_el': e_calc['prop_el'], \
                                   'prop_tot': e_calc['prop_nuc'] - e_calc['prop_el']}, dest=0, tag=1)
                else:
                    # exit
                    break

        # barrier
        comm.Barrier()


if __name__ == '__main__':
    main()


