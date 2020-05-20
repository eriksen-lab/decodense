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
    'xc': 'pbe0',
    'loc': 'ibo-2',
    'pop': 'iao',
    'part': 'atoms'
}
UNIT = 'au'
N_ATOMS = 3
RST_FREQ = 50

# input / output
INPUT = os.getcwd() + '/water_therm_1000.mat'
OUTPUT = os.getcwd() + '/{:}_{:}_{:}_{:}_{:}_{:}/'.format(PARAMS['prop'], PARAMS['xc'] if PARAMS['xc'] != '' else 'hf', \
                                                          PARAMS['basis'], PARAMS['loc'] if PARAMS['loc'] != '' else 'can', \
                                                          PARAMS['pop'], PARAMS['part'], PARAMS['prop'])

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
            # number of slaves and tasks
            n_slaves = size - 1
            n_tasks = data['R'].shape[0]

            # start_idx
            if restart:
                res_el = np.load(OUTPUT + 'elec.npy')
                res_nuc = np.load(OUTPUT + 'nuc.npy')
                start_idx = np.argmax(res_el[:, 0] == 0.)
            else:
                res_el = np.zeros([n_tasks, N_ATOMS], dtype=np.float64)
                res_nuc = np.zeros([n_tasks, N_ATOMS], dtype=np.float64)
                start_idx = 0

            # loop over molecules in data set
            for mol_idx, mol_geo in enumerate(data['R'][start_idx:], start_idx):

                # probe for available slaves
                comm.Probe(source=MPI.ANY_SOURCE, tag=1, status=stat)
                # receive slave results
                res = comm.recv(source=stat.source, tag=1)
                # retrieve results
                if res is not None:
                    res_el[res['idx']] = res['prop_el']
                    res_nuc[res['idx']] = res['prop_nuc']
                    if res['idx'] % RST_FREQ == 0:
                        # save results
                        np.save(OUTPUT + 'elec', res_el)
                        np.save(OUTPUT + 'nuc', res_nuc)
                        # print status
                        prog = (res['idx'] + 1) / n_tasks
                        status = int(round(50 * prog))
                        remainder = (50 - status)
                        print(' STATUS:   [{:}]   ---  {:>6.2f} %'.format('#' * status + '-' * remainder, prog * 100.))


                # send mol_dict to slave
                comm.send({'idx': mol_idx, \
                           'struct': [[int(z), mol_geo[i]] for i, z in enumerate(data['Z'][mol_idx]) if 0. < z]}, \
                          dest=stat.source, tag=2)

            # done with all tasks
            while n_slaves > 0:

                # probe for available slaves
                comm.Probe(source=MPI.ANY_SOURCE, tag=1, status=stat)
                # receive slave results
                res = comm.recv(source=stat.source, tag=1)
                # save results
                if res is not None:
                    res_el[res['idx']] = res['prop_el']
                    res_nuc[res['idx']] = res['prop_nuc']
                    if res['idx'] % RST_FREQ == 0:
                        np.save(OUTPUT + 'elec', res_el)
                        np.save(OUTPUT + 'nuc', res_nuc)

                # send exit signal to slave
                comm.send(None, dest=stat.source, tag=2)
                # remove slave
                n_slaves -= 1

            # save final results
            np.save(OUTPUT + 'elec', res_el)
            np.save(OUTPUT + 'nuc', res_nuc)
            # print final status
            print(' STATUS:   [{:}]   ---  {:>6.2f} %'.format('#' * 50 + '-' * 0, 100.))
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
                    res = decodense.main(mol, decomp)
                    # send results to master
                    comm.send({'idx': mol_dict['idx'], 'prop_nuc': res['prop_nuc'], 'prop_el': res['prop_el']}, dest=0, tag=1)
                else:
                    # exit
                    break

        # barrier
        comm.Barrier()


if __name__ == '__main__':
    main()


