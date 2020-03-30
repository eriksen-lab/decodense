#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
import os.path
import numpy as np
from mpi4py import MPI
from pyscf import gto

import decodense

# input / output
INPUT = os.getcwd() + '/qm7/'
OUTPUT = os.getcwd() + '/qm7_out/'
# decodense variables
BASIS = 'ccpvdz'
XC = 'pbe0'

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

        # master
        if rank == 0:

            # restart
            if not os.path.isdir(OUTPUT):
                os.mkdir(OUTPUT)
                restart = False
            else:
                restart = True
            # full list of molecules
            molecules = sorted(os.listdir(INPUT))
            # number of slaves and tasks
            n_slaves = size - 1
            n_tasks = len(molecules)

            # start_idx
            n_results = len(os.listdir(OUTPUT))
            assert n_results % 3 == 0, 'restart error: each structure must have an *_el.npy, *_tot.npy, and *_atom.npy file'
            start_idx = n_results // 3

            # loop over molecules in data set
            for mol_idx, mol_name in enumerate(molecules[start_idx:], start_idx):

                # probe for available slaves
                comm.Probe(source=MPI.ANY_SOURCE, tag=1, status=stat)
                # receive slave results
                res = comm.recv(source=stat.source, tag=1)
                # save results
                if res is not None:
                    np.save(OUTPUT + res['name'] + '_el', res['prop_el'])
                    np.save(OUTPUT + res['name'] + '_tot', res['prop_tot'])
                    np.save(OUTPUT + res['name'] + '_atom', res['prop_atom'])

                # send mol_dict to slave
                comm.send({'name': mol_name, 'struct': gto.format_atom(INPUT + mol_name)}, dest=stat.source, tag=2)

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
                    np.save(OUTPUT + res['name'] + '_el', res['prop_el'])
                    np.save(OUTPUT + res['name'] + '_tot', res['prop_tot'])
                    np.save(OUTPUT + res['name'] + '_atom', res['prop_atom'])

                # send exit signal to slave
                comm.send(None, dest=stat.source, tag=2)
                # remove slave
                n_slaves -= 1

        else: # slaves

            # init decomp object
            decomp = decodense.DecompCls(xc = XC)

            # send availability to master
            comm.send(None, dest=0, tag=1)

            # receive work from master
            while True:

                # receive mol_dict
                mol_dict = comm.recv(source=0, tag=2)
                # perform task
                if mol_dict is not None:
                    # init molecule
                    mol = gto.M(verbose = 0, output = None, unit = 'bohr', basis = BASIS, atom = mol_dict['struct'])
                    # decodense calc
                    e_calc = decodense.main(mol, decomp)
                    # atomic energies
                    e_atom = np.array([decodense.atom_energies[XC.upper()][BASIS.upper()][mol.atom_pure_symbol(atom)] for atom in range(mol.natm)])
                    # send results to master
                    comm.send({'name': mol_dict['name'][:-4], 'prop_el': e_calc['prop_el'], \
                               'prop_tot': e_calc['prop_tot'], 'prop_atom': e_calc['prop_tot'] - e_atom}, dest=0, tag=1)
                else:
                    # exit
                    break


if __name__ == '__main__':
    main()


