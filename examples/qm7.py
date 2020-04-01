#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
import os.path
import re
import numpy as np
from mpi4py import MPI
from pyscf import gto

import decodense

# input / output
STR_LENGTH = 4 # e.g., 0001, 0010, 0100, 1000
INPUT = os.getcwd() + '/qm7/'
OUTPUT = os.getcwd() + '/qm7_out/'

# decodense variables
BASIS = 'ccpvdz'
XC = 'pbe0'
#XC = 'wb97x_d'
LOC = 'ibo-2'
#LOC = 'ibo-4'
PROP = 'energy'
#PROP = 'dipole'

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
        decomp = decodense.DecompCls(xc = XC, basis = BASIS, loc = LOC, prop = PROP)

        # master
        if rank == 0:

            # make output dir
            if not os.path.isdir(OUTPUT):
                os.mkdir(OUTPUT)
            # full list of molecules
            molecules = np.array([int(i) for j in sorted(os.listdir(INPUT)) for i in re.findall('(\d+)', j)])
            # number of slaves and tasks
            n_slaves = size - 1
            n_tasks = molecules.size

            # start_idx
            results = np.array([int(i) for j in sorted(os.listdir(OUTPUT)) for i in re.findall('(\d+)', j)])
            rst_idx = np.setdiff1d(molecules, results)[0]
            assert results.size % 3 == 0, 'restart error: invalid number of *_el.npy, *_tot.npy, and *_atom.npy files'
            start_idx = np.where(molecules == rst_idx)[0][0]

            # loop over molecules in data set
            for mol_idx, mol_name in enumerate(molecules[start_idx:], start_idx):

                # recast mol_name as string with padded zeros
                mol_str = str(mol_name).zfill(STR_LENGTH)

                # probe for available slaves
                comm.Probe(source=MPI.ANY_SOURCE, tag=1, status=stat)
                # receive slave results
                res = comm.recv(source=stat.source, tag=1)
                # save results
                if res is not None:
                    np.save(OUTPUT + res['name'] + '_el', res['prop_el'])
                    np.save(OUTPUT + res['name'] + '_tot', res['prop_tot'])
                    if PROP == 'energy':
                        np.save(OUTPUT + res['name'] + '_atom', res['prop_atom'])

                # send mol_dict to slave
                comm.send({'name': mol_str, 'struct': gto.format_atom(INPUT + mol_str + '.xyz')}, dest=stat.source, tag=2)

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
                    if PROP == 'energy':
                        np.save(OUTPUT + res['name'] + '_atom', res['prop_atom'])

                # send exit signal to slave
                comm.send(None, dest=stat.source, tag=2)
                # remove slave
                n_slaves -= 1

            # write final info
            with open(OUTPUT + 'info.txt', 'w') as f_info:
                f_info.write(decodense.table_info(decomp))

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
                    mol = gto.M(verbose = 0, output = None, unit = 'bohr', basis = BASIS, atom = mol_dict['struct'])
                    # decodense calc
                    e_calc = decodense.main(mol, decomp)
                    # send results to master
                    if PROP == 'energy':
                        # atomic energies
                        e_atom = np.array([decodense.atom_energies[XC.upper()][BASIS.upper()][mol.atom_pure_symbol(atom)] for atom in range(mol.natm)])
                        comm.send({'name': mol_dict['name'], 'prop_el': e_calc['prop_el'], \
                                   'prop_tot': e_calc['prop_tot'], 'prop_atom': e_calc['prop_tot'] - e_atom}, dest=0, tag=1)
                    else:
                        comm.send({'name': mol_dict['name'], 'prop_el': e_calc['prop_el'], \
                                   'prop_tot': e_calc['prop_tot']}, dest=0, tag=1)
                else:
                    # exit
                    break

        # final barrier
        comm.Barrier()
        MPI.Finalize()


if __name__ == '__main__':
    main()


