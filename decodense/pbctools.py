#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
pbc module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import numpy as np
from scipy.special import erf, erfc
#from typing import List, Tuple, Dict, Union, Any

import pyscf.lib
from pyscf.lib import logger
from pyscf.pbc import gto as pbc_gto  
from pyscf.pbc import scf as pbc_scf 


# almost identical to ewald in cell.py
# TODO: only works for 3D cells, extend to lower dim.
# TODO: only implemented for nuc-nuc repulsion
def ewald_e_nuc(cell: pbc_gto.Cell) -> np.ndarray:
    """
    this function returns the nuc-nuc repulsion energy for a cell
    """ 
    '''Perform real (R) and reciprocal (G) space Ewald sum for the energy,
       partitioned into atomic contributions.
    Formulation of Martin, App. F2.
    Returns:
        array of floats
            The Ewald energy consisting of overlap, self, and G-space sum.
    See Also:
        pyscf.pbc.gto.get_ewald_params
    '''
    def cut_mesh_for_ewald(cell, mesh):
        mesh = np.copy(mesh)
        mesh_max = np.asarray(np.linalg.norm(cell.lattice_vectors(), axis=1) * 2,
                              dtype=int)  # roughly 2 grids per bohr
        if (cell.dimension < 2 or
            (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
            mesh_max[cell.dimension:] = mesh[cell.dimension:]

        mesh_max[mesh_max<80] = 80
        mesh[mesh>mesh_max] = mesh_max[mesh>mesh_max]
        return mesh

    # If lattice parameter is not set, the cell object is treated as a mole
    # object. The nuclear repulsion energy is computed.
    if cell.a is None:
        return mole.energy_nuc(cell)

    if cell.natm == 0:
        return 0

    # get the Ewald 'eta' (exponent eta^2 of the model Gaussian charges) and 
    # 'cut' (real space cut-off) parameters 
    #if ew_eta is None: ew_eta = mol.get_ewald_params()[0]
    #if ew_cut is None: ew_cut = mol.get_ewald_params()[1]
    ew_eta = cell.get_ewald_params()[0]
    ew_cut = cell.get_ewald_params()[1]

    # atom coord: [a,b] a=atom, b=cart.coord
    chargs = cell.atom_charges()
    coords = cell.atom_coords()

    # (Cartesian, unitful) lattice translation vectors for nearby images
    # in bohr (prim. lattice vectors (cell.a) in Å)
    Lall = cell.get_lattice_Ls(rcut=ew_cut)

    # distances between atoms in cell 0 and nearby images
    # [L,i,j,d] where L is cell index; i is atom index in cell 0; 
    # j is atom index in cell L; d is cart. component
    rLij = coords[:,None,:] - coords[None,:,:] + Lall[:,None,None,:]
    # euclidean distances 
    # (n_neighb_cells x n_atoms x n_atoms)
    r = np.sqrt(np.einsum('Lijx,Lijx->Lij', rLij, rLij))
    rLij = None
    # "eliminate" self-distances -> self-terms skipped (R) sum? 
    r[r<1e-16] = 1e200
    
    # (R) Ewald sum (shape: n_atoms)
    ewovrl_atomic = .5 * np.einsum('i,j,Lij->i', chargs, chargs, erfc(ew_eta * r) / r)
    
    # Ewald self-term: cancels self-contribution in (G) sum
    # last line of Eq. (F.5) in Martin
    ewself_factor = -.5 * 2 * ew_eta / np.sqrt(np.pi)
    ewself_atomic = np.einsum('i,i->i', chargs,chargs)
    ewself_atomic = ewself_atomic.astype(float)
    ewself_atomic *= ewself_factor 
    if cell.dimension == 3:
        ewself_atomic += -.5 * (chargs*np.sum(chargs)).astype(float) * np.pi/(ew_eta**2 * cell.vol)

    # g-space sum (using g grid) (Eq. (F.6) in Electronic Structure by Richard M. Martin
    #, but note errors as below)
    # Eq. (F.6) in Martin is off by a factor of 2, the
    # exponent is wrong (8->4) and the square is in the wrong place
    #
    # Formula should be
    #   1/2 * 4\pi / Omega \sum_I \sum_{G\neq 0} |ZS_I(G)|^2 \exp[-|G|^2/4\eta^2]
    # where
    #   ZS_I(G) = \sum_a Z_a exp (i G.R_a)
    # See also Eq. (32) of ewald.pdf at
    #   http://www.fisica.uniud.it/~giannozz/public/ewald.pdf
    #
    # (g-grid) of reciprocal lattice vectors
    mesh = cut_mesh_for_ewald(cell, cell.mesh)
    Gv, Gvbase, Gv_weights = cell.get_Gv_weights(mesh)
    absG2 = np.einsum('gi,gi->g', Gv, Gv)
    # exclude the G=0 vector
    absG2[absG2==0] = 1e200

    if cell.dimension != 2 or cell.low_dim_ft_type == 'inf_vacuum':
        coulG = 4*np.pi / absG2
        # todo is this omega in eq.(5)?
        coulG *= Gv_weights
        # get_SI(k_vecs) gets the structure factors, n_atm*n_grid 
        # todo get them only once, save?
        ZSI_total = np.einsum("i,ij->j", chargs, cell.get_SI(Gv))
        ZSI_atomic = np.einsum("i,ij->ij", chargs, cell.get_SI(Gv)) 
        ZexpG2_atomic = ZSI_atomic * np.exp(-absG2/(4*ew_eta**2))
        # todo diff if atomic part conjugated insead?
        ewg_atomic = .5 * np.einsum('j,ij,j->i', ZSI_total.conj(), ZexpG2_atomic, coulG).real

    else:
        logger.warn(cell, 'No method for PBC dimension %s, dim-type %s.',
                    cell.dimension)
        raise NotImplementedError
    
    #TODO maybe our own warnings instead of pyscf logger
    logger.debug(cell, 'Ewald components = %.15g, %.15g, %.15g', ewovrl_atomic, ewself_atomic, ewg_atomic)
    return ewovrl_atomic + ewself_atomic + ewg_atomic

