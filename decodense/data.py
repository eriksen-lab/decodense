#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
data module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

# https://en.wikipedia.org/wiki/Hartree
AU_TO_KCAL_MOL = 627.5094740631
AU_TO_EV = 27.211386245988
AU_TO_KJ_MOL = 2625.4996394799
# https://calculla.com/dipole_moment_units_converter
AU_TO_DEBYE = 2.54174623

atom_energies = {
    'PBE0': {
        'CCPVDZ': {
            'H': -0.500289865232364,
            'C': -37.7990132018731,
            'N': -54.5307478229269,
            'O': -74.9930412090715,
            'S': -397.967687965865,
        },
        'DEF2TZVP': {
            'H': -0.501036290609284,
            'C': -37.8053740252837,
            'N': -54.5438199191733,
            'O': -75.0186004407083,
            'S': -397.973710162509,
        },
    },
    'WB97X_D': {
        'CCPVDZ': {
            'H': -0.501881000145483,
            'C': -37.8357259442631,
            'N': -54.5730355132859,
            'O': -75.0443112471207,
            'S': -398.101531061186,
        },
        'DEF2TZVP': {
            'H': -0.502666338167257,
            'C': -37.8423799255518,
            'N': -54.5868960961547,
            'O': -75.0708847493225,
            'S': -398.107679431829,
        },
    },
}


