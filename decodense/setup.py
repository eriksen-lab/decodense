#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
setup module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import os.path
import shutil
import numpy as np
from typing import List, Tuple, Union

import system
import results

def setup() -> system.DecompCls:
    """
    set decomp info
    """
    # decomp object
    decomp = system.DecompCls()
    decomp.atom, decomp.param = system.set_param(decomp.param)
    if 'xc' in decomp.param.keys():
        decomp.param['dft'] = True

    # rm out dir if present
    if os.path.isdir(results.OUT):
        shutil.rmtree(results.OUT, ignore_errors=True)

    # make main out dir
    os.mkdir(results.OUT)
    if decomp.param['cube']:
        # make hf out dirs
        os.mkdir(results.OUT + '/hf_can')
        os.mkdir(results.OUT + '/hf_loc')
        # make dft out dirs
        if decomp.param['dft']:
            os.mkdir(results.OUT + '/dft_can')
            os.mkdir(results.OUT + '/dft_loc')

    # init logger
    sys.stdout = tools.Logger(results.RES_FILE) # type: ignore

    return decomp


