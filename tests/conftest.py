#!/usr/bin/env python
# -*- coding: utf-8 -*

import pytest
from pyscf import gto, scf


# rhf water
@pytest.fixture
def mf_h2o_rhf():
    mol = gto.M(
        verbose=0, output=None, basis="sto-3g", symmetry=True, atom="geom/h2o.xyz"
    )
    mf = scf.RHF(mol).run()
    return mf
