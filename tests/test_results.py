#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
from decodense.decomp import CompKeys, DecompCls
from decodense.results import atoms, orbs, ResultsCls


# mock molecule
class MockMolAtoms:
    def __init__(self, symbols):
        self._symbols = symbols
        self.natm = len(symbols)

    def atom_symbol(self, i):
        return self._symbols[i]


# to_dataframe
# attribute access and dataframe access must give the same results for atom partitioning
def test_to_dataframe():
    mol = MockMolAtoms(["X", "Y", "Z"])
    res = {
        CompKeys.el: np.array([1.3, -0.7, 2.6]),
        CompKeys.tot: np.array([0.4, 3.9, -1.2]),
    }
    decomp = DecompCls(part="atoms")
    decomp.res = res
    results = ResultsCls(mol, decomp)
    assert "X0" in str(results)
    assert np.allclose(results.to_dataframe()[CompKeys.tot], results.tot)


# attribute access and dataframe access must give the same results for orbital partitioning
def test_to_dataframe_orbitals():
    res = {
        CompKeys.el: [np.array([1.3, -0.7]), np.array([2.6])],
        CompKeys.mo_occ: (np.array([1.0, 1.0]), np.array([1.0])),
        CompKeys.orbsym: (
            np.array(["A1", "B2"], dtype=object),
            np.array(["A1"], dtype=object),
        ),
    }
    decomp = DecompCls(part="orbitals")
    decomp.res = res
    results = ResultsCls(None, decomp)
    assert "B2" in str(results)
    assert len(results.to_dataframe()) == 3
    assert np.allclose(results.to_dataframe()[CompKeys.el], np.concatenate(results.el))


# atoms
# atom dataframe structure and unit scaling must be correct for energy
def test_atoms():
    mol = MockMolAtoms(["X", "Y"])
    res = {
        CompKeys.el: np.array([1.0, 4.0]),
        CompKeys.kin: np.array([2.0, 8.0]),
    }
    df = atoms(mol, res, "au")
    assert len(df) == 2
    assert list(df.index) == ["X0", "Y1"]
    assert np.allclose(df[CompKeys.el], [1.0, 4.0])
    assert np.allclose(df[CompKeys.kin], [2.0, 8.0])
    df_ev = atoms(mol, res, "ev")
    assert np.allclose(df_ev[CompKeys.el], np.array([1.0, 4.0]) * 27.211386245988)


# atom dataframe structure and column naming must be correct for dipole moment
def test_atoms_dipole():
    mol = MockMolAtoms(["X", "Y"])
    res = {CompKeys.el: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    df = atoms(mol, res, "au")
    assert list(df.columns) == ["Elect. (x)", "Elect. (y)", "Elect. (z)"]
    assert np.allclose(df.values, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


# orbs
# natural density orbitals must be paired correctly and unit scaling must be correct
def test_orbs_ndo():
    res = {
        CompKeys.el: [np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([])],
        CompKeys.mo_occ: (np.array([-0.8, -0.3, 0.0, 0.3, 0.8]), np.array([])),
        CompKeys.orbsym: (
            np.array(list("abcde"), dtype=object),
            np.array([], dtype=object),
        ),
    }
    df = orbs(None, res, "au", ndo=True)
    assert np.allclose(df[CompKeys.el], [1.0, 5.0, 2.0, 4.0, 3.0])
    assert np.allclose(df[CompKeys.mo_occ], [-0.8, 0.8, -0.3, 0.3, 0.0])
    df_ev = orbs(None, res, "ev", ndo=True)
    assert np.allclose(
        df_ev[CompKeys.el], np.array([1.0, 5.0, 2.0, 4.0, 3.0]) * 27.211386245988
    )


# orbital dataframe structure and column naming must be correct for dipole moment
def test_orbs_dipole():
    res = {
        CompKeys.el: [
            np.array([[1.3, -0.7, 2.6], [0.4, 3.9, -1.2]]),
            np.array([[-2.1, 0.8, 1.5]]),
        ],
        CompKeys.mo_occ: (np.array([1.0, 1.0]), np.array([1.0])),
        CompKeys.orbsym: (
            np.array(["A1", "B2"], dtype=object),
            np.array(["A1"], dtype=object),
        ),
    }
    df = orbs(None, res, "au", ndo=False)
    assert np.allclose(df[CompKeys.el + " (x)"], [1.3, 0.4, -2.1])
    assert np.allclose(df[CompKeys.el + " (y)"], [-0.7, 3.9, 0.8])
    assert np.allclose(df[CompKeys.el + " (z)"], [2.6, -1.2, 1.5])
    assert np.allclose(df[CompKeys.tot + " (x)"], [1.3, 0.4, -2.1])
    assert list(df[CompKeys.orbsym]) == ["A1", "B2", "A1"]
