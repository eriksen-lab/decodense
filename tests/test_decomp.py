#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import pytest
from decodense.decomp import DecompCls, sanity_check


# DecompCls
# DecompCls must set the correct part/part_method defaults
def test_decomp_cls_part_method_defaults():
    decomp_eda = DecompCls(part="eda")
    assert decomp_eda.part == "atoms"
    assert decomp_eda.part_method == "ao"
    decomp_atoms = DecompCls(part="atoms")
    assert decomp_atoms.part_method == "mo"
    decomp_orb = DecompCls(part="orbitals")
    assert decomp_orb.part_method is None


# sanity_check
# sanity check must reject invalid attributes
@pytest.mark.parametrize(
    "attr,bad_value,exception,message",
    [
        ("minao", "bad", ValueError, "invalid minao basis"),
        ("mo_basis", "bad", ValueError, "invalid MO basis"),
        ("pop_method", "bad", ValueError, "invalid population scheme"),
        ("mo_init", "bad", ValueError, "invalid MO start guess"),
        ("loc_exp", 3, ValueError, "invalid localization exponent"),
        ("part", "bad", ValueError, "invalid partitioning"),
        ("part_method", "bad", ValueError, "invalid partitioning method"),
        ("prop", "bad", ValueError, "invalid property"),
        ("unit", "bad", ValueError, "invalid unit"),
        ("verbose", -1, ValueError, "invalid verbosity"),
        ("verbose", 6, ValueError, "invalid verbosity"),
        ("ndo", "not_a_bool", TypeError, "invalid NDO argument"),
        ("write", "bad", ValueError, "invalid write format"),
        ("write", 123, TypeError, "invalid write format argument"),
        ("writename", 123, TypeError, "invalid write name argument"),
        ("verbose", "not_an_int", TypeError, "invalid verbosity"),
        ("unit", 123, TypeError, "invalid unit"),
    ],
)
def test_sanity_check_rejects_invalid_attr(attr, bad_value, exception, message):
    decomp = DecompCls(**{attr: bad_value})
    with pytest.raises(exception, match=message):
        sanity_check(None, None, decomp, np.zeros((2, 2)), None)


# sanity check must reject an invalid gauge_origin
@pytest.mark.parametrize(
    "bad_gauge_origin,exception",
    [
        ([0.0, 0.0], ValueError),  # too short
        ([0.0, 0.0, 0.0, 0.0], ValueError),  # too long
        (["a", "b", "c"], ValueError),  # right length, wrong element type
        ("not a list", TypeError),  # wrong container type entirely
    ],
)
def test_sanity_check_gauge_origin_invalid(bad_gauge_origin, exception):
    decomp = DecompCls(gauge_origin=bad_gauge_origin)
    with pytest.raises(exception, match="invalid gauge origin"):
        sanity_check(None, None, decomp, np.zeros((2, 2)), None)


# sanity check must reject an invalid mo coefficient
@pytest.mark.parametrize(
    "mo_coeff,exception",
    [
        ([1, 2, 3], TypeError),  # not an array or tuple
        (np.zeros(5), ValueError),  # array of wrong dimension
        ((np.zeros((2, 2)),), TypeError),  # tuple of wrong length
    ],
)
def test_sanity_check_rejects_invalid_mo_coeff(mo_coeff, exception):
    decomp = DecompCls()
    with pytest.raises(exception, match="invalid mo coefficients"):
        sanity_check(None, None, decomp, mo_coeff, None)


# sanity check must reject an invalid mo occupation
@pytest.mark.parametrize(
    "mo_occ,exception",
    [
        ("not valid", TypeError),  # not an array or tuple
        (np.zeros((2, 2, 2)), ValueError),  # array of wrong dimension
        ((np.zeros(2),), TypeError),  # tuple of wrong length
    ],
)
def test_sanity_check_rejects_invalid_mo_occ(mo_occ, exception):
    decomp = DecompCls()
    mo_coeff = np.zeros((2, 2))
    with pytest.raises(exception, match="invalid mo occupation"):
        sanity_check(None, None, decomp, mo_coeff, mo_occ)


# sanity check must reject write unless part="atoms" and part_method="mo"
@pytest.mark.parametrize(
    "kwargs",
    [
        {"part": "orbitals", "write": "cube"},
        {"part": "atoms", "part_method": "ao", "write": "numpy"},
    ],
)
def test_sanity_check_write_requires_atoms_mo(kwargs):
    decomp = DecompCls(**kwargs)
    with pytest.raises(ValueError, match="write is only implemented"):
        sanity_check(None, None, decomp, np.zeros((2, 2)), None)


# sanity check must accept valid input
@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # fully default
        {"verbose": 0},
        {"verbose": 5},
        {"gauge_origin": [0.0, 0.0, 0.0]},
        {"part": "orbitals"},
        {"part": "atoms", "part_method": "ao"},
        {"part": "eda"},
    ],
)
def test_sanity_check_accepts_valid_input(kwargs):
    mo_coeff = np.zeros((2, 2))
    decomp = DecompCls(**kwargs)
    sanity_check(None, None, decomp, mo_coeff, None)
