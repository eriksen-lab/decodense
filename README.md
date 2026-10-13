![License](https://img.shields.io/github/license/eriksen-lab/decodense)
![CI](https://github.com/eriksen-lab/decodense/actions/workflows/ci.yml/badge.svg)

# Decodense: A Decomposed Mean-Field Theory Code

Decodense decomposes mean-field (HF/KS-DFT) results in terms of either 
atom- or orbital-wise contributions to a molecular energy or dipole moment.

The following papers document the theory and should be cited in any work
using decodense:

- Eriksen, J. J. *Mean-field density matrix decompositions*. J. Chem. Phys.
  **2020**, 153, 214109. DOI:
  [10.1063/5.0030764](https://doi.org/10.1063/5.0030764).
- Eriksen, J. J. *Electronic excitations through the prism of mean-field
  decomposition techniques*. J. Chem. Phys. **2022**, 156, 061101. DOI:
  [10.1063/5.0082938](https://doi.org/10.1063/5.0082938).
- Zamok, L.; Eriksen, J. J. *Atomic Decompositions of Periodic
  Electronic-Structure Simulations*. J. Phys. Chem. A **2025**, 129,
  385–395. DOI:
  [10.1021/acs.jpca.4c06651](https://doi.org/10.1021/acs.jpca.4c06651).

## Installation

To install:

```sh
git clone https://github.com/eriksen-lab/decodense.git
cd decodense
pip install -e .
```

The installation can be tested by running

```sh
pytest
```

in the `tests` folder. If pytest is not installed yet, install it via

```sh
pip install pytest
```

### Prerequisites

- Most examples in **`examples`** use the **`mf_to_otr`** function, which wraps 
  PySCF **`HF`** and **`KS`** objects into their OpenTrustRegion
  counterparts, and the **`PipekMezeyOTR`** class for localization. Orbital
  optimization and internal stability analysis are also performed using the
  **`kernel`** and **`stability_check`** member functions, respectively. Running
  these therefore also requires
  [OpenTrustRegion](https://github.com/eriksen-lab/opentrustregion) and its
  PySCF extension,
  [pyscf_opentrustregion](https://github.com/eriksen-lab/pyscf_opentrustregion).
  After installing both, add the extension's path to **`PYSCF_EXT_PATH`**:
  
    ```sh
  export PYSCF_EXT_PATH=path/to/pyscf_opentrustregion:$PYSCF_EXT_PATH
  ```
  
### Settings

Pass these keyword arguments to **`DecompCls`** to configure the decomposition:

| Setting | Type | Default | Description |
|---|---|---|---|
| **`part`** | `str` | `"atoms"` | Partitioning target: `"atoms"`, `"orbitals"` |
| **`part_method`** | `str` | `"mo"` if `part="atoms"`,  `None` otherwise | Partitioning method for `part="atoms"`: `"mo"` (the MO-based scheme of the 2020 JCP paper by Eriksen) or `"ao"` (Nakai's AO-based energy density analysis\*). |
| **`pop_method`** | `str` | `"mulliken"` | Population scheme: `"mulliken"`, `"lowdin"`, `"meta_lowdin"`, `"becke"`, or `"iao"`. |
| **`prop`** | `str` | `"energy"` | Property to decompose: `"energy"` or `"dipole"`. |
| **`minao`** | `str` | `"MINAO"` | Reference basis for IAO population analysis: `"MINAO"` or `"ANO"`. |
| **`mo_basis`** | `str` | `"can"` | MO basis for the decomposition: `"can"` (canonical), `"fb"` (Foster-Boys), or `"pm"` (Pipek-Mezey). |
| **`gauge_origin`** | `np.ndarray` | `[0, 0, 0]` | Gauge origin used for dipole-moment decomposition. |
| **`ndo`** | `bool` | `False` | Natural Difference Orbitals (NDOs) decomposition. |
| **`unit`** | `str` | `"au"` | Output unit: `"au"`, `"kcal_mol"`, `"ev"`, `"kj_mol"`, or `"debye"`. |
| **`atom_ref`** | `dict` | `None` | Dictionary of atomic reference energies.  If provided, an atom-wise decomposition will also calculate the difference between the atom-wise contributions and the reference energies. The expected format is `{ "atom_label" : atom_energy }`. |

\* H. Nakai, *Energy density analysis with Kohn–Sham orbitals*, Chem. Phys.
Lett. **2002**, 363, 73–79. DOI:
[10.1016/S0009-2614(02)01151-X](https://doi.org/10.1016/S0009-2614(02)01151-X).
