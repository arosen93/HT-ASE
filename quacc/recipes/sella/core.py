"""
Core recipes for the Sella code
"""
from __future__ import annotations

from copy import deepcopy

import covalent as ct
import numpy as np
from ase.atoms import Atoms
from ase.io import read
from monty.dev import requires
from sella import Sella

try:
    from newtonnet.utils.ase_interface import MLAseCalculator as NewtonNet
except ImportError:
    NewtonNet = None

# TODO: Make sure code doesn't crash if sella isn't installed
# TODO: Don't hardcode fmax or steps


@ct.electron
@requires(NewtonNet is not None, "NewtonNet must be installed")
def optimize_and_analyze(atoms: Atoms, ml_path: str, config_path: str) -> dict:
    mlcalculator = NewtonNet(model_path=ml_path, settings_path=config_path)
    atoms.set_calculator(mlcalculator)
    opt = Sella(atoms, internal=True, logfile=f"sella.log", trajectory=f"sella.traj")
    opt.run(fmax=0.01, steps=5)

    traj = read(f"sella.traj", index=":")
    mlcalculator.calculate(traj)
    H = mlcalculator.results["hessian"]
    n_atoms = np.shape(H)[0]
    A = np.reshape(H, (n_atoms * 3, n_atoms * 3))
    eigvals, eigvecs = np.linalg.eig(A)
    return_vars = {
        "eigvals": np.array2string(eigvals),
        "eigvecs": np.array2string(eigvecs),
    }

    summary = {
        "name": name,
        "input": {"atoms": deepcopy(atoms)},
        "output": {"return_vars": return_vars},
    }
    return summary
