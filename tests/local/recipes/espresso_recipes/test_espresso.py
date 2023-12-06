from pathlib import Path
from shutil import which

import numpy as np
import pytest
from ase.build import bulk
from ase.io.espresso import construct_namelist

from quacc import SETTINGS
from quacc.recipes.espresso.core import ph_job, static_job

pytestmark = pytest.mark.skipif(which("pw.x") is None, reason="QE not installed")

DEFAULT_SETTINGS = SETTINGS.model_copy()


def test_static_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    atoms = bulk("Si")

    input_data = {
        "occupations": "smearing",
        "smearing": "gaussian",
        "degauss": 0.005,
        "mixing_mode": "plain",
        "mixing_beta": 0.6,
        "pseudo_dir": Path(__file__).parent,
        "conv_thr": 1.0e-6,
    }

    pseudopotentials = {"Si": "Si.upf"}

    results = static_job(
        atoms, input_data=input_data, pseudopotentials=pseudopotentials, kspacing=0.5
    )

    input_data = dict(construct_namelist(input_data))

    assert np.allclose(results["atoms"].positions, atoms.positions, atol=1.0e-4)

    assert np.allclose(results["atoms"].cell, atoms.cell, atol=1.0e-3)
    assert (results["atoms"].symbols == atoms.symbols).all()
    assert input_data.items() <= results["parameters"]["input_data"].items()


def test_ph_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    atoms = bulk("Li")

    SETTINGS.ESPRESSO_PP_PATH = Path(__file__).parent

    input_data = {
        "calculation": "scf",
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.02,
        "mixing_mode": "TF",
        "mixing_beta": 0.7,
        "conv_thr": 1.0e-6,
    }

    ph_loose = {"tr2_ph": 1e-10}

    pseudopotentials = {"Li": "Li.upf"}

    pw_results = static_job(
        atoms, input_data=input_data, pseudopotentials=pseudopotentials, kspacing=0.25
    )

    ph_results = ph_job(input_data=ph_loose, copy_files=pw_results["dir_name"])

    assert (0, 0, 0) in ph_results["results"]
    assert np.allclose(
        ph_results["results"][(0, 0, 0)]["atoms"].positions,
        atoms.positions,
        atol=1.0e-4,
    )
    # ph.x cell param are not defined to a very high level of accuracy,
    # atol = 1.0e-3 is needed here...
    assert np.allclose(
        ph_results["results"][(0, 0, 0)]["atoms"].cell, atoms.cell, atol=1.0e-3
    )
    assert (ph_results["results"][(0, 0, 0)]["atoms"].symbols == atoms.symbols).all()

    sections = ["atoms", "eqpoints", "freqs", "kpoints", "mode_symmetries", "modes"]

    for key in sections:
        assert key in ph_results["results"][(0, 0, 0)]
