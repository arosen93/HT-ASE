"""Core recipes for VASP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import os
from monty.os.path import zpath
from pathlib import Path
from pymatgen.io.vasp import Vasprun
from quacc import flow, job
from quacc.recipes.vasp._base import base_fn

if TYPE_CHECKING:
    from typing import Any
    from ase import Atoms

    from quacc.schemas._aliases.vasp import DoubleRelaxSchema, VaspSchema
    from quacc.utils.files import Filenames, SourceDirectory


@job
def nscf_job(
    atoms: Atoms,
    prev_dir: SourceDirectory,
    bandgap: float | None = None,
#    nbands_factor: float | None = None,
    preset: str | None = "BulkSet",
    kpoints_mode: Literal["uniform", "line"] = "uniform",
    calculate_optics: bool = False,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out a non-self-consistent field (NSCF) calculation.

    Parameters
    ----------
    atoms
        Atoms object.
    prev_dir: that is generally the folder (str | path) of the static_job 
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    kpoints_mode
        Type of k-points mode. Options are "uniform" or "line". "boltztrap" may
        be implemented in future.
    calculate_optics
        Whether to calculate optical properties.
    bandgap
        The band gap of the material.
    nbands_factor
        It increases NBANDS. However, to use it, the vasprun.xml(.gz) must
        provided in prev_dir.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][].

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.vasp_summarize_run][].
        See the type-hint for the data structure.
    """

    calc_defaults = {
        "lorbit": 11,
        "lwave": False,
        "lcharg": False,
        "nsw": 0,
        "isym": 0,
        "icharg": 11,
        "kspacing": None,
        "nedos": 5001,
    }

    # check the expected files in prev_dir
    basics_to_copy = ["CHGCAR*", "WAVECAR*", "vasprun.xml*"]
    files_to_copy = {prev_dir: basics_to_copy}
    chgar_files = [file for file in os.listdir(prev_dir) if file.startswith('CHGCAR')]
    if not chgar_files:
        raise FileNotFoundError("No CHGCAR* file exists in the specified directory.")
    vasprun_exists = any(file.startswith('vasprun.xml') for file in os.listdir(prev_dir))
    if vasprun_exists:
        vasprun_path = Path(prev_dir, "vasprun.xml")
        if (vasprun_path_gz := Path(str(vasprun_path) + ".gz")).exists():
            vasprun_path = zpath(vasprun_path_gz) # if vasprun.xml.gz, zpath will decompress it
            vasprun = Vasprun(vasprun_path)
        else:
            print("Warning: vasprun.xml* file does not exist in the specified directory.")
    for file_name in basics_to_copy[1:]: #actually it checks the existence of WAVECAR*
        matching_files = [file for file in os.listdir(prev_dir) if file.startswith(file_name[:-1])]
        if not matching_files:
            print(f"Warning: {file_pattern} file does not exist in the specified directory.")


    updates: dict[str, Any] = {}

#    if vasprun is not None and nbands_factor is not None:
#        nbands_factor = nbands_factor
#        nbands = int(np.ceil(vasprun.parameters["NBANDS"] * nbands_factor))
#        updates["nbands"] = nbands

    if kpoints_mode == "uniform":
        # Use tetrahedron method for DOS and optics calculations
        updates.update({"ismear": -5, "isym": 2})
    elif kpoints_mode == "line":
        sigma = 0.2 if bandgap == 0 else 0.01
        updates.update({"ismear": 0, "sigma": sigma})

    if calculate_optics:
        updates.update(
            {"loptics": True, "lreal": False, "cshift": 1e-5}
        )

    # integrate updates to calc_kwargs
    calc_kwargs.update(updates)
    copy_files = {prev_dir: ["CHGCAR*", "WAVECAR*"]}

    return base_fn(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP NSCF"},
        copy_files = copy_files, 
    )


@job
def static_job(
    atoms: Atoms,
    preset: str | None = "BulkSet",
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Carry out a single-point calculation.

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to [quacc.calculators.vasp.vasp.Vasp][].

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.vasp_summarize_run][].
        See the type-hint for the data structure.
    """

    calc_defaults = {
        "ismear": -5,
        "laechg": True,
        "lcharg": True,
        "lreal": False,
        "lwave": True,
        "nedos": 5001,
        "nsw": 0,
    }
    return base_fn(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP Static"},
        copy_files=copy_files,
    )


@job
def relax_job(
    atoms: Atoms,
    preset: str | None = "BulkSet",
    relax_cell: bool = True,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    **calc_kwargs,
) -> VaspSchema:
    """
    Relax a structure.

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    relax_cell
        True if a volume relaxation (ISIF = 3) should be performed. False if
        only the positions (ISIF = 2) should be updated.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    **calc_kwargs
        Custom kwargs for the Vasp calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to the [quacc.calculators.vasp.vasp.Vasp][] calculator.

    Returns
    -------
    VaspSchema
        Dictionary of results from [quacc.schemas.vasp.vasp_summarize_run][].
        See the type-hint for the data structure.
    """

    calc_defaults = {
        "ediffg": -0.02,
        "isif": 3 if relax_cell else 2,
        "ibrion": 2,
        "isym": 0,
        "lcharg": False,
        "lwave": False,
        "nsw": 200,
        "symprec": 1e-8,
    }
    return base_fn(
        atoms,
        preset=preset,
        calc_defaults=calc_defaults,
        calc_swaps=calc_kwargs,
        additional_fields={"name": "VASP Relax"},
        copy_files=copy_files,
    )


@flow
def double_relax_flow(
    atoms: Atoms,
    preset: str | None = "BulkSet",
    relax_cell: bool = True,
    relax1_kwargs: dict[str, Any] | None = None,
    relax2_kwargs: dict[str, Any] | None = None,
) -> DoubleRelaxSchema:
    """
    Double-relax a structure. This is particularly useful for a few reasons:

    1. To carry out a cheaper pre-relaxation before the high-quality run.

    2. To carry out a GGA calculation before a meta-GGA or hybrid calculation
    that requires the GGA wavefunction.

    3. To carry out volume relaxations where large changes in volume
    can require a second relaxation to resolve forces.

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Preset to use from `quacc.calculators.vasp.presets`.
    relax_cell
        True if a volume relaxation (ISIF = 3) should be performed. False if
        only the positions (ISIF = 2) should be updated.
    relax1_kwargs
        Dictionary of custom kwargs for the first relaxation.
    relax2_kwargs
        Dictionary of custom kwargs for the second relaxation.

    Returns
    -------
    DoubleRelaxSchema
        Dictionary of results. See the type-hint for the data structure.
    """
    relax1_kwargs = relax1_kwargs or {}
    relax2_kwargs = relax2_kwargs or {}

    # Run first relaxation
    summary1 = relax_job(atoms, preset=preset, relax_cell=relax_cell, **relax1_kwargs)

    # Run second relaxation
    summary2 = relax_job(
        summary1["atoms"],
        preset=preset,
        relax_cell=relax_cell,
        copy_files={summary1["dir_name"]: ["WAVECAR*"]},
        **relax2_kwargs,
    )

    return {"relax1": summary1, "relax2": summary2}
