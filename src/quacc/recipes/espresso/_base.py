"""Base jobs for espresso."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ase.atoms import Atoms
from ase.io.espresso import Namelist
from ase.io.espresso_namelist.keys import ALL_KEYS

from quacc.calculators.espresso.espresso import (
    Espresso,
    EspressoProfile,
    EspressoTemplate,
)
from quacc.calculators.espresso.utils import prepare_copy_files
from quacc.runners.ase import run_calc, run_opt
from quacc.schemas.ase import summarize_opt_run, summarize_run
from quacc.utils.dicts import recursive_dict_merge

if TYPE_CHECKING:
    from typing import Any

    from quacc.schemas._aliases.ase import RunSchema
    from quacc.utils.files import Filenames, SourceDirectory


def run_and_summarize(
    atoms: Atoms | None = None,
    preset: str | None = None,
    template: EspressoTemplate | None = None,
    profile: EspressoProfile | None = None,
    calc_defaults: dict[str, Any] | None = None,
    calc_swaps: dict[str, Any] | None = None,
    parallel_info: dict[str, Any] | None = None,
    additional_fields: dict[str, Any] | None = None,
    copy_files: (
        SourceDirectory
        | list[SourceDirectory]
        | dict[SourceDirectory, Filenames]
        | None
    ) = None,
) -> RunSchema:
    """
    Base function to carry out espresso recipes.

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Name of the preset to use
    template
        EspressoTemplate to use
    profile
        EspressoProfile to use
    calc_defaults
        The default calculator parameters.
    calc_swaps
        Custom kwargs for the espresso calculator. Set a value to
        `quacc.Remove` to remove a pre-existing key entirely. For a list of available
        keys, refer to the [ase.calculators.espresso.Espresso][] calculator.
    parallel_info
        Dictionary of parallelization information.
    additional_fields
        Any additional fields to supply to the summarizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.

    Returns
    -------
    RunSchema
        Dictionary of results from [quacc.schemas.ase.summarize_run][]
    """
    atoms, copy_files = _prepare_calc(
        atoms=atoms,
        preset=preset,
        template=template,
        profile=profile,
        calc_defaults=calc_defaults,
        calc_swaps=calc_swaps,
        parallel_info=parallel_info,
        copy_files=copy_files,
    )

    geom_file = template.outputname if template.binary == "pw" else None

    final_atoms = run_calc(atoms, geom_file=geom_file, copy_files=copy_files)

    return summarize_run(
        final_atoms, atoms, move_magmoms=True, additional_fields=additional_fields
    )


def run_and_summarize_opt(
    atoms: Atoms | None = None,
    preset: str | None = None,
    relax_cell: bool = False,
    template: EspressoTemplate | None = None,
    profile: EspressoProfile | None = None,
    calc_defaults: dict[str, Any] | None = None,
    calc_swaps: dict[str, Any] | None = None,
    opt_defaults: dict[str, Any] | None = None,
    opt_params: dict[str, Any] | None = None,
    parallel_info: dict[str, Any] | None = None,
    additional_fields: dict[str, Any] | None = None,
    copy_files: (
        SourceDirectory
        | list[SourceDirectory]
        | dict[SourceDirectory, Filenames]
        | None
    ) = None,
) -> RunSchema:
    """
    Base function to carry out espresso recipes with ASE optimizers.

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Name of the preset to use
    relax_cell
        Whether to relax the cell or not.
    template
        EspressoTemplate to use
    profile
        EspressoProfile to use
    calc_defaults
        The default calculator parameters.
    calc_swaps
        Custom kwargs for the espresso calculator. Set a value to
        `quacc.Remove` to remove a pre-existing key entirely. For a list of available
        keys, refer to the [ase.calculators.espresso.Espresso][] calculator.
    opt_defaults
        The default optimization parameters.
    opt_params
        Dictionary of parameters to pass to the optimizer. pass "optimizer"
        to change the optimizer being used. "fmax" and "max_steps" are commonly
        used keywords. See the ASE documentation for more information.
    parallel_info
        Dictionary of parallelization information.
    additional_fields
        Any additional fields to supply to the summarizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.

    Returns
    -------
    RunSchema
        Dictionary of results from [quacc.schemas.ase.summarize_run][]
    """
    atoms, copy_files = _prepare_calc(
        atoms=atoms,
        preset=preset,
        template=template,
        profile=profile,
        calc_defaults=calc_defaults,
        calc_swaps=calc_swaps,
        parallel_info=parallel_info,
        copy_files=copy_files,
    )

    opt_flags = recursive_dict_merge(opt_defaults, opt_params)

    dyn = run_opt(atoms, relax_cell=relax_cell, copy_files=copy_files, **opt_flags)

    return summarize_opt_run(
        dyn, move_magmoms=True, additional_fields=additional_fields
    )


def _prepare_calc(
    atoms: Atoms | None = None,
    preset: str | None = None,
    template: EspressoTemplate | None = None,
    profile: EspressoProfile | None = None,
    calc_defaults: dict[str, Any] | None = None,
    calc_swaps: dict[str, Any] | None = None,
    parallel_info: dict[str, Any] | None = None,
    copy_files: (
        SourceDirectory
        | list[SourceDirectory]
        | dict[SourceDirectory, Filenames]
        | None
    ) = None,
) -> Atoms:
    """
    Commonly used preparation function to merge parameters
    and attach an Espresso calculator accordingly.

    Parameters
    ----------
    atoms
        Atoms object
    preset
        Name of the preset to use
    template
        EspressoTemplate to use
    profile
        EspressoProfile to use
    calc_defaults
        The default calculator parameters.
    calc_swaps
        Custom kwargs for the espresso calculator. Set a value to
        `quacc.Remove` to remove a pre-existing key entirely. For a list of available
        keys, refer to the [ase.calculators.espresso.Espresso][] calculator.
    parallel_info
        Dictionary of parallelization information.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.

    Returns
    -------
    Atoms
        Atoms object with attached Espresso calculator.
    """
    atoms = Atoms() if atoms is None else atoms
    calc_defaults = calc_defaults or {}
    calc_swaps = calc_swaps or {}

    calc_defaults["input_data"] = Namelist(calc_defaults.get("input_data"))
    calc_swaps["input_data"] = Namelist(calc_swaps.get("input_data"))

    binary = template.binary if template else "pw"

    if binary in ALL_KEYS:
        calc_defaults["input_data"].to_nested(binary=binary, **calc_defaults)
        calc_swaps["input_data"].to_nested(binary=binary, **calc_swaps)

    calc_flags = recursive_dict_merge(calc_defaults, calc_swaps)

    calc = Espresso(
        input_atoms=atoms,
        preset=preset,
        parallel_info=parallel_info,
        template=template,
        profile=profile,
        **calc_flags,
    )

    if copy_files:
        if isinstance(copy_files, (str, Path)):
            copy_files = [copy_files]

        exact_files_to_copy = prepare_copy_files(calc._user_calc_params, binary=binary)
        updated_copy_files = {source: exact_files_to_copy for source in copy_files}

    atoms.calc = calc

    return atoms, updated_copy_files
