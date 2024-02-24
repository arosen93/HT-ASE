"""Base jobs for Gaussian."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase.calculators.gaussian import Gaussian

from quacc import SETTINGS
from quacc.runners.ase import run_calc
from quacc.schemas.cclib import cclib_summarize_run
from quacc.utils.dicts import recursive_dict_merge

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from ase.atoms import Atoms

    from quacc.schemas._aliases.cclib import cclibSchema

_LABEL = "Gaussian"
LOG_FILE = f"{_LABEL}.log"
GAUSSIAN_CMD = f"{SETTINGS.GAUSSIAN_CMD} < {_LABEL}.com > {LOG_FILE}"


def base_fn(
    atoms: Atoms,
    calc_defaults: dict[str, Any] | None = None,
    calc_swaps: dict[str, Any] | None = None,
    additional_fields: dict[str, Any] | None = None,
    copy_files: list[str | Path] | dict[str | Path, list[str | Path]] | None = None,
) -> cclibSchema:
    """
    Base job function for carrying out Gaussian recipes.

    Parameters
    ----------
    atoms
        Atoms object
    calc_defaults
        Default parameters for the calculator.
    calc_swaps
        Dictionary of custom kwargs for the Gaussian calculator. Set a value to
        `quacc.Remove` to remove a pre-existing key entirely. For a list of available
        keys, refer to the [ase.calculators.gaussian.Gaussian][] calculator.
    additional_fields
        Additional fields to supply to the summarizer.
    copy_files
        Files to copy from source to scratch directory. If a list, the files will be
        copied as-specified. If a dictionary, the keys are the base directory and the
        values are the individual files to copy within that directory. If None, no files will
        be copied.

    Returns
    -------
    cclibSchema
        Dictionary of results, as specified in
        [quacc.schemas.cclib.cclib_summarize_run][]
    """
    calc_flags = recursive_dict_merge(calc_defaults, calc_swaps)

    atoms.calc = Gaussian(command=GAUSSIAN_CMD, label=_LABEL, **calc_flags)
    atoms = run_calc(atoms, geom_file=LOG_FILE, copy_files=copy_files)

    return cclib_summarize_run(atoms, LOG_FILE, additional_fields=additional_fields)
