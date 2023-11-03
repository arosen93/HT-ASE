"""Phonon recipes for TBLite"""
from __future__ import annotations

from typing import TYPE_CHECKING

from quacc import flow
from quacc.recipes.phonons import run_phonons
from quacc.runners.ase import run_calc
from quacc.schemas.ase import summarize_run
from quacc.schemas.phonopy import summarize_phonopy
from quacc.utils.dicts import merge_dicts

try:
    from tblite.ase import TBLite
except ImportError:
    TBLite = None

if TYPE_CHECKING:
    from typing import Any, Literal

    from ase import Atoms
    from numpy.typing import ArrayLike

    from quacc.recipes.phonons import PhononSchema


@flow
def phonon_flow(
    atoms: Atoms,
    method: Literal["GFN1-xTB", "GFN2-xTB", "IPEA1-xTB"] = "GFN2-xTB",
    supercell_matrix: ArrayLike = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
    atom_disp: float = 0.015,
    t_step: float = 10,
    t_min: float = 0,
    t_max: float = 1000,
    calc_swaps: dict[str, Any] | None = None,
) -> PhononSchema:
    """
    Carry out a phonon calculation.

    Parameters
    ----------
    atoms
        Atoms object
    method
        GFN1-xTB, GFN2-xTB, and IPEA1-xTB.
    calc_swaps
        Dictionary of custom kwargs for the EMT calculator. Set a value to
        `None` to remove a pre-existing key entirely. For a list of available
        keys, refer to the `tblite.ase.TBLite` calculator.

        !!! Info "Calculator defaults"

            ```python
            {"method": method}
            ```
    copy_files
        Files to copy to the runtime directory.

    Returns
    -------
    PhononSchema
        Dictionary of results from [quacc.schemas.phonopy.summarize_phonopy][]
    """

    defaults = {"method": method}
    flags = merge_dicts(defaults, calc_swaps)
    atoms.calc = TBLite(**flags)

    phonon = run_phonons(
        atoms,
        supercell_matrix=supercell_matrix,
        atom_disp=atom_disp,
        t_step=t_step,
        t_min=t_min,
        t_max=t_max,
    )
    return summarize_phonopy(
        phonon,
        input_atoms=atoms,
        additional_fields={"name": "TBLite Phonons"},
    )
