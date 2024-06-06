"""Core recipes for universal machine-learned interatomic potentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quacc import job
from quacc.recipes.mlp._base import pick_calculator
from quacc.runners.ase import Runner
from quacc.schemas.ase import summarize_opt_run, summarize_run
from quacc.utils.dicts import recursive_dict_merge

if TYPE_CHECKING:
    from typing import Literal

    from ase.atoms import Atoms

    from quacc.runners.ase import OptParams
    from quacc.schemas._aliases.ase import OptSchema, RunSchema


@job
def static_job(
    atoms: Atoms,
    method: Literal["mace-mp-0", "m3gnet", "chgnet"],
    properties: list[str] | None = None,
    **calc_kwargs,
) -> RunSchema:
    """
    Carry out a single-point calculation.

    Parameters
    ----------
    atoms
        Atoms object
    method
        Universal ML interatomic potential method to use
    properties
        A list of properties to obtain. Defaults to ["energy", "forces"]
    **calc_kwargs
        Custom kwargs for the underlying calculator. Set a value to
        `quacc.Remove` to remove a pre-existing key entirely. For a list of available
        keys, refer to the `mace.calculators.mace_mp`, `chgnet.model.dynamics.CHGNetCalculator`,
        or `matgl.ext.ase.M3GNetCalculator` calculators.

    Returns
    -------
    RunSchema
        Dictionary of results from [quacc.schemas.ase.summarize_run][].
        See the type-hint for the data structure.
    """
    atoms.calc = pick_calculator(method, **calc_kwargs)
    if properties is None:
        properties = ["energy", "forces"]
    final_atoms = Runner(atoms).run_calc(properties=properties)
    return summarize_run(
        final_atoms, atoms, additional_fields={"name": f"{method} Static"}
    )


@job
def relax_job(
    atoms: Atoms,
    method: Literal["mace-mp-0", "m3gnet", "chgnet"],
    relax_cell: bool = False,
    opt_params: OptParams | None = None,
    **calc_kwargs,
) -> OptSchema:
    """
    Relax a structure.

    Parameters
    ----------
    atoms
        Atoms object
    method
        Universal ML interatomic potential method to use
    relax_cell
        Whether to relax the cell.
    opt_params
        Dictionary of custom kwargs for the optimization process. For a list
        of available keys, refer to [quacc.runners.ase.Runner.run_opt][].
    **calc_kwargs
        Custom kwargs for the underlying calculator. Set a value to
        `quacc.Remove` to remove a pre-existing key entirely. For a list of available
        keys, refer to the `mace.calculators.mace_mp`, `chgnet.model.dynamics.CHGNetCalculator`,
        or `matgl.ext.ase.M3GNetCalculator` calculators.

    Returns
    -------
    OptSchema
        Dictionary of results from [quacc.schemas.ase.summarize_opt_run][].
        See the type-hint for the data structure.
    """
    opt_defaults = {"fmax": 0.05}
    opt_flags = recursive_dict_merge(opt_defaults, opt_params)

    atoms.calc = pick_calculator(method, **calc_kwargs)

    dyn = Runner(atoms).run_opt(relax_cell=relax_cell, **opt_flags)

    return summarize_opt_run(dyn, additional_fields={"name": f"{method} Relax"})
