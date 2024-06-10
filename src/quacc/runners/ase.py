"""Utility functions for running ASE calculators with ASE-based methods."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path
from shutil import copy, copytree
from typing import TYPE_CHECKING, Callable

import numpy as np
from ase import Atoms
from ase.calculators import calculator
from ase.filters import FrechetCellFilter
from ase.io import Trajectory, read, write
from ase.mep import NEB
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from monty.dev import requires
from monty.os.path import zpath

from quacc import SETTINGS
from quacc.atoms.core import copy_atoms, get_final_atoms_from_dynamics
from quacc.runners.prep import calc_cleanup, calc_setup, terminate
from quacc.utils.dicts import recursive_dict_merge
has_sella = bool(find_spec("sella"))
has_geodesic_interpolate = bool(find_spec("geodesic_interpolate"))

if has_geodesic_interpolate:
    from geodesic_interpolate.geodesic import Geodesic
    from geodesic_interpolate.interpolation import redistribute

if TYPE_CHECKING:
    from typing import Any, TypedDict

    from ase.optimize.optimize import Optimizer

    from quacc.utils.files import Filenames, SourceDirectory

    class OptParams(TypedDict, total=False):
        """
        Type hint for `opt_params` used throughout quacc.
        """

        fmax: float
        max_steps: int
        optimizer: Optimizer = BFGS
        optimizer_kwargs: OptimizerKwargs | None
        store_intermediate_results: bool
        fn_hook: Callable | None
        run_kwargs: dict[str, Any] | None

    class OptimizerKwargs(TypedDict, total=False):
        """
        Type hint for `optimizer_kwargs` in [quacc.runners.ase.run_opt][].
        """

        restart: Path | str | None  # default = None
        append_trajectory: bool  # default = False

    class VibKwargs(TypedDict, total=False):
        """
        Type hint for `vib_kwargs` in [quacc.runners.ase.run_vib][].
        """

        indices: list[int] | None  # default = None
        delta: float  # default = 0.01
        nfree: int  # default = 2


def run_calc(
    atoms: Atoms,
    geom_file: str | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    properties: list[str] | None = None,
) -> Atoms:
    """
    Run a calculation in a scratch directory and copy the results back to the original
    directory. This can be useful if file I/O is slow in the working directory, so long
    as file transfer speeds are reasonable.

    This is a wrapper around atoms.get_potential_energy(). Note: This function
    does not modify the atoms object in-place.

    Parameters
    ----------
    atoms
        The Atoms object to run the calculation on.
    geom_file
        The filename of the log file that contains the output geometry, used to
        update the atoms object's positions and cell after a job. It is better
        to specify this rather than relying on ASE's
        atoms.get_potential_energy() function to update the positions, as this
        varies between codes.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.
    properties
        List of properties to calculate. Defaults to ["energy"] if `None`.

    Returns
    -------
    Atoms
        The updated Atoms object.
    """
    # Copy atoms so we don't modify it in-place
    atoms = copy_atoms(atoms)

    # Perform staging operations
    tmpdir, job_results_dir = calc_setup(atoms, copy_files=copy_files)

    # Run calculation
    if properties is None:
        properties = ["energy"]

    try:
        atoms.calc.calculate(atoms, properties, calculator.all_changes)
    except Exception as exception:
        terminate(tmpdir, exception)

    # Most ASE calculators do not update the atoms object in-place with a call
    # to .get_potential_energy(), which is important if an internal optimizer is
    # used. This section is done to ensure that the atoms object is updated to
    # the final geometry if `geom_file` is provided.
    # Note: We have to be careful to make sure we don't lose the calculator
    # object, as this contains important information such as the parameters
    # and output properties (e.g. final magnetic moments).
    if geom_file:
        atoms_new = read(zpath(tmpdir / geom_file))
        if isinstance(atoms_new, list):
            atoms_new = atoms_new[-1]

        # Make sure the atom indices didn't get updated somehow (sanity check).
        # If this happens, there is a serious problem.
        if (
            np.array_equal(atoms_new.get_atomic_numbers(), atoms.get_atomic_numbers())
            is False
        ):
            raise ValueError("Atomic numbers do not match between atoms and geom_file.")

        atoms.positions = atoms_new.positions
        atoms.cell = atoms_new.cell

    # Perform cleanup operations
    calc_cleanup(atoms, tmpdir, job_results_dir)

    return atoms


def run_opt(
    atoms: Atoms,
    relax_cell: bool = False,
    fmax: float = 0.01,
    max_steps: int = 1000,
    optimizer: Optimizer = BFGS,
    optimizer_kwargs: OptimizerKwargs | None = None,
    store_intermediate_results: bool = False,
    fn_hook: Callable | None = None,
    run_kwargs: dict[str, Any] | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
) -> Optimizer:
    """
    Run an ASE-based optimization in a scratch directory and copy the results back to
    the original directory. This can be useful if file I/O is slow in the working
    directory, so long as file transfer speeds are reasonable.

    This is a wrapper around the optimizers in ASE. Note: This function does not
    modify the atoms object in-place.

    Parameters
    ----------
    atoms
        The Atoms object to run the calculation on.
    relax_cell
        Whether to relax the unit cell shape and volume.
    fmax
        Tolerance for the force convergence (in eV/A).
    max_steps
        Maximum number of steps to take.
    optimizer
        Optimizer class to use.
    optimizer_kwargs
        Dictionary of kwargs for the optimizer. Takes all valid kwargs for ASE
        Optimizer classes. Refer to `_set_sella_kwargs` for Sella-related
        kwargs and how they are set.
    store_intermediate_results
        Whether to store the files generated at each intermediate step in the
        optimization. If enabled, they will be stored in a directory named
        `stepN` where `N` is the step number, starting at 0.
    fn_hook
        A custom function to call after each step of the optimization.
        The function must take the instantiated dynamics class as
        its only argument.
    run_kwargs
        Dictionary of kwargs for the run() method of the optimizer.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.

    Returns
    -------
    Optimizer
        The ASE Optimizer object.
    """
    # Copy atoms so we don't modify it in-place
    atoms = copy_atoms(atoms)

    # Perform staging operations
    tmpdir, job_results_dir = calc_setup(atoms, copy_files=copy_files)

    # Set defaults
    optimizer_kwargs = recursive_dict_merge(
        {
            "logfile": "-" if SETTINGS.DEBUG else tmpdir / "opt.log",
            "restart": tmpdir / "opt.json",
        },
        optimizer_kwargs,
    )
    run_kwargs = run_kwargs or {}

    # Check if trajectory kwarg is specified
    if "trajectory" in optimizer_kwargs:
        msg = "Quacc does not support setting the `trajectory` kwarg."
        raise ValueError(msg)

    # Handle optimizer kwargs
    if optimizer.__name__.startswith("SciPy"):
        optimizer_kwargs.pop("restart", None)
    elif optimizer.__name__ == "Sella":
        _set_sella_kwargs(atoms, optimizer_kwargs)
    elif optimizer.__name__ == "IRC":
        optimizer_kwargs.pop("restart", None)

    # Define the Trajectory object
    traj_file = tmpdir / "opt.traj"
    traj = Trajectory(traj_file, "w", atoms=atoms)
    optimizer_kwargs["trajectory"] = traj

    # Set volume relaxation constraints, if relevant
    if relax_cell and atoms.pbc.any():
        atoms = FrechetCellFilter(atoms)

    # Run optimization
    try:
        with traj, optimizer(atoms, **optimizer_kwargs) as dyn:
            if optimizer.__name__.startswith("SciPy"):
                # https://gitlab.com/ase/ase/-/issues/1475
                dyn.run(fmax=fmax, steps=max_steps, **run_kwargs)
            else:
                for i, _ in enumerate(
                    dyn.irun(fmax=fmax, steps=max_steps, **run_kwargs)
                ):
                    if store_intermediate_results:
                        _copy_intermediate_files(
                            tmpdir,
                            i,
                            files_to_ignore=[
                                traj_file,
                                optimizer_kwargs["restart"],
                                optimizer_kwargs["logfile"],
                            ],
                        )
                    if fn_hook:
                        fn_hook(dyn)
    except Exception as exception:
        terminate(tmpdir, exception)

    # Store the trajectory atoms
    dyn.traj_atoms = read(traj_file, index=":")

    # Perform cleanup operations
    calc_cleanup(get_final_atoms_from_dynamics(dyn), tmpdir, job_results_dir)

    return dyn


def run_vib(
    atoms: Atoms,
    vib_kwargs: VibKwargs | None = None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
) -> Vibrations:
    """
    Run an ASE-based vibration analysis in a scratch directory and copy the results back
    to the original directory. This can be useful if file I/O is slow in the working
    directory, so long as file transfer speeds are reasonable.

    This is a wrapper around the vibrations module in ASE. Note: This function
    does not modify the atoms object in-place.

    Parameters
    ----------
    atoms
        The Atoms object to run the calculation on.
    vib_kwargs
        Dictionary of kwargs for the [ase.vibrations.Vibrations][] class.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.

    Returns
    -------
    Vibrations
        The updated Vibrations module
    """
    # Copy atoms so we don't modify it in-place
    atoms = copy_atoms(atoms)

    # Set defaults
    vib_kwargs = vib_kwargs or {}

    # Perform staging operations
    tmpdir, job_results_dir = calc_setup(atoms, copy_files=copy_files)

    # Run calculation
    vib = Vibrations(atoms, name=str(tmpdir / "vib"), **vib_kwargs)
    try:
        vib.run()
    except Exception as exception:
        terminate(tmpdir, exception)

    # Summarize run
    vib.summary(log=sys.stdout if SETTINGS.DEBUG else str(tmpdir / "vib_summary.log"))

    # Perform cleanup operations
    calc_cleanup(vib.atoms, tmpdir, job_results_dir)

    return vib


def run_path_opt(
        images,
        relax_cell: bool = False,
        fmax: float = 0.01,
        max_steps: int | None = 1000,
        optimizer: Optimizer = NEBOptimizer,
        optimizer_kwargs: OptimizerKwargs | None = None,
        run_kwargs: dict[str, Any] | None = None,
        neb_kwargs: dict[str, Any] | None = None,
        copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
) -> list[Atoms]:
    """
    Run NEB


    Returns
    -------
    optimizer object
    """
    # Copy atoms so we don't modify it in-place
    images = copy_atoms(images)
    neb = NEB(images, **neb_kwargs)

    dir_lists = []
    # Perform staging operations
    # this calc_setup function is not suited for multiple Atoms objects
    for image in images:
        tmpdir_i, job_results_dir_i = calc_setup(image, copy_files=copy_files)
        dir_lists.append([tmpdir_i, job_results_dir_i])

    # Set defaults
    optimizer_kwargs = recursive_dict_merge(
        {
            "logfile": "-" if SETTINGS.DEBUG else dir_lists[0][0] / "opt.log",
            "restart": dir_lists[0][0] / "opt.json",
        },
        optimizer_kwargs,
    )
    run_kwargs = run_kwargs or {}

    # Check if trajectory kwarg is specified
    if "trajectory" in optimizer_kwargs:
        msg = "Quacc does not support setting the `trajectory` kwarg."
        raise ValueError(msg)

    # Define the Trajectory object
    traj_file = dir_lists[0][0] / "neb.traj"
    traj = Trajectory(traj_file, "w", atoms=neb)
    optimizer_kwargs["trajectory"] = traj

    # Set volume relaxation constraints, if relevant
    for image in images:
        if relax_cell and image.pbc.any():
            image = FrechetCellFilter(image)

    # Run optimization
    with traj, optimizer(neb, **optimizer_kwargs) as dyn:
        dyn.run(fmax=fmax, steps=max_steps, **run_kwargs)

    # Store the trajectory atoms
    dyn.traj_atoms = read(traj_file, index=":")

    # Perform cleanup operations
    for ii, image in enumerate(images):
        calc_cleanup(image, dir_lists[ii][0], dir_lists[ii][1])

    return dyn


def _geodesic_interpolate_wrapper(
    reactant: Atoms,
    product: Atoms,
    nimages: int = 20,
    perform_sweep: bool | None = None,
    convergence_tolerance: float = 2e-3,
    max_iterations: int = 15,
    max_micro_iterations: int = 20,
    morse_scaling: float = 1.7,
    geometry_friction: float = 1e-2,
    distance_cutoff: float = 3.0,
) -> tuple[list[str], list[list[float]]]:
    """
    Interpolates between two geometries and optimizes the path.

    Parameters:
    -----------
    reactant_product_atoms : List[Atoms]
        List of ASE Atoms objects containing initial and final geometries.
    nimages : int, optional
        Number of images for interpolation. Default is 20.
    perform_sweep : Optional[bool], optional
        Whether to sweep across the path optimizing one image at a time.
        Default is to perform sweeping updates if there are more than 35 atoms.
    convergence_tolerance : float, optional
        Convergence tolerance. Default is 2e-3.
    max_iterations : int, optional
        Maximum number of minimization iterations. Default is 15.
    max_micro_iterations : int, optional
        Maximum number of micro iterations for the sweeping algorithm. Default is 20.
    morse_scaling : float, optional
        Exponential parameter for the Morse potential. Default is 1.7.
    geometry_friction : float, optional
        Size of friction term used to prevent very large changes in geometry. Default is 1e-2.
    distance_cutoff : float, optional
        Cut-off value for the distance between a pair of atoms to be included in the coordinate system. Default is 3.0.

    Returns:
    --------
    Tuple[List[str], List[List[float]]]
        A tuple containing the list of symbols and the smoothed path.
    """
    # Read the initial geometries.
    chemical_symbols = reactant.get_chemical_symbols()

    # First redistribute number of images. Perform interpolation if too few and subsampling if too many images are given
    raw_interpolated_positions = redistribute(
        chemical_symbols,
        [reactant.positions, product.positions],
        nimages,
        tol=convergence_tolerance * 5,
    )

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    geodesic_smoother = Geodesic(
        chemical_symbols,
        raw_interpolated_positions,
        morse_scaling,
        threshold=distance_cutoff,
        friction=geometry_friction,
    )
    if perform_sweep is None:
        perform_sweep = len(chemical_symbols) > 35
    if perform_sweep:
        geodesic_smoother.sweep(
            tol=convergence_tolerance,
            max_iter=max_iterations,
            micro_iter=max_micro_iterations,
        )
    else:
        geodesic_smoother.smooth(
            tol=convergence_tolerance,
            max_iter=max_iterations,
        )
    return [Atoms(symbols=chemical_symbols, positions=geom) for geom in geodesic_smoother.path]


@requires(has_sella, "Sella must be installed. Refer to the quacc documentation.")
def _set_sella_kwargs(atoms: Atoms, optimizer_kwargs: dict[str, Any]) -> None:
    """
    Modifies the `optimizer_kwargs` in-place to address various Sella-related
    parameters. This function does the following for the specified key/value pairs in
    `optimizer_kwargs`:

    1. Sets `order = 0` if not specified (i.e. minimization rather than TS
    by default).

    2. If `internal` is not defined and not `atoms.pbc.any()`, set it to `True`.

    Parameters
    ----------
    atoms
        The Atoms object.
    optimizer_kwargs
        The kwargs for the Sella optimizer.

    Returns
    -------
    None
    """
    if "order" not in optimizer_kwargs:
        optimizer_kwargs["order"] = 0

    if not atoms.pbc.any() and "internal" not in optimizer_kwargs:
        optimizer_kwargs["internal"] = True


def _copy_intermediate_files(
    tmpdir: Path, step_number: int, files_to_ignore: list[Path] | None = None
) -> None:
    """
    Copy all files in the working directory to a subdirectory named `stepN` where `N`
    is the step number. This is useful for storing intermediate files generated during
    an ASE relaaxation.

    Parameters
    ----------
    tmpdir
        The working directory.
    step_number
        The step number.
    files_to_ignore
        A list of files to ignore when copying files to the subdirectory.

    Returns
    -------
    None
    """
    files_to_ignore = files_to_ignore or []
    store_path = tmpdir / f"step{step_number}"
    store_path.mkdir()
    for item in tmpdir.iterdir():
        if not item.name.startswith("step") and item not in files_to_ignore:
            if item.is_file():
                copy(item, store_path)
            elif item.is_dir():
                copytree(item, store_path / item.name)
