"""Utility functions for running ASE calculators with ASE-based methods."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from shutil import copy, copytree
from typing import TYPE_CHECKING, Callable

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import Trajectory, read
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from monty.dev import requires
from monty.os.path import zpath

from quacc import SETTINGS
from quacc.atoms.core import copy_atoms
from quacc.runners.prep import calc_cleanup, calc_setup, terminate
from quacc.utils.dicts import recursive_dict_merge

has_sella = bool(find_spec("sella"))

if has_sella:
    pass


if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, TypedDict

    from ase.atoms import Atoms
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
        Type hint for `optimizer_kwargs` in [quacc.runners.ase.Runner.run_opt][].
        """

        restart: Path | str | None  # default = None
        append_trajectory: bool  # default = False

    class VibKwargs(TypedDict, total=False):
        """
        Type hint for `vib_kwargs` in [quacc.runners.ase.Runner.run_vib][].
        """

        indices: list[int] | None  # default = None
        delta: float  # default = 0.01
        nfree: int  # default = 2


class Runner:
    """
    Run various types of calculations in a scratch directory and copy the results back
    to the original directory.
    """

    def __init__(
        self,
        atoms: Atoms | None,
        copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
    ) -> None:
        """
        Initialize the Runner object.

            Files to copy (and decompress) from source to the runtime directory.
        Parameters
        ----------
        atoms
            The Atoms object to run calculations on.
        copy_files
            Files to copy (and decompress) from source to the runtime directory.

        Returns
        -------
        None
        """
        self.atoms = copy_atoms(atoms) if atoms else atoms
        self.copy_files = copy_files
        self.tmpdir, self.job_results_dir = calc_setup(
            self.atoms, copy_files=self.copy_files
        )

    def run_calc(self, geom_file: str | None = None, get_forces: bool = False) -> Atoms:
        """
        This is a wrapper around `atoms.get_potential_energy()`. Note: This function
        does not modify the atoms object in-place.

        Parameters
        ----------
        geom_file
            The filename of the log file that contains the output geometry, used to
            update the atoms object's positions and cell after a job. It is better
            to specify this rather than relying on ASE's
            `atoms.get_potential_energy()` function to update the positions, as the
            latter behavior varies between codes.
        get_forces
            Whether to use `atoms.get_forces()` instead of `atoms.get_potential_energy()`.

        Returns
        -------
        Atoms
            The updated Atoms object.
        """
        # Run calculation
        try:
            if get_forces:
                self.atoms.get_forces()
            else:
                self.atoms.get_potential_energy()
        except Exception as exception:
            terminate(self.tmpdir, exception)

        # Most ASE calculators do not update the atoms object in-place with a call
        # to .get_potential_energy(), which is important if an internal optimizer is
        # used. This section is done to ensure that the atoms object is updated to
        # the final geometry if `geom_file` is provided.
        # Note: We have to be careful to make sure we don't lose the calculator
        # object, as this contains important information such as the parameters
        # and output properties (e.g. final magnetic moments).
        if geom_file:
            atoms_new = read(zpath(self.tmpdir / geom_file))
            if isinstance(atoms_new, list):
                atoms_new = atoms_new[-1]

            # Make sure the atom indices didn't get updated somehow (sanity check).
            # If this happens, there is a serious problem.
            if (
                np.array_equal(
                    atoms_new.get_atomic_numbers(), self.atoms.get_atomic_numbers()
                )
                is False
            ):
                raise ValueError(
                    "Atomic numbers do not match between atoms and geom_file."
                )

            self.atoms.positions = atoms_new.positions
            self.atoms.cell = atoms_new.cell

        # Perform cleanup operations
        self._cleanup()

        return self.atoms

    def run_opt(
        self,
        relax_cell: bool = False,
        fmax: float = 0.01,
        max_steps: int = 1000,
        optimizer: Optimizer = BFGS,
        optimizer_kwargs: OptimizerKwargs | None = None,
        store_intermediate_results: bool = False,
        fn_hook: Callable | None = None,
        run_kwargs: dict[str, Any] | None = None,
    ) -> Optimizer:
        """
        This is a wrapper around the optimizers in ASE. Note: This function does not
        modify the atoms object in-place.

        Parameters
        ----------
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
            Dictionary of kwargs for the `run()` method of the optimizer.

        Returns
        -------
        Optimizer
            The ASE Optimizer object.
        """
        # Set defaults
        optimizer_kwargs = recursive_dict_merge(
            {
                "logfile": "-" if SETTINGS.DEBUG else self.tmpdir / "opt.log",
                "restart": self.tmpdir / "opt.json",
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
            self._set_sella_kwargs(optimizer_kwargs)
        elif optimizer.__name__ == "IRC":
            optimizer_kwargs.pop("restart", None)

        # Define the Trajectory object
        traj_file = self.tmpdir / "opt.traj"
        traj = Trajectory(traj_file, "w", atoms=self.atoms)
        optimizer_kwargs["trajectory"] = traj

        # Set volume relaxation constraints, if relevant
        if relax_cell and self.atoms.pbc.any():
            self.atoms = FrechetCellFilter(self.atoms)

        # Run optimization
        try:
            with traj, optimizer(self.atoms, **optimizer_kwargs) as dyn:
                if optimizer.__name__.startswith("SciPy"):
                    # https://gitlab.coms/ase/ase/-/issues/1475
                    dyn.run(fmax=fmax, steps=max_steps, **run_kwargs)
                else:
                    for i, _ in enumerate(
                        dyn.irun(fmax=fmax, steps=max_steps, **run_kwargs)
                    ):
                        if store_intermediate_results:
                            self._copy_intermediate_files(
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
            terminate(self.tmpdir, exception)

        # Store the trajectory atoms
        dyn.traj_atoms = read(traj_file, index=":")

        # Perform cleanup operations
        self._cleanup()

        return dyn

    def run_vib(self, vib_kwargs: VibKwargs | None = None) -> Vibrations:
        """
        Run an ASE-based vibration analysis in a scratch directory and copy the results back
        to the original directory. This can be useful if file I/O is slow in the working
        directory, so long as file transfer speeds are reasonable.

        This is a wrapper around the vibrations module in ASE. Note: This function
        does not modify the atoms object in-place.

        Parameters
        ----------
        vib_kwargs
            Dictionary of kwargs for the [ase.vibrations.Vibrations][] class.

        Returns
        -------
        Vibrations
            The updated Vibrations module
        """
        # Set defaults
        vib_kwargs = vib_kwargs or {}

        # Run calculation
        vib = Vibrations(self.atoms, name=str(self.tmpdir / "vib"), **vib_kwargs)
        try:
            vib.run()
        except Exception as exception:
            terminate(self.tmpdir, exception)

        # Summarize run
        vib.summary(
            log=sys.stdout if SETTINGS.DEBUG else str(self.tmpdir / "vib_summary.log")
        )

        # Perform cleanup operations
        self._cleanup()

        return vib

    def _cleanup(self) -> None:
        """
        Perform cleanup operations on the scratch directory.

        Returns
        -------
        None
        """
        calc_cleanup(self.atoms, self.tmpdir, self.job_results_dir)

    def _copy_intermediate_files(
        self, step_number: int, files_to_ignore: list[Path] | None = None
    ) -> None:
        """
        Copy all files in the working directory to a subdirectory named `stepN` where `N`
        is the step number. This is useful for storing intermediate files generated during
        an ASE relaaxation.

        Parameters
        ----------
        step_number
            The step number.
        files_to_ignore
            A list of files to ignore when copying files to the subdirectory.

        Returns
        -------
        None
        """
        files_to_ignore = files_to_ignore or []
        store_path = self.tmpdir / f"step{step_number}"
        store_path.mkdir()
        for item in self.tmpdir.iterdir():
            if not item.name.startswith("step") and item not in files_to_ignore:
                if item.is_file():
                    copy(item, store_path)
                elif item.is_dir():
                    copytree(item, store_path / item.name)

    @requires(has_sella, "Sella must be installed. Refer to the quacc documentation.")
    def _set_sella_kwargs(self, optimizer_kwargs: dict[str, Any]) -> None:
        """
        Modifies the `optimizer_kwargs` in-place to address various Sella-related
        parameters. This function does the following for the specified key/value pairs in
        `optimizer_kwargs`:

        1. Sets `order = 0` if not specified (i.e. minimization rather than TS
        by default).

        2. If `internal` is not defined and not `atoms.pbc.any()`, set it to `True`.

        Parameters
        ----------
        optimizer_kwargs
            The kwargs for the Sella optimizer.

        Returns
        -------
        None
        """
        if "order" not in optimizer_kwargs:
            optimizer_kwargs["order"] = 0

        if not self.atoms.pbc.any() and "internal" not in optimizer_kwargs:
            optimizer_kwargs["internal"] = True
