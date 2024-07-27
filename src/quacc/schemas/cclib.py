"""Schemas for molecular DFT codes parsed by cclib."""

from __future__ import annotations

import logging
import os
from inspect import getmembers, isclass
from pathlib import Path
from typing import TYPE_CHECKING

import cclib
from ase.atoms import Atoms
from cclib.io import ccread

from quacc import QuaccDefault, get_settings
from quacc.atoms.core import get_final_atoms_from_dynamics
from quacc.schemas.ase import Summarize
from quacc.utils.dicts import finalize_dict, recursive_dict_merge
from quacc.utils.files import find_recent_logfile

if TYPE_CHECKING:
    from typing import Any

    from ase.optimize.optimize import Optimizer
    from maggma.core import Store

    from quacc.types import (
        CclibAnalysis,
        DefaultSetting,
        PopAnalysisAttributes,
        cclibASEOptSchema,
        cclibBaseSchema,
        cclibSchema,
    )


LOGGER = logging.getLogger(__name__)


class CclibSummarize:
    """
    Summarize a calculation using cclib.
    """

    def __init__(
        self,
        logfile_extensions: str | list[str],
        directory: Path | str | None = None,
        pop_analyses: list[CclibAnalysis] | None = None,
        check_convergence: bool | DefaultSetting = QuaccDefault,
        additional_fields: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the CclibSummarize object.

        Parameters
        ----------
        logfile_extensions
            Possible extensions of the log file (e.g. ".log", ".out", ".txt",
            ".chk"). Note that only a partial match is needed. For instance, `.log`
            will match `.log.gz` and `.log.1.gz`. If multiple files with this
            extension are found, the one with the most recent change time will be
            used. For an exact match only, put in the full file name.
        directory
            The path to the folder containing the calculation outputs. A value of
            None specifies the calculator directory.
        pop_analyses
            The name(s) of any cclib post-processing analysis to run. Note that for
            bader, ddec6, and hirshfeld, a cube file (.cube, .cub) must reside in
            directory. Supports: "cpsa", "mpa", "lpa", "bickelhaupt", "density",
            "mbo", "bader", "ddec6", "hirshfeld".
        check_convergence
            Whether to throw an error if geometry optimization convergence is not
            reached. Defaults to True in settings.
        additional_fields
            Additional fields to add to the task document.

        Returns
        -------
        None
        """
        self.directory = directory
        self.logfile_extensions = logfile_extensions
        self.pop_analyses = pop_analyses
        self.check_convergence = check_convergence
        self._settings = get_settings()
        self.check_convergence = (
            self._settings.CHECK_CONVERGENCE
            if self.check_convergence == QuaccDefault
            else self.check_convergence
        )
        self.additional_fields = additional_fields or {}

    def run(
        self, final_atoms: Atoms, store: Store | None | DefaultSetting = QuaccDefault
    ) -> cclibSchema:
        """
        Get tabulated results from a molecular DFT run and store them in a database-friendly
        format. This is meant to be a general parser built on top of cclib.

        Parameters
        ----------
        final_atoms
            ASE Atoms object following a calculation.
        store
            Maggma Store object to store the results in. Defaults to `QuaccSettings.STORE`

        Returns
        -------
        cclibSchema
            Dictionary representation of the task document
        """
        directory = Path(self.directory or final_atoms.calc.directory)
        store = self._settings.STORE if store == QuaccDefault else store

        # Get the cclib base task document
        cclib_task_doc = make_base_cclib_schema(
            directory, self.logfile_extensions, analysis=self.pop_analyses
        )
        attributes = cclib_task_doc["attributes"]
        metadata = attributes["metadata"]

        if self.check_convergence and attributes.get("optdone") is False:
            msg = f"Optimization not complete. Refer to {directory}"
            raise RuntimeError(msg)

        # Now we construct the input Atoms object. Note that this is not necessarily
        # the same as the initial Atoms from the relaxation because the DFT
        # package may have re-oriented the system. We only try to store the
        # input if it is XYZ-formatted though since the Atoms object does not
        # support internal coordinates or Gaussian Z-matrix.
        if metadata.get("coord_type") == "xyz" and metadata.get("coords") is not None:
            coords_obj = metadata["coords"]
            symbols = [row[0] for row in coords_obj]
            positions = [row[1:] for row in coords_obj]
            input_atoms = Atoms(symbols=symbols, positions=positions)
        else:
            input_atoms = cclib_task_doc["trajectory"][0]

        if nsteps := len([f for f in os.listdir(directory) if f.startswith("step")]):
            intermediate_cclib_task_docs = {
                "steps": {
                    n: make_base_cclib_schema(
                        directory / f"step{n}", self.logfile_extensions
                    )
                    for n in range(nsteps)
                    if (directory / f"step{n}").is_dir()
                }
            }
        else:
            intermediate_cclib_task_docs = {}

        # Get the base task document for the ASE run
        run_task_doc = Summarize(
            charge_and_multiplicity=(attributes["charge"], attributes["mult"])
        ).run(final_atoms, input_atoms, store=None)

        # Create a dictionary of the inputs/outputs
        unsorted_task_doc = (
            run_task_doc
            | intermediate_cclib_task_docs
            | cclib_task_doc
            | self.additional_fields
        )
        return finalize_dict(
            unsorted_task_doc,
            directory=directory,
            gzip_file=self._settings.GZIP_FILES,
            store=store,
        )

    def opt(
        self,
        dyn: Optimizer,
        trajectory: list[Atoms] | None = None,
        store: Store | None | DefaultSetting = QuaccDefault,
    ) -> cclibASEOptSchema:
        """
        Merges the results of a cclib run with the results of an ASE optimizer run.

        Parameters
        ----------
        dyn
            The ASE optimizer object
        trajectory
            ASE Trajectory object or list[Atoms] from reading a trajectory file. If
            None, the trajectory must be found in `dyn.trajectory.filename`.
        store
            Maggma Store object to store the results in. Defaults to `QuaccSettings.STORE`

        Returns
        -------
        cclibASEOptSchema
            Dictionary representation of the task document
        """
        store = self._settings.STORE if store == QuaccDefault else store

        final_atoms = get_final_atoms_from_dynamics(dyn)
        directory = Path(self.directory or final_atoms.calc.directory)
        cclib_summary = self.run(final_atoms, store=None)
        opt_run_summary = Summarize(
            charge_and_multiplicity=(
                cclib_summary["charge"],
                cclib_summary["spin_multiplicity"],
            ),
            additional_fields=self.additional_fields,
        ).opt(
            dyn,
            trajectory=trajectory,
            check_convergence=self.check_convergence,
            store=None,
        )
        unsorted_task_doc = recursive_dict_merge(cclib_summary, opt_run_summary)
        return finalize_dict(
            unsorted_task_doc,
            directory=directory,
            gzip_file=self._settings.GZIP_FILES,
            store=store,
        )


def make_base_cclib_schema(
    directory: str | Path,
    logfile_extensions: CclibAnalysis | list[CclibAnalysis],
    analysis: CclibAnalysis | list[CclibAnalysis] | None = None,
    proatom_dir: Path | str | None = None,
) -> cclibBaseSchema:
    """
    Create a TaskDocument from a log file.

    For a full description of each field, see
    https://cclib.github.io/data.html.

    Parameters
    ----------
    directory
        The path to the folder containing the calculation outputs.
    logfile_extensions
        Possible extensions of the log file (e.g. ".log", ".out", ".txt",
        ".chk"). Note that only a partial match is needed. For instance,
        `.log` will match `.log.gz` and `.log.1.gz`. If multiple files with
        this extension are found, the one with the most recent change time
        will be used. For an exact match only, put in the full file name.
    analysis
        The name(s) of any cclib post-processing analysis to run. Note that
        for bader, ddec6, and hirshfeld, a cube file (.cube, .cub) must be
        in dir_name. Supports: cpsa, mpa, lpa, bickelhaupt, density, mbo,
        bader, ddec6, hirshfeld.
    proatom_dir
        The path to the proatom directory if ddec6 or hirshfeld analysis are
        requested. See https://cclib.github.io/methods.html for details. If
        None, the PROATOM_DIR environment variable must point to the proatom
        directory.

    Returns
    -------
    cclibBaseSchema
        A TaskDocument dictionary summarizing the inputs/outputs of the log
        file.
    """
    # Find the most recent log file with the given extension in the
    # specified directory.
    logfile = find_recent_logfile(directory, logfile_extensions)
    if not logfile:
        msg = f"Could not find file with extension {logfile_extensions} in {directory}"
        raise FileNotFoundError(msg)

    # Let's parse the log file with cclib
    cclib_obj = ccread(logfile, logging.ERROR)
    if not cclib_obj:
        msg = f"Could not parse {logfile}"
        raise RuntimeError(msg)

    # Fetch all the attributes (i.e. all input/outputs from cclib)
    attributes = cclib_obj.getattributes()

    # monty datetime bug workaround:
    # github.com/materialsvirtuallab/monty/issues/275
    if wall_time := attributes["metadata"].get("wall_time"):
        attributes["metadata"]["wall_time"] = [*map(str, wall_time)]
    if cpu_time := attributes["metadata"].get("cpu_time"):
        attributes["metadata"]["cpu_time"] = [*map(str, cpu_time)]

    # Construct the trajectory
    coords = cclib_obj.atomcoords
    trajectory = [
        Atoms(numbers=list(cclib_obj.atomnos), positions=coord) for coord in coords
    ]

    # Get the final energy to store as its own key/value pair
    final_scf_energy = (
        cclib_obj.scfenergies[-1] if cclib_obj.scfenergies is not None else None
    )

    # Store the HOMO/LUMO energies for convenience
    if cclib_obj.moenergies is not None and cclib_obj.homos is not None:
        homo_energies, lumo_energies, gaps = get_homos_lumos(
            cclib_obj.moenergies, cclib_obj.homos
        )
        min_gap = min(gaps) if gaps else None
    else:
        homo_energies, lumo_energies, gaps, min_gap = (None, None, None, None)

    # Construct additional attributes
    additional_attributes = {
        "final_scf_energy": final_scf_energy,
        "homo_energies": homo_energies,
        "lumo_energies": lumo_energies,
        "homo_lumo_gaps": gaps,
        "min_homo_lumo_gap": min_gap,
    }

    # Calculate any population analysis properties
    popanalysis_attributes = {}
    if analysis:
        if isinstance(analysis, str):
            analysis = [analysis]
        analysis = [a.lower() for a in analysis]

        # Look for .cube or .cub files
        cubefile_path = find_recent_logfile(directory, [".cube", ".cub"])

        for analysis_name in analysis:
            if calc_attributes := cclib_calculate(
                cclib_obj, analysis_name, cubefile_path, proatom_dir
            ):
                popanalysis_attributes[analysis_name] = calc_attributes
            else:
                popanalysis_attributes[analysis_name] = None

    return {
        "logfile": str(logfile).split(":")[-1],
        "attributes": attributes | additional_attributes,
        "pop_analysis": popanalysis_attributes or None,
        "trajectory": trajectory,
    }


def cclib_calculate(
    cclib_obj,
    method: CclibAnalysis,
    cube_file: Path | str | None = None,
    proatom_dir: Path | str | None = None,
) -> PopAnalysisAttributes | None:
    """
    Run a cclib population analysis.

    Parameters
    ----------
    cclib_obj
        The cclib object to run the population analysis on.
    method
        The population analysis method to use.
    cube_file
        The path to the cube file to use for the population analysis. Needed
        only for Bader, DDEC6, and Hirshfeld
    proatom_dir
        The path to the proatom directory to use for the population analysis.
        Needed only for DDEC6 and Hirshfeld.

    Returns
    -------
    PopAnalysisAttributes | None
        The results of the population analysis.
    """
    method = method.lower()
    cube_methods = ["bader", "ddec6", "hirshfeld"]
    proatom_methods = ["ddec6", "hirshfeld"]

    if method in cube_methods:
        if not cube_file:
            msg = f"A cube file must be provided for {method}."
            raise ValueError(msg)
        if not Path(cube_file).exists():
            msg = f"Cube file {cube_file} does not exist."
            raise FileNotFoundError(msg)
    if method in proatom_methods:
        if not proatom_dir:
            if os.getenv("PROATOM_DIR") is None:
                msg = "PROATOM_DIR environment variable or proatom_dir kwarg needs to be set."
                raise OSError(msg)
            proatom_dir = os.path.expandvars(os.environ["PROATOM_DIR"])
        if not Path(proatom_dir).exists():
            msg = f"Protatom directory {proatom_dir} does not exist. Returning None."
            raise FileNotFoundError(msg)
    cclib_methods = getmembers(cclib.method, isclass)
    method_class = next(
        (
            cclib_method[1]
            for cclib_method in cclib_methods
            if cclib_method[0].lower() == method
        ),
        None,
    )
    if method_class is None:
        msg = f"{method} is not a valid cclib population analysis method."
        raise ValueError(msg)

    if method in cube_methods:
        vol = cclib.method.volume.read_from_cube(str(cube_file))
        if method in proatom_methods:
            m = method_class(cclib_obj, vol, str(proatom_dir))
        else:
            m = method_class(cclib_obj, vol)
    else:
        m = method_class(cclib_obj)

    try:
        m.calculate()
    except Exception as e:
        LOGGER.warning(f"Could not calculate {method}: {e}")
        return None

    # The list of available attributes after a calculation. This is hardcoded
    # for now until https://github.com/cclib/cclib/issues/1097 is resolved. Once
    # it is, we can delete this and just do `return
    # calc_attributes.getattributes()`.
    avail_attributes = [
        "aoresults",
        "fragresults",
        "fragcharges",
        "density",
        "donations",
        "bdonations",
        "repulsions",
        "matches",
        "refcharges",
    ]
    return {
        attribute: getattr(m, attribute)
        for attribute in avail_attributes
        if hasattr(m, attribute)
    }


def get_homos_lumos(
    moenergies: list[list[float]], homo_indices: list[int]
) -> tuple[list[float], list[float], list[float]] | tuple[list[float], None, None]:
    """
    Calculate the HOMO, LUMO, and HOMO-LUMO gap energies in eV.

    Parameters
    ----------
    moenergies
        List of MO energies. For restricted calculations, List[List[float]] is
        length one. For unrestricted, it is length two.
    homo_indices
        Indices of the HOMOs.

    Returns
    -------
    homo_energies
        The HOMO energies (eV), split by alpha and beta
    lumo_energies
        The LUMO energies (eV), split by alpha and beta
    homo_lumo_gaps
        The HOMO-LUMO gaps (eV), calculated as LUMO_alpha-HOMO_alpha and
        LUMO_beta-HOMO_beta
    """
    homo_energies = [moenergies[i][h] for i, h in enumerate(homo_indices)]
    # Make sure that the HOMO+1 (i.e. LUMO) is in moenergies (sometimes virtual
    # orbitals aren't printed in the output)
    for i, h in enumerate(homo_indices):
        if len(moenergies[i]) < h + 2:
            return homo_energies, None, None
    lumo_energies = [moenergies[i][h + 1] for i, h in enumerate(homo_indices)]
    homo_lumo_gaps = [
        lumo_energies[i] - homo_energies[i] for i in range(len(homo_energies))
    ]
    return homo_energies, lumo_energies, homo_lumo_gaps
