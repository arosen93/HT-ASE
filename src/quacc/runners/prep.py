"""Prepration for runners."""

from __future__ import annotations

import errno
import os
import shutil
import time
from logging import getLogger
from pathlib import Path
from shutil import move
from typing import TYPE_CHECKING

from monty.shutil import gzip_dir

from quacc import JobFailure, get_settings
from quacc.utils.files import copy_decompress_files, make_unique_dir

if TYPE_CHECKING:
    from ase.atoms import Atoms

    from quacc.types import Filenames, SourceDirectory

LOGGER = getLogger(__name__)


def calc_setup(
    atoms: Atoms | None,
    copy_files: SourceDirectory | dict[SourceDirectory, Filenames] | None = None,
) -> tuple[Path, Path]:
    """
    Perform staging operations for a calculation, including copying files to the scratch
    directory, setting the calculator's directory, decompressing files, and creating a
    symlink to the scratch directory.

    Parameters
    ----------
    atoms
        The Atoms object to run the calculation on. Must have a calculator
        attached. If None, no modifications to the calculator's directory will be made.
    copy_files
        Files to copy (and decompress) from source to the runtime directory.

    Returns
    -------
    Path
        The path to the unique tmpdir, where the calculation will be run. It will be
        deleted after the calculation is complete. By default, this will be
        located within `QuaccSettings.SCRATCH_DIR`, but if that is not set, it will
        be located within the `QuaccSettings.RESULTS_DIR`. For conenience, a symlink
        to this directory will be made in the `QuaccSettings.RESULTS_DIR`.
    Path
        The path to the results_dir, where the files will ultimately be stored.
        By defualt, this will be the `QuaccSettings.RESULTS_DIR`, but if
        `QuaccSettings.CREATE_UNIQUE_DIR` is set, it will be a unique directory
        within the `QuaccSettings.RESULTS_DIR`.
    """
    # Create a tmpdir for the calculation
    settings = get_settings()
    tmpdir_base = settings.SCRATCH_DIR or settings.RESULTS_DIR
    tmpdir = make_unique_dir(base_path=tmpdir_base, prefix="tmp-quacc-")
    LOGGER.info(f"Calculation will run at {tmpdir}")

    # Set the calculator's directory
    if atoms is not None:
        atoms.calc.directory = tmpdir

    # Define the results directory
    job_results_dir = settings.RESULTS_DIR
    if settings.CREATE_UNIQUE_DIR:
        job_results_dir /= f"{tmpdir.name.split('tmp-')[-1]}"

    # Create a symlink to the tmpdir
    if os.name != "nt" and settings.SCRATCH_DIR:
        symlink_path = settings.RESULTS_DIR / f"symlink-{tmpdir.name}"
        symlink_path.symlink_to(tmpdir, target_is_directory=True)

    # Copy files to tmpdir and decompress them if needed
    if copy_files:
        if isinstance(copy_files, str | Path):
            copy_files = {copy_files: "*"}

        for source_directory, filenames in copy_files.items():
            if source_directory is not None:
                copy_decompress_files(source_directory, filenames, tmpdir)

    return tmpdir, job_results_dir


def calc_cleanup(
    atoms: Atoms | None, tmpdir: Path | str, job_results_dir: Path | str
) -> None:
    """
    Perform cleanup operations for a calculation, including gzipping files, copying
    files back to the original directory, and removing the tmpdir.

    Parameters
    ----------
    atoms
        The Atoms object after the calculation. Must have a calculator
        attached. If None, no modifications to the calculator's directory will be made.
    tmpdir
        The path to the tmpdir, where the calculation will be run. It will be
        deleted after the calculation is complete.
    job_results_dir
        The path to the job_results_dir, where the files will ultimately be
        stored.

    Returns
    -------
    None
    """
    job_results_dir, tmpdir = Path(job_results_dir), Path(tmpdir)
    settings = get_settings()

    # Safety check
    if "tmp-" not in tmpdir.name:
        msg = f"{tmpdir} does not appear to be a tmpdir... exiting for safety!"
        raise ValueError(msg)

    # Update the calculator's directory
    if atoms is not None:
        atoms.calc.directory = job_results_dir

    # Gzip files in tmpdir
    if settings.GZIP_FILES:
        gzip_dir(tmpdir)

    # Move files from tmpdir to job_results_dir
    if settings.CREATE_UNIQUE_DIR:
        move(tmpdir, job_results_dir)
    else:
        for file_name in os.listdir(tmpdir):
            move(tmpdir / file_name, job_results_dir / file_name)
        rmtree(tmpdir)
    LOGGER.info(f"Calculation results stored at {job_results_dir}")

    # Remove symlink to tmpdir
    if os.name != "nt" and settings.SCRATCH_DIR:
        symlink_path = settings.RESULTS_DIR / f"symlink-{tmpdir.name}"
        symlink_path.unlink(missing_ok=True)


def terminate(tmpdir: Path | str, exception: Exception) -> None:
    """
    Terminate a calculation and move files to a failed directory.

    Parameters
    ----------
    tmpdir
        The path to the tmpdir, where the calculation was run.
    exception
        The exception that caused the calculation to fail.

    Returns
    -------
    None

    Raises
    -------
    JobFailure
        The exception that caused the calculation to fail plus additional
        metadata.
    """
    tmpdir = Path(tmpdir)
    settings = get_settings()
    job_failed_dir = tmpdir.with_name(tmpdir.name.replace("tmp-", "failed-"))
    tmpdir.rename(job_failed_dir)

    msg = f"Calculation failed! Files stored at {job_failed_dir}"
    LOGGER.info(msg)

    if os.name != "nt" and settings.SCRATCH_DIR:
        old_symlink_path = settings.RESULTS_DIR / f"symlink-{tmpdir.name}"
        symlink_path = settings.RESULTS_DIR / f"symlink-{job_failed_dir.name}"
        old_symlink_path.unlink(missing_ok=True)
        symlink_path.symlink_to(job_failed_dir, target_is_directory=True)

    raise JobFailure(job_failed_dir, message=msg, parent_error=exception) from exception


def rmtree(*args, max_retries=3, retry_wait=10, **kwargs):
    """
    rmtree that will retry if we get common NFS errors (files still being deleted, etc)
    Adapted from https://github.com/teojgo/reframe/blob/master/reframe/utility/osext.py
    """
    if "onerror" in kwargs and kwargs["onerror"] is not None:
        shutil.rmtree(*args, **kwargs)
        return

    for i in range(max_retries):
        try:
            shutil.rmtree(*args, **kwargs)
            return
        except OSError as e:
            if i == max_retries:
                raise
            if e.errno in {errno.ENOTEMPTY, errno.EBUSY}:
                time.sleep(retry_wait)
            else:
                raise
