"""Common workflows for phonons."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

from monty.dev import requires

from quacc import job, subflow
from quacc.atoms.phonons import get_phonopy, phonopy_atoms_to_ase_atoms
from quacc.runners.phonons import run_phonopy
from quacc.schemas.phonons import summarize_phonopy

has_deps = find_spec("phonopy") is not None and find_spec("seekpath") is not None

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms

    from quacc import Job
    from quacc.schemas._aliases.phonons import PhononSchema


@subflow
@requires(
    has_deps, "Phonopy and seekpath must be installed. Run `pip install quacc[phonons]`"
)
def phonon_subflow(
    atoms: Atoms,
    force_job: Job,
    relax_job: Job | None = None,
    symprec: float = 1e-4,
    min_lengths: float | tuple[float, float, float] | None = 20.0,
    supercell_matrix: (
        tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None
    ) = None,
    displacement: float = 0.01,
    t_step: float = 10,
    t_min: float = 0,
    t_max: float = 1000,
    phonopy_kwargs: dict[str, Any] | None = None,
    additional_fields: dict[str, Any] | None = None,
) -> PhononSchema:
    """
    Calculate phonon properties.

    Parameters
    ----------
    atoms
        Atoms object with calculator attached.
    force_job
        The static job to calculate the forces.
    relax_job
        The job used to relax the structure before calculating the forces.
    symprec
        Precision for symmetry detection.
    min_lengths
        Minimum length of each lattice dimension (A).
    supercell_matrix
        The supercell matrix to use. If specified, it will override any
        value specified by `min_lengths`.
    displacement
        Atomic displacement (A).
    t_step
        Temperature step (K).
    t_min
        Min temperature (K).
    t_max
        Max temperature (K).
    phonopy_kwargs
        Additional kwargs to pass to the Phonopy class.
    additional_fields
        Additional fields to store in the database.

    Returns
    -------
    PhononSchema
        Dictionary of results from [quacc.schemas.phonons.summarize_phonopy][]
    """
    fixed_atoms = fixed_atoms or []
    fixed_atoms = np.array([i in fixed_atoms for i in range(len(atoms))])

    phonon, atoms_to_add = get_phonopy(
        atoms,
        fixed_atoms=fixed_atoms,
        min_lengths=min_lengths,
        supercell_matrix=supercell_matrix,
        symprec=symprec,
        displacement=displacement,
        phonopy_kwargs=phonopy_kwargs,
    )

    fixed_atoms = np.full(len(phonon.supercell), False)
    fixed_atoms = np.append(fixed_atoms, [True] * len(atoms_to_add))
    fixed_atoms = fixed_atoms.astype(bool)

    supercells = [
        phonopy_atoms_to_ase_atoms(s) + atoms_to_add
        for s in phonon.supercells_with_displacements
    ]

    @subflow
    def _get_forces_subflow(supercells: list[Atoms]) -> list[dict]:
        return [
            force_job(supercell) for supercell in supercells if supercell is not None
        ]

    @job
    def _thermo_job(atoms: Atoms, force_job_results: list[dict]) -> PhononSchema:
        phonon = get_phonopy(
            atoms,
            min_lengths=min_lengths,
            supercell_matrix=supercell_matrix,
            symprec=symprec,
            displacement=displacement,
            phonopy_kwargs=phonopy_kwargs,
        )
        parameters = force_job_results[-1].get("parameters")
        forces = [
            output["results"]["forces"][~fixed_atoms, :] for output in force_job_results
        ]
        phonon_results = run_phonopy(
            phonon,
            forces,
            symmetrize=fixed_atoms.any(),
            t_step=t_step,
            t_min=t_min,
            t_max=t_max,
        )

        return summarize_phonopy(
            phonon,
            atoms,
            phonon_results.directory,
            parameters=parameters,
            additional_fields=additional_fields,
        )

    if relax_job is not None:
        atoms = relax_job(atoms)["atoms"]

    force_job_results = _get_forces_subflow(supercells)
    return _thermo_job(atoms, force_job_results)
