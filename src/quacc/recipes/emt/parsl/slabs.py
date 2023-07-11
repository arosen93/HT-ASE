"""Slab recipes for EMT"""
from __future__ import annotations

from ase import Atoms
from parsl import join_app, python_app
from parsl.app.python import PythonApp
from parsl.dataflow.futures import AppFuture

from quacc.recipes.emt.core import relax_job, static_job
from quacc.util.slabs import make_max_slabs_from_bulk


# See https://github.com/Parsl/parsl/issues/2793 for why we need to strip the @ct.electron
# decorator off the PythonApp kwargs
def bulk_to_slabs_flow(
    atoms: Atoms | dict,
    slabgen_kwargs: dict | None = None,
    slab_relax: PythonApp = python_app(relax_job.electron_object.function),
    slab_static: PythonApp | None = python_app(static_job.electron_object.function),
    slab_relax_kwargs: dict | None = None,
    slab_static_kwargs: dict | None = None,
) -> AppFuture:
    """
    Workflow consisting of:

    1. Slab generation

    2. Slab relaxations

    3. Slab statics (optional)

    Parameters
    ----------
    atoms
        Atoms object or a dictionary with the key "atoms" and an Atoms object as the value
    slabgen_kwargs
        Additional keyword arguments to pass to make_max_slabs_from_bulk()
    slab_relax
        Default PythonApp to use for the relaxation of the slab structures.
    slab_static
        Default PythonApp to use for the static calculation of the slab structures.
    slab_relax_kwargs
        Additional keyword arguments to pass to the relaxation calculation.
    slab_static_kwargs
        Additional keyword arguments to pass to the static calculation.

    Returns
    -------
    AppFuture
        An AppFuture whose .result() is a list[dict]
    """
    atoms = atoms if isinstance(atoms, Atoms) else atoms["atoms"]
    slab_relax_kwargs = slab_relax_kwargs or {}
    slab_static_kwargs = slab_static_kwargs or {}
    slabgen_kwargs = slabgen_kwargs or {}

    if "relax_cell" not in slab_relax_kwargs:
        slab_relax_kwargs["relax_cell"] = False

    @join_app
    def _relax_distributed(slabs):
        return [slab_relax(slab, **slab_relax_kwargs) for slab in slabs]

    @join_app
    def _relax_and_static_distributed(slabs):
        return [
            slab_static(
                slab_relax(slab, **slab_relax_kwargs),
                **slab_static_kwargs,
            )
            for slab in slabs
        ]

    slabs = make_max_slabs_from_bulk(atoms, **slabgen_kwargs)

    if slab_static is None:
        return _relax_distributed(slabs)

    return _relax_and_static_distributed(slabs)
