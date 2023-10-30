"""A Q-Chem calculator built on Pymatgen and Custodian functionality"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING

from ase.calculators.calculator import FileIOCalculator
from pymatgen.io.qchem.inputs import QCInput
from pymatgen.io.qchem.sets import QChemDictSet
from pymatgen.io.qchem.utils import lower_and_check_unique

from quacc.calculators.qchem import custodian
from quacc.calculators.qchem.io import read_qchem, write_qchem
from quacc.calculators.qchem.params import get_molecule, get_rem_swaps

if TYPE_CHECKING:
    from typing import Any, ClassVar, Literal

    from ase import Atoms

    from quacc.calculators.qchem.io import Results


class QChem(FileIOCalculator):
    """
    Custom Q-Chem calculator built on Pymatgen and Custodian.
    """

    implemented_properties: ClassVar[list[str]] = [
        "energy",
        "forces",
        "hessian",
        "enthalpy",
        "entropy",
        "qc_output",
        "qc_input",
        "custodian",
    ]
    results: ClassVar[Results] = {}

    def __init__(
        self,
        atoms: Atoms | list[Atoms] | Literal["read"],
        charge: int,
        spin_multiplicity: int,
        rem: dict,
        opt: dict[str, list[str]] | None = None,
        pcm: dict | None = None,
        solvent: dict | None = None,
        smx: dict | None = None,
        scan: dict[str, list] | None = None,
        van_der_waals: dict[str, float] | None = None,
        vdw_mode: Literal["atomic", "sequential"] = "atomic",
        plots: dict | None = None,
        nbo: dict | None = None,
        geom_opt: dict | None = None,
        cdft: list[list[dict]] | None = None,
        almo_coupling: list[list[tuple[int, int]]] | None = None,
        svp: dict | None = None,
        pcm_nonels: dict | None = None,
        qchem_dict_set_kwargs: dict[str, Any] | None = None,
        **fileiocalculator_kwargs,
    ) -> None:
        """
        Initialize the Q-Chem calculator. Most of the input parameters here
        are meant to mimic those in `pymatgen.io.qchem.inputs.QCInput`. See
        the documentation for that class for more information.

        Parameters
        ----------
        atoms
            The Atoms object to be used for the calculation. "read" can be used
            in multi_job QChem input files where the molecule is read in from
            the previous calculation.
        charge
            The total charge of the molecular system.
        spin_multiplicity
            The spin multiplicity of the molecular system.
        rem
            A dictionary of all the input parameters for the rem section of
            QChem input file. e.g. rem = {'method': 'rimp2', 'basis': '6-31*G++'
            ... }
        opt
            A dictionary of opt sections, where each opt section is a key and
            the corresponding values are a list of strings. Strings must be
            formatted as instructed by the QChem manual. The different opt
            sections are: CONSTRAINT, FIXED, DUMMY, and CONNECT e.g. opt =
            {"CONSTRAINT": ["tors 2 3 4 5 25.0", "tors 2 5 7 9 80.0"], "FIXED":
            ["2 XY"]}
        pcm
            A dictionary of the PCM section, defining behavior for use of the
            polarizable continuum model. e.g. pcm = {"theory": "cpcm",
            "hpoints": 194}
        solvent
            A dictionary defining the solvent parameters used with PCM. e.g.
            solvent = {"dielectric": 78.39, "temperature": 298.15}
        smx
            A dictionary defining solvent parameters used with the SMD method, a
            solvent method that adds short-range terms to PCM. e.g. smx =
            {"solvent": "water"}
        scan
            A dictionary of scan variables. Because two constraints of the same
            type are allowed (for instance, two torsions or two bond stretches),
            each TYPE of variable (stre, bend, tors) should be its own key in
            the dict, rather than each variable. Note that the total number of
            variable (sum of lengths of all lists) CANNOT be more than two. e.g.
            scan = {"stre": ["3 6 1.5 1.9 0.1"], "tors": ["1 2 3 4 -180 180
            15"]}
        van_der_waals
            A dictionary of custom van der Waals radii to be used when
            constructing cavities for the PCM model or when computing, e.g.
            Mulliken charges. They keys are strs whose meaning depends on the
            value of vdw_mode, and the values are the custom radii in angstroms.
        vdw_mode
            Method of specifying custom van der Waals radii - 'atomic' or
            'sequential'. In 'atomic' mode (default), dict keys represent the
            atomic number associated with each radius (e.g., 12 = carbon). In
            'sequential' mode, dict keys represent the sequential position of a
            single specific atom in the input structure.
        plots
            A dictionary of all the input parameters for the plots section of
            the QChem input file.
        nbo
            A dictionary of all the input parameters for the nbo section of the
            QChem input file.
        geom_opt
            A dictionary of input parameters for the geom_opt section of the
            QChem input file. This section is required when using the libopt3
            geometry optimizer.
        cdft
            A list of lists of dictionaries, where each dictionary represents a
            charge constraint in the cdft section of the QChem input file. Each
            entry in the main list represents one state (allowing for
            multi-configuration calculations using constrained density
            functional theory - configuration interaction (CDFT-CI). Each state
            is represented by a list, which itself contains some number of
            constraints (dictionaries).

            1. For a single-state calculation with two constraints:
            cdft=[[
                {"value": 1.0, "coefficients": [1.0], "first_atoms": [1],
                "last_atoms": [2], "types": [None]}, {"value": 2.0,
                "coefficients": [1.0, -1.0], "first_atoms": [1, 17],
                "last_atoms": [3, 19],
                    "types": ["s"]}
            ]]

            Note that a type of None will default to a charge constraint (which
            can also be accessed by requesting a type of "c" or "charge".

            2. For a multi-reference calculation:
            cdft=[
                [
                    {"value": 1.0, "coefficients": [1.0], "first_atoms": [1],
                    "last_atoms": [27],
                        "types": ["c"]},
                    {"value": 0.0, "coefficients": [1.0], "first_atoms": [1],
                    "last_atoms": [27],
                        "types": ["s"]},
                ], [
                    {"value": 0.0, "coefficients": [1.0], "first_atoms": [1],
                    "last_atoms": [27],
                        "types": ["c"]},
                    {"value": -1.0, "coefficients": [1.0], "first_atoms": [1],
                    "last_atoms": [27],
                        "types": ["s"]},
                ]
            ]
        almo_coupling
            A list of lists of int 2-tuples used for calculations of
            diabatization and state coupling calculations
                relying on the absolutely localized molecular orbitals (ALMO)
                methodology. Each entry in the main list represents a single
                state (two states are included in an ALMO calculation). Within a
                single state, each 2-tuple represents the charge and spin
                multiplicity of a single fragment.
            e.g. almo=[[(1, 2), (0, 1)], [(0, 1), (1, 2)]]
        svp
            TODO.
        pcm_nonels
            TODO.
        qchem_dict_set_kwargs
            Arguments to be passed to `pymatgen.io.qchem.sets.QChemDictSet`,
            which will generate a `QCInput`. If specified, this will be used
            directly to instantiate a custom input set, overriding any other
            specified kwargs (atoms, charge, and spin_multiplicity will be used
            as-provided).
        **fileiocalculator_kwargs
            Additional arguments to be passed to
            `ase.calculators.calculator.FileIOCalculator`.

        Returns
        -------
        None
        """

        # Assign variables to self
        self.atoms = atoms
        self.charge = charge
        self.spin_multiplicity = spin_multiplicity
        self.rem = rem
        self.opt = opt
        self.pcm = pcm
        self.solvent = solvent
        self.smx = smx
        self.scan = scan
        self.van_der_waals = van_der_waals
        self.vdw_mode = vdw_mode
        self.plots = plots
        self.nbo = nbo
        self.geom_opt = geom_opt
        self.cdft = cdft
        self.almo_coupling = almo_coupling
        self.svp = svp
        self.pcm_nonels = pcm_nonels
        self.qchem_dict_set_kwargs = qchem_dict_set_kwargs or {}
        self.fileiocalculator_kwargs = fileiocalculator_kwargs

        # Instantiate previous orbital coefficients
        self.default_parameters = None
        self._prev_orbital_coeffs = None

        if "directory" in self.fileiocalculator_kwargs:
            raise NotImplementedError("The directory kwarg is not supported.")

        # Clean up parameters
        self._cleanup_attrs()

        # Get Q-Chem executable command
        self.command = self._manage_environment()

        # Instantiate the calculator
        FileIOCalculator.__init__(
            self,
            restart=None,
            ignore_bad_restart_file=FileIOCalculator._deprecated,
            label=None,
            atoms=self.atoms,
            **self.fileiocalculator_kwargs,
        )

    def write_input(
        self,
        atoms: Atoms,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        """
        Write the Q-Chem input files.

        Parameters
        ----------
        atoms
            The Atoms object to be used for the calculation.
        properties
            List of properties to calculate.
        system_changes
            List of changes to the system since last calculation.

        Returns
        -------
        None
        """
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        if self.qchem_dict_set_kwargs:
            qc_input = QChemDictSet(self._molecule, **self.qchem_dict_set_kwargs)
        else:
            qc_input = QCInput(
                self._molecule,
                self.rem,
                opt=self.opt,
                pcm=self.pcm,
                solvent=self.solvent,
                smx=self.smx,
                scan=self.scan,
                van_der_waals=self.van_der_waals,
                vdw_mode=self.vdw_mode,
                plots=self.plots,
                nbo=self.nbo,
                geom_opt=self.geom_opt,
                cdft=self.cdft,
                almo_coupling=self.almo_coupling,
                svp=self.svp,
                pcm_nonels=self.pcm_nonels,
            )
        write_qchem(
            qc_input,
            prev_orbital_coeffs=self._prev_orbital_coeffs,
        )

    def read_results(self) -> None:
        """
        Read the Q-Chem output files. Update the .results and
        ._prev_orbital_coeffs attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        results, prev_orbital_coeffs = read_qchem()
        self.results = results
        self._prev_orbital_coeffs = prev_orbital_coeffs

    def _manage_environment(self) -> str:
        """
        Return the command to run the Q-Chem calculator via Custodian.

        Returns
        -------
        str
            The command flag to run Q-Chem with Custodian.
        """

        qchem_custodian_script = Path(inspect.getfile(custodian)).resolve()
        return f"python {qchem_custodian_script}"

    def _cleanup_attrs(self) -> None:
        """
        Clean up self attribute parameters.
        """
        self.rem = get_rem_swaps(self.rem)
        for attr in [
            "rem",
            "pcm",
            "solvent",
            "smx",
            "scan",
            "van_der_waals",
            "plots",
            "nbo",
            "geom_opt",
            "svp",
            "pcm_nonels",
        ]:
            attr_val = lower_and_check_unique(getattr(self, attr))
            setattr(self, attr, attr_val)

        self._molecule = get_molecule(self.atoms, self.charge, self.spin_multiplicity)
        self._set_default_params()

    def _set_default_params(self) -> None:
        """
        Store the parameters that have been passed to the Q-Chem calculator in
        FileIOCalculator's self.default_parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        params = {
            "charge": self.charge,
            "spin_multiplicity": self.spin_multiplicity,
            "rem": self.rem,
            "opt": self.opt,
            "pcm": self.pcm,
            "solvent": self.solvent,
            "smx": self.smx,
            "scan": self.scan,
            "van_der_waals": self.van_der_waals,
            "vdw_mode": self.vdw_mode,
            "plots": self.plots,
            "nbo": self.nbo,
            "geom_opt": self.geom_opt,
            "cdft": self.cdft,
            "almo_coupling": self.almo_coupling,
            "svp": self.svp,
            "pcm_nonels": self.pcm_nonels,
        }

        self.default_parameters = {k: v for k, v in params.items() if v is not None}
