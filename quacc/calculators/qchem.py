"""
A Q-Chem calculator built on Pymatgen and Custodian functionality
"""
from __future__ import annotations

import inspect
import os
import struct

from ase import Atoms, units
from ase.calculators.calculator import FileIOCalculator
from monty.io import zopen
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.qchem.outputs import QCOutput
from pymatgen.io.qchem.sets import ForceSet

from quacc.custodian import qchem as custodian_qchem


class QChem(FileIOCalculator):
    """

    Parameters
    ----------
    input_atoms
        The Atoms object to be used for the calculation.
    cores
        Number of cores to use for the Q-Chem calculation.
    charge
        The total charge of the molecular system.
        Effectively defaults to zero.
    spin_multiplicity
        The spin multiplicity of the molecular system.
        Effectively defaults to the lowest spin state given the molecular structure and charge.
    qchem_input_params
        Dictionary of Q-Chem input parameters to be passed to pymatgen's ForceSet.
    use_custodian
        Whether to use Custodian to run Q-Chem.
        Default is True in settings.
    **kwargs
        Additional arguments to be passed to the Q-Chem calculator. Takes all valid
        ASE calculator arguments, in addition to those custom to Quacc.

    Returns
    -------
    Atoms
        The ASE Atoms object with attached Q-Chem calculator.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        input_atoms: Atoms,
        cores: None | int = None,
        charge: None | int = None,
        spin_multiplicity: None | int = None,
        qchem_input_params: dict = None,
        **kwargs,
    ):
        # Assign variables to self
        self.input_atoms = input_atoms
        self.cores = cores
        self.charge = charge
        self.spin_multiplicity = spin_multiplicity
        self.qchem_input_params = qchem_input_params or {}
        self.kwargs = kwargs
        self.default_parameters = {
            "cores": self.cores,
            "charge": self.charge,
            "spin_multiplicity": self.spin_multiplicity,
        }
        for key in self.qchem_input_params:
            if key == "overwrite_inputs":
                for subkey in self.qchem_input_params[key]:
                    for subsubkey in self.qchem_input_params[key][subkey]:
                        self.default_parameters[
                            "overwrite_" + subkey + "_" + subsubkey
                        ] = self.qchem_input_params[key][subkey][subsubkey]
            else:
                self.default_parameters[key] = self.qchem_input_params[key]

        # Get Q-Chem executable command
        self.command = self._manage_environment()

        # Instantiate the calculator
        FileIOCalculator.__init__(
            self,
            restart=None,
            ignore_bad_restart_file=FileIOCalculator._deprecated,
            label=None,
            atoms=self.input_atoms,
            **self.kwargs,
        )

    def _manage_environment(self) -> str:
        """
        Manage the environment for the Q-Chem calculator.

        Returns
        -------
        str
            The command flag to pass to the Q-Chem calculator.
        """

        # Return the command flag
        run_qchem_custodian_file = os.path.abspath(inspect.getfile(custodian_qchem))
        if self.cores is not None:
            return f"python {run_qchem_custodian_file} {self.cores}"
        else:
            return f"python {run_qchem_custodian_file}"

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        mol = AseAtomsAdaptor.get_molecule(atoms)
        if self.charge is not None:
            if self.spin_multiplicity is not None:
                mol.set_charge_and_spin(
                    charge=self.charge, spin_multiplicity=self.spin_multiplicity
                )
            else:
                mol.set_charge_and_spin(charge=self.charge)
        qcin = ForceSet(mol, **self.qchem_input_params)
        qcin.write("mol.qin")

    def read_results(self):
        data = QCOutput("mol.qout").data
        self.results["energy"] = data["final_energy"] * units.Hartree
        tmp_grad_data = []
        with zopen("131.0", mode="rb") as file:
            binary = file.read()
            for ii in range(int(len(binary) / 8)):
                tmp_grad_data.append(
                    struct.unpack("d", binary[ii * 8 : (ii + 1) * 8])[0]
                )
        grad = []
        for ii in range(int(len(tmp_grad_data) / 3)):
            grad.append(
                [
                    float(tmp_grad_data[ii * 3]),
                    float(tmp_grad_data[ii * 3 + 1]),
                    float(tmp_grad_data[ii * 3 + 2]),
                ]
            )
        if data["pcm_gradients"] is not None:
            gradient = data["pcm_gradients"][0]
        else:
            gradient = data["gradients"][0]
        for ii, subgrad in enumerate(grad):
            for jj, val in enumerate(subgrad):
                if abs(gradient[ii, jj] - val) > 1e-6:
                    raise ValueError(
                        "Difference between gradient value in scratch file vs. output file should not be this large! Exiting..."
                    )
                gradient[ii, jj] = val
        self.results["forces"] = gradient * (-units.Hartree / units.Bohr)
