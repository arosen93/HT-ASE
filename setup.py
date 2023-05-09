from pathlib import Path

from setuptools import find_packages, setup

module_dir = Path(__file__).resolve().parent

with open(module_dir / "README.md") as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(
        name="quacc",
        description="A platform to enable high-throughput, database-driven quantum chemistry and computational materials science",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Andrew S. Rosen",
        author_email="asrosen93@gmail.com",
        url="https://github.com/arosen93/quacc",
        python_requires=">=3.8.0",
        version="0.0.6",
        packages=find_packages(),
        license="modified BSD",
        keywords="high-throughput automated workflow dft",
        data_files=["LICENSE.md"],
        zip_safe=False,
        install_requires=[
            "ase @ git+https://gitlab.com/ase/ase.git",
            "atomate2 @ git+https://github.com/materialsproject/atomate2.git",
            "pymatgen==2023.5.8",
            "jobflow==0.1.11",
            "emmet-core==0.54.3",
            "cclib==1.7.2",
            "monty==2023.5.8",
            "numpy",
        ],
        extras_require={
            "covalent": [
                "covalent==0.220.0",
                "covalent-slurm-plugin==0.12.1",
                "covalent-ssh-plugin==0.17.0",
                "maggma==0.50.3",
            ],
            "fireworks": ["fireworks==2.0.3", "jobflow==0.1.11"],
            "vasp": ["custodian==2023.5.7"],
            "xtb": ["tblite[ase]==0.3.0", "xtb==22.1"],
            "docs": [
                "sphinx==6.2.1",
                "furo==2023.3.27",
                "m2r2==0.3.3.post2",
                "ipython==8.13.2",
                "nbsphinx==0.9.1",
                "nbsphinx-link==1.3.0",
                "autodoc_pydantic==1.8.0",
            ],
            "dev": ["pytest==7.3.1", "black==23.3.0", "isort==5.12.0"],
        },
        tests_require=["pytest"],
        include_package_data=True,
    )
