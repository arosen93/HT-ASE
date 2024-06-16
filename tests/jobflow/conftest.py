from __future__ import annotations

from pathlib import Path
from shutil import rmtree

TEST_RESULTS_DIR = Path(__file__).parent / "_test_results"
TEST_SCRATCH_DIR = Path(__file__).parent / "_test_scratch"


def pytest_sessionstart():
    import os

    file_dir = Path(__file__).parent
    os.environ["QUACC_CONFIG_FILE"] = str(file_dir / "quacc.yaml")
    os.environ["QUACC_RESULTS_DIR"] = str(TEST_RESULTS_DIR)
    os.environ["QUACC_SCRATCH_DIR"] = str(TEST_SCRATCH_DIR)
    os.environ["JOBFLOW_CONFIG_FILE"] = str(file_dir / "jobflow.yaml")

def pytest_sessionfinish(exitstatus):
    rmtree(TEST_RESULTS_DIR, ignore_errors=True)
    if exitstatus == 0:
        rmtree(TEST_SCRATCH_DIR, ignore_errors=True)
