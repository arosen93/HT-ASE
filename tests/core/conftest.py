import os
from pathlib import Path

from maggma.stores import MemoryStore

from quacc import SETTINGS


def pytest_sessionstart():
    file_dir = Path(__file__).resolve().parent
    test_results_dir = file_dir / ".test_results"
    test_scratch_dir = file_dir / ".test_scratch"
    SETTINGS.PRIMARY_STORE = MemoryStore()
    SETTINGS.WORKFLOW_ENGINE = "local"
    SETTINGS.RESULTS_DIR = test_results_dir
    SETTINGS.SCRATCH_DIR = test_scratch_dir
    os.makedirs(test_results_dir, exist_ok=True)
    os.makedirs(test_scratch_dir, exist_ok=True)
