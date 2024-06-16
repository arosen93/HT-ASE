from __future__ import annotations

import pytest

prefect = pytest.importorskip("prefect")

from pathlib import Path

from quacc import (
    SETTINGS,
    change_settings,
    flow,
    job,
    redecorate,
    strip_decorator,
    subflow,
)
from quacc.wflow_tools.customizers import customize_funcs


def test_patch():
    @job
    def task1(val: float) -> dict:
        return {"input": val, "result": val * 100}

    @job
    def task2(val: float) -> dict:
        return {"input": val, "result": val * 200}

    @flow
    def workflow(val):
        future1 = task1(val)
        return task2(future1["result"])

    assert workflow(1).result() == {"input": 100, "result": 20000}


def test_patch_local():
    with change_settings({"PREFECT_AUTO_SUBMIT": False}):

        @job
        def task1(val: float) -> dict:
            return {"input": val, "result": val * 100}

        @job
        def task2(val: float) -> dict:
            return {"input": val, "result": val * 200}

        @flow
        def workflow(val):
            future1 = task1(val)
            return task2(future1["result"])

        assert workflow(1) == {"input": 100, "result": 20000}


def test_prefect_decorators(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @job
    def add(a, b):
        return a + b

    @job
    def mult(a, b):
        return a * b

    @job
    def make_more(val):
        return [val] * 3

    @subflow
    def add_distributed(vals, c):
        return [add(val, c) for val in vals]

    @subflow
    def add_distributed2(vals, c, op):
        return [op(val, c) for val in vals]

    @flow
    def workflow(a, b, c):
        return mult(add(a, b), c)

    @flow
    def dynamic_workflow(a, b, c):
        result1 = add(a, b)
        result2 = make_more(result1)
        return add_distributed(result2, c)

    @flow
    def dynamic_workflow2(a, b, c):
        result1 = add(a, b)
        result2 = make_more(result1)
        return add_distributed2(result2, c, add)

    @flow
    def dynamic_workflow3(a, b, c):
        result1 = add(a, b)
        result2 = make_more(result1)
        result3 = add_distributed(result2, c)
        result4 = add_distributed(result3, c)
        return add(result4[0], c)

    @flow
    def add_flow(a, b):
        return add(a, b)

    assert add_flow(1, 2).result() == 3
    assert workflow(1, 2, 3).result() == 9
    assert [r.result() for r in dynamic_workflow(1, 2, 3)] == [6, 6, 6]
    assert [r.result() for r in dynamic_workflow2(1, 2, 3)] == [6, 6, 6]
    assert dynamic_workflow3(1, 2, 3).result() == 12


def test_prefect_decorators_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with change_settings({"PREFECT_AUTO_SUBMIT": False}):

        @job
        def add(a, b):
            return a + b

        @job
        def mult(a, b):
            return a * b

        @job
        def make_more(val):
            return [val] * 3

        @subflow
        def add_distributed(vals, c):
            return [add(val, c) for val in vals]

        @subflow
        def add_distributed2(vals, c, op):
            return [op(val, c) for val in vals]

        @flow
        def workflow(a, b, c):
            return mult(add(a, b), c)

        @flow
        def dynamic_workflow(a, b, c):
            result1 = add(a, b)
            result2 = make_more(result1)
            return add_distributed(result2, c)

        @flow
        def dynamic_workflow2(a, b, c):
            result1 = add(a, b)
            result2 = make_more(result1)
            return add_distributed2(result2, c, add)

        @flow
        def dynamic_workflow3(a, b, c):
            result1 = add(a, b)
            result2 = make_more(result1)
            result3 = add_distributed(result2, c)
            result4 = add_distributed(result3, c)
            return add(result4[0], c)

        @flow
        def add_flow(a, b):
            return add(a, b)

        assert add_flow(1, 2) == 3
        assert workflow(1, 2, 3) == 9
        results = dynamic_workflow(1, 2, 3)
        assert results == [6, 6, 6]
        results = dynamic_workflow2(1, 2, 3)
        assert results == [6, 6, 6]
        assert dynamic_workflow3(1, 2, 3) == 12


def test_strip_decorators():
    @job
    def add(a, b):
        return a + b

    @flow
    def add2(a, b):
        return a + b

    @subflow
    def add3(a, b):
        return a + b

    stripped_add = strip_decorator(add)
    assert stripped_add(1, 2) == 3

    stripped_add2 = strip_decorator(add2)
    assert stripped_add2(1, 2) == 3

    stripped_add3 = strip_decorator(add3)
    assert stripped_add3(1, 2) == 3


def test_state_patch():
    @job
    def testjob(n):
        return {"result": n + 5}

    @flow
    def subflow(n):
        return testjob(n + 1)

    @flow
    def my_flow():
        # Try running a job
        job_result = testjob(1)  # return State

        # Try using the job as an input to a flow
        flow_result = subflow(job_result["result"])

        # Try using the outputs of a flow as the input to a job
        testjob(flow_result["result"])

        return 2

    # Note: No need for `.result()` here because we are returning data, not a PrefectFuture
    # Doing `my_flow(return_state=True)` would force the need for a `.result()`. In general,
    # Prefect flows return the object as-is unless `return_state=True` is set. It's just
    # that in general throughout quacc, we are often returning an unresolved `PrefectFuture`
    assert my_flow() == 2


def test_change_settings_redecorate_job(tmp_path_factory):
    tmp_dir1 = tmp_path_factory.mktemp("dir1")

    @job
    def write_file_job(name="job.txt"):
        with open(Path(SETTINGS.RESULTS_DIR, name), "w") as f:
            f.write("test file")

    write_file_job = redecorate(
        write_file_job, job(settings_swap={"RESULTS_DIR": tmp_dir1})
    )

    @flow
    def my_flow():
        return write_file_job()

    my_flow().result()
    assert Path(tmp_dir1 / "job.txt").exists()


def test_change_settings_redecorate_flow(tmp_path_factory):
    tmp_dir2 = tmp_path_factory.mktemp("dir2")

    @job
    def write_file_job(name="job.txt"):
        with open(Path(SETTINGS.RESULTS_DIR, name), "w") as f:
            f.write("test file")

    @flow
    def write_file_flow(name="flow.txt", job_decorators=None):
        write_file_job_ = customize_funcs(
            ["write_file_job"], [write_file_job], decorators=job_decorators
        )
        return write_file_job_(name=name)

    # Test with redecorating a job in a flow
    write_file_flow(
        job_decorators={"write_file_job": job(settings_swap={"RESULTS_DIR": tmp_dir2})}
    ).result()
    assert Path(tmp_dir2 / "flow.txt").exists()
