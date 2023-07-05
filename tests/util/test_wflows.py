import pytest

try:
    from quacc.util.wflows import launch_runner, make_runner

    dask_prefect = True
except ImportError:
    dask_prefect = None


@pytest.mark.skipif(
    dask_prefect is None,
    reason="Dask and Prefect dependencies must be installed.",
)
def test_make_prefect_runner(capsys):
    # NOTE: I don't know how to either turn off the stdout/stderr
    # from dask-jobqueue or mock it.
    cluster_kwargs = {
        "cores": 1,
        "memory": "1GB",
        "processes": 1,
        "account": "MyAccount",
    }
    make_runner(cluster_kwargs)


@pytest.mark.skipif(
    dask_prefect is None,
    reason="Dask and Prefect dependencies must be installed.",
)
def test_launch_prefect_runner():
    # NOTE: I don't know how to either turn off the stdout/stderr
    # from dask-jobqueue or mock it.
    cluster_kwargs = {
        "cores": 1,
        "memory": "1GB",
        "processes": 1,
        "account": "MyAccount",
    }
    launch_runner(cluster_kwargs, verbose=True)
