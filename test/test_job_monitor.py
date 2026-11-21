"""
Completion detection in JobManager._monitor_job_execution: a job is complete only when
every rank explicitly reports JOB_COMPLETED; query errors and unknown states never count.
"""

from types import SimpleNamespace

import pytest

from optimus.controller.job_manager import (
    MAX_STATUS_MISSES,
    JobManager,
    JobRecord,
    NonRetryableJobError,
)
from optimus.protocol.generated.messages_pb2 import JobState, JobStatusResponse


class _Client:
    def __init__(self, script):
        self.script = list(script)

    def get_job_status(self, job_id, timeout=2.0):
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        status, msg = item if isinstance(item, tuple) else (item, "")
        return JobStatusResponse(job_id=job_id, status=status, error_message=msg)


def _manager(tmp_path, scripts):
    clients = {host: _Client(s) for host, s in scripts.items()}
    jm = JobManager(
        scheduler=None,
        launcher=None,
        probe_manager=SimpleNamespace(probe_all=lambda: None, get_all_states=lambda: []),
        client_manager=SimpleNamespace(get_client=lambda host, port: clients[host]),
        runs_dir=str(tmp_path / "runs"),
    )
    nodes = [SimpleNamespace(hostname=h, port=1, rank=i) for i, h in enumerate(scripts)]
    record = JobRecord(job_id="j", dataset_id="d", dataset_version="v", world_size=len(nodes),
                       status=JobState.JOB_RUNNING)
    run = lambda: jm._monitor_job_execution(  # noqa: E731
        record, SimpleNamespace(selected_nodes=nodes), None, poll_interval=0
    )
    return run, record


def test_query_errors_do_not_mark_job_completed(tmp_path):
    # Every poll fails on every node: previously this returned True (COMPLETED).
    run, record = _manager(tmp_path, {"a": [TimeoutError("blip")], "b": [TimeoutError("blip")]})
    assert run() is False
    assert "consecutive status checks failed" in record.error_message


def test_transient_blip_then_completion(tmp_path):
    R, C = JobState.JOB_RUNNING, JobState.JOB_COMPLETED
    run, _ = _manager(
        tmp_path,
        {"a": [TimeoutError("blip"), R, C], "b": [TimeoutError("blip"), C, C]},
    )
    assert run() is True


def test_unknown_job_counts_as_miss_not_hang(tmp_path):
    run, _ = _manager(tmp_path, {"a": [JobState.JOB_UNKNOWN], "b": [JobState.JOB_COMPLETED]})
    assert run() is False


def test_failure_is_retryable_by_default(tmp_path):
    run, record = _manager(tmp_path, {"a": [(JobState.JOB_FAILED, "Gloo timeout")]})
    assert run() is False
    assert "Gloo timeout" in record.error_message


def test_dataset_missing_is_non_retryable(tmp_path):
    msg = "Process exited with code 1\nDatasetUnavailableError: Dataset 'x' not found"
    run, _ = _manager(tmp_path, {"a": [(JobState.JOB_FAILED, msg)]})
    with pytest.raises(NonRetryableJobError, match="DatasetUnavailableError"):
        run()


def test_miss_limit_is_sane():
    assert MAX_STATUS_MISSES >= 3


def test_executor_log_tail_includes_traceback(tmp_path):
    from optimus.worker.executor import _log_tail

    log = tmp_path / "j_rank0.log"
    log.write_text("noise\n" * 100 + "Traceback (most recent call last):\nValueError: boom\n")
    tail = _log_tail(str(log))
    assert "ValueError: boom" in tail and "j_rank0.log" in tail
    assert tail.count("noise") < 30
    assert _log_tail(str(tmp_path / "missing.log")) == ""
