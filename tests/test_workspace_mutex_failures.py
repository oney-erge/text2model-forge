"""A failed workspace lock operation must not strand later Studio requests."""

import threading
from pathlib import Path

import pytest

from text2model_forge.studio_store import StudioStore, _WorkspaceMutex


@pytest.mark.parametrize("failure", ["open", "write", "lock", "unlock"])
def test_workspace_lock_failure_does_not_block_other_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    store = StudioStore(tmp_path)
    mutex = store._lock
    real_open = Path.open
    opened = []

    class FailingWrite:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def write(self, data):
            raise OSError("synthetic lock write failure")

    def fail_open(path, *args, **kwargs):
        if path == mutex.lock_path:
            if failure == "open":
                raise OSError("synthetic lock open failure")
            handle = real_open(path, *args, **kwargs)
            opened.append(handle)
            return FailingWrite(handle)
        return real_open(path, *args, **kwargs)

    def fail_lock_operation(_):
        raise OSError("synthetic lock operation failure")

    with monkeypatch.context() as patch:
        if failure in {"open", "write"}:
            patch.setattr(Path, "open", fail_open)
        elif failure == "lock":
            patch.setattr(_WorkspaceMutex, "_lock_file", staticmethod(fail_lock_operation))
        else:
            patch.setattr(_WorkspaceMutex, "_unlock_file", staticmethod(fail_lock_operation))
        with pytest.raises(OSError, match="synthetic"), mutex:
            pass

    assert all(handle.closed for handle in opened)
    acquired = threading.Event()

    def acquire_from_another_thread():
        with mutex:
            acquired.set()

    thread = threading.Thread(target=acquire_from_another_thread, daemon=True)
    thread.start()
    assert acquired.wait(1), "a later request is stuck after the lock failure"
    thread.join(timeout=1)
