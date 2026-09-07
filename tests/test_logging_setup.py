"""Per-run logging: each set_logging call must redirect the log file."""

import logging

from pyclm.run_pyclm import _PYCLM_HANDLER_FLAG, remove_pyclm_log_handlers, set_logging


def test_set_logging_switches_file_per_run(tmp_path):
    first = tmp_path / "run1"
    second = tmp_path / "run2"
    first.mkdir()
    second.mkdir()
    test_logger = logging.getLogger("pyclm.tests.logging")

    try:
        set_logging(first)
        test_logger.info("first-run-message")

        set_logging(second)
        test_logger.info("second-run-message")
    finally:
        remove_pyclm_log_handlers()

    log1 = (first / "log.log").read_text()
    log2 = (second / "log.log").read_text()

    assert "first-run-message" in log1
    assert "second-run-message" not in log1
    assert "second-run-message" in log2
    assert "first-run-message" not in log2

    root = logging.getLogger()
    assert not any(getattr(h, _PYCLM_HANDLER_FLAG, False) for h in root.handlers)
