import logging
import threading
import time
from queue import Queue

from pyclm.core.base_process import BaseProcess


class MockProcess(BaseProcess):
    def __init__(self, stop_event=None):
        super().__init__(stop_event, name="mock")
        self.inbox = Queue()
        self.register_queue(self.inbox, self.handle_message)
        self.processed_count = 0

    def handle_message(self, msg):
        if msg == "stop":
            return True
        if msg == "boom":
            raise RuntimeError("handler failure")
        self.processed_count += 1
        return False


def test_process_loop():
    stop_event = threading.Event()
    process = MockProcess(stop_event)

    # Run process in a separate thread because it blocks
    t = threading.Thread(target=process.process)
    t.start()

    try:
        # Send messages
        process.inbox.put("msg1")
        process.inbox.put("msg2")

        # Allow time to process
        time.sleep(0.1)

        assert process.processed_count == 2

        # Test stop signal via message
        process.inbox.put("stop")
        t.join(timeout=1.0)
        assert not t.is_alive()

    finally:
        if t.is_alive():
            stop_event.set()
            t.join()


def test_handler_errors_are_counted_and_loop_continues(caplog):
    process = MockProcess(threading.Event())
    process.inbox.put("msg1")
    process.inbox.put("boom")
    process.inbox.put("msg2")
    process.inbox.put("stop")

    with caplog.at_level(logging.ERROR):
        process.process()

    assert process.processed_count == 2
    assert process.error_count == 1
    assert "handler failure" in caplog.text


def test_stop_event_ends_loop():
    stop_event = threading.Event()
    process = MockProcess(stop_event)
    t = threading.Thread(target=process.process)
    t.start()
    stop_event.set()
    t.join(timeout=1.0)
    assert not t.is_alive()
