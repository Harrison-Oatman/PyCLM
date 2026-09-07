import logging
from collections.abc import Callable
from queue import Empty, Queue
from threading import Event
from time import sleep
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class QueueHandler(NamedTuple):
    queue: Queue
    handler: Callable[[Any], bool | None]


class BaseProcess:
    """
    Base class for processes that poll queues.
    Eliminates busy waiting by sleeping when all queues are empty.

    An exception raised by a handler is logged with its traceback and counted
    in ``error_count``; the process keeps running.
    """

    def __init__(self, stop_event: Event | None = None, name: str = "process"):
        self.stop_event = stop_event
        self.name = name
        self.queues: list[QueueHandler] = []
        self.sleep_interval = 0.001
        self.error_count = 0

    def register_queue(self, queue: Queue, handler: Callable[[Any], bool | None]):
        """
        Register a queue to be polled.
        :param queue: The queue to poll.
        :param handler: A callable that takes the item from the queue.
                        It can optionally return True to signal the process loop to break (stop).
        """
        self.queues.append(QueueHandler(queue, handler))

    def process(self):
        """
        Main process loop.
        Polls all registered queues. SLEEPS if no work was done in a cycle.
        """
        logger.info(f"Started {self.name}")

        while True:
            if self.stop_event and self.stop_event.is_set():
                logger.info(f"Force closing {self.name}")
                break

            did_work = False

            for q_handler in self.queues:
                try:
                    item = q_handler.queue.get_nowait()
                except Empty:
                    continue

                did_work = True

                try:
                    should_stop = q_handler.handler(item)
                except Exception:
                    self.error_count += 1
                    logger.error(
                        f"Error handling item in {self.name} "
                        f"({self.error_count} total)",
                        exc_info=True,
                    )
                    continue

                if should_stop:
                    logger.info(f"{self.name} received stop signal from handler")
                    return

            # If no queues had items, sleep briefly to avoid 100% CPU
            if not did_work:
                sleep(self.sleep_interval)

        logger.info(f"Stopped {self.name}")
