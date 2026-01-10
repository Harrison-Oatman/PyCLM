import logging
from collections.abc import Callable
from multiprocessing import Queue
from queue import Empty
from threading import Event
from time import sleep
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class QueueHandler(NamedTuple):
    queue: Queue
    handler: Callable[[Any], None]


class BaseProcess:
    """
    Base class for processes that poll queues.
    Eliminates busy waiting by sleeping when all queues are empty.
    """

    def __init__(self, stop_event: Event | None = None, name: str = "process"):
        self.stop_event = stop_event
        self.name = name
        self.queues: list[QueueHandler] = []
        self.sleep_interval = 0.001

    def register_queue(self, queue: Queue, handler: Callable[[Any], bool | None]):
        """
        Register a queue to be polled.
        :param queue: The multiprocessing Queue to poll.
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
                queue = q_handler.queue
                handler = q_handler.handler

                if not queue.empty():
                    try:
                        item = queue.get_nowait()

                        should_stop = handler(item)
                        if should_stop:
                            logger.info(
                                f"{self.name} received stop signal from handler"
                            )
                            return

                        did_work = True
                    except Empty:
                        # Race condition handling: empty() said False but get_nowait() raised Empty
                        pass
                    except Exception as e:
                        logger.error(
                            f"Error handling item in {self.name}: {e}", exc_info=True
                        )

            # If no queues had items, sleep briefly to avoid 100% CPU
            if not did_work:
                sleep(self.sleep_interval)

        logger.info(f"Stopped {self.name}")
