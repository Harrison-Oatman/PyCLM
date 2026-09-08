import logging
from collections.abc import Callable
from queue import Empty, Queue
from threading import Event
from time import sleep
from typing import Any, ClassVar, NamedTuple

from .messages import Message

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


class PipelineProcess(BaseProcess):
    """
    A process that receives frame-derived data from the
    :class:`~pyclm.core.router.Router` and may publish more.

    Subclasses declare what they make in ``produces`` (``{kind: (input
    kinds, ...)}``), say what they want in :meth:`subscriptions`, and
    implement :meth:`handle_data`. The router calls :meth:`attach` with the
    process's data inbox; data and the final ``StreamCloseMessage`` arrive on
    it in order. When the stream ends the default :meth:`on_stream_end` ends
    this process's own stream and exits the loop.

    ``continuous`` producers need every acquired frame of their inputs once
    they run for a channel at all (tracking); others run at whatever cadence
    their consumers ask for (segmentation). ``always_active`` processes are
    started even when nothing is routed to them (the pattern process answers
    the Manager; the writer records every frame).
    """

    produces: ClassVar[dict[str, tuple[str, ...]]] = {}
    continuous: ClassVar[bool] = False
    always_active: ClassVar[bool] = False

    def __init__(self, stop_event: Event | None = None, name: str = "process"):
        super().__init__(stop_event, name)
        self.router = None
        self.data_inbox: Queue | None = None
        self.stream_ended = False
        self._own_stream_ended = False

    # ------------------------------------------------------------ routing
    def subscriptions(self, plan) -> list:
        """What this process wants, as :class:`~pyclm.core.router.Subscription`s."""
        return []

    def can_produce(self, kind: str, experiment, channel: str) -> bool:
        """Whether this process can make ``kind`` for ``channel`` of ``experiment``."""
        return kind in self.produces

    def attach(self, router, data_inbox: Queue) -> None:
        self.router = router
        self.data_inbox = data_inbox
        self.register_queue(data_inbox, self._handle_inbox)

    def publish(self, data) -> int:
        if self.router is None:
            raise RuntimeError(f"{self.name} is not attached to a router")
        return self.router.publish(data)

    def end_stream(self) -> None:
        """Tell the router this process will publish nothing more (idempotent)."""
        if self._own_stream_ended or self.router is None:
            return
        self._own_stream_ended = True
        self.router.end_stream(self.name)

    # ----------------------------------------------------------- handling
    def _handle_inbox(self, item) -> bool:
        if isinstance(item, Message):
            if item.message == "stream_close":
                self.stream_ended = True
                logger.info(f"{self.name}: data stream ended")
                return bool(self.on_stream_end())
            raise ValueError(f"{self.name}: unexpected message on data inbox: {item}")
        self.handle_data(item)
        return False

    def handle_data(self, data) -> None:
        raise NotImplementedError

    def on_stream_end(self) -> bool:
        """Called once after the last data item. Return True to exit the loop."""
        self.end_stream()
        return True

    def process(self):
        try:
            super().process()
        finally:
            self.end_stream()
