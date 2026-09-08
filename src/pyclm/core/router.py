"""
Routing of frame-derived data between pipeline processes.

Every piece of data the pipeline moves (a raw frame, a segmentation, a
track table) has a *kind* (:data:`KINDS`) and an identity from the plan
(``data.event.index``: experiment ``p``, channel ``c``, timepoint ``t``).
The :class:`Router` holds a table ``(experiment, channel, kind) →
[consumer, ...]`` built once by :meth:`Router.resolve` from what the
registered processes want (``subscriptions(plan)``) and can make
(``produces``), and fans data out with :meth:`Router.publish`. It also
derives the shutdown fan-in: a consumer receives one
:class:`~pyclm.core.messages.StreamCloseMessage` on its data inbox when
every producer feeding it has ended its stream (:meth:`Router.end_stream`).

The router is not a thread. ``publish`` runs on the producer's thread and
costs one ``queue.put`` per delivery; queues pass references, so consumers
share the object and must not mutate it.

A registered process needs ``name`` and ``attach(router, inbox)``, and may
have ``produces`` (``{kind: (input kinds, ...)}``), ``subscriptions(plan)``,
``can_produce(kind, experiment, channel)``, ``continuous`` (a producer that
needs every frame of its inputs once it runs at all, like tracking) and
``always_active``; see :class:`~pyclm.core.base_process.PipelineProcess`.

See docs/stage3-router-design.md.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING, Any

from .kinds import base_kind
from .messages import StreamCloseMessage

if TYPE_CHECKING:
    from .plan import AcquisitionPlan

logger = logging.getLogger(__name__)

# base kinds; a segmentation from a named [segmentation.<name>] table travels
# as "seg:<name>" and is produced by the "seg" producer (see core/kinds.py)
KINDS = ("raw", "seg", "tracks")
CADENCES = ("always", "pattern")


class RoutingError(ValueError):
    """The registered processes cannot satisfy what the plan and methods ask for."""


@dataclass(frozen=True)
class Subscription:
    """
    One process's interest in one kind of data for one experiment channel.

    ``cadence`` is ``"always"`` (every acquired frame) or ``"pattern"`` (only
    at timepoints where the experiment's pattern is due, i.e.
    ``plan.pattern_due``). ``demand=False`` marks a subscription that records
    what is produced anyway without causing a stage to run (the writer's).
    """

    consumer: str
    experiment: str
    channel: str
    kind: str
    cadence: str = "always"
    demand: bool = True

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.experiment, self.channel, self.kind)


@dataclass
class _Delivery:
    consumer: str
    cadence: str
    demand: bool

    def merge(self, cadence: str, demand: bool) -> bool:
        """Widen this delivery. Returns True if the cadence changed."""
        changed = cadence == "always" and self.cadence != "always"
        if changed:
            self.cadence = "always"
        self.demand = self.demand or demand
        return changed

    def label(self) -> str:
        out = self.consumer
        if self.cadence != "always":
            out += f"@{self.cadence}"
        if not self.demand:
            out += "(record)"
        return out


def _subscriptions_of(proc: Any, plan: AcquisitionPlan) -> list[Subscription]:
    fn = getattr(proc, "subscriptions", None)
    return list(fn(plan)) if fn is not None else []


class Router:
    """See the module docstring. Build with :meth:`add` then :meth:`resolve`."""

    def __init__(self, plan: AcquisitionPlan):
        self.plan = plan
        self._procs: dict[str, Any] = {}
        self._producers: dict[str, str] = {}
        self._table: dict[tuple[str, str, str], list[_Delivery]] = {}
        self._inboxes: dict[str, Queue] = {}
        self._upstreams: dict[str, set[str]] = {}
        self._ended: dict[str, set[str]] = {}
        self._active: list[str] = []
        self._lock = threading.Lock()
        self._warned: set[tuple] = set()
        self.resolved = False
        # frames published for a key nobody subscribes to (a health counter)
        self.undeliverable = 0

    # ------------------------------------------------------------ building
    def add(self, proc: Any) -> None:
        """Register a process (before :meth:`resolve`)."""
        if self.resolved:
            raise RoutingError("cannot add a process after the router is resolved")
        name = proc.name
        if name in self._procs:
            raise RoutingError(f"a process named {name!r} is already registered")
        self._procs[name] = proc
        for kind in getattr(proc, "produces", {}):
            if kind not in KINDS:
                raise RoutingError(f"{name!r} produces unknown kind {kind!r}")
            other = self._producers.get(kind)
            if other is not None:
                raise RoutingError(f"{name!r} and {other!r} both produce {kind!r}")
            self._producers[kind] = name

    def resolve(self) -> Router:
        """
        Build the table, derive producer-side subscriptions, validate, attach
        inboxes. Rules (docs/stage3-router-design.md §3.4):

        1. every ``demand`` subscription pulls in the producer of its kind,
           subscribing that producer to its input kinds at the widest cadence
           any downstream consumer asks for (``"always"`` if the producer is
           ``continuous``);
        2. ``demand=False`` subscriptions are honoured only for data that is
           produced anyway;
        3. a demanded kind with no producer, or one whose producer cannot
           serve that experiment, is a :class:`RoutingError`;
        4. a consumer's stream ends after the raw producer's and after each
           producer feeding it; processes receiving nothing are inactive
           unless ``always_active``.
        """
        if self.resolved:
            raise RoutingError("router already resolved")
        if "raw" not in self._producers:
            raise RoutingError("no registered process produces raw frames")

        explicit: list[Subscription] = []
        for proc in self._procs.values():
            for sub in _subscriptions_of(proc, self.plan):
                self._check(sub)
                explicit.append(sub)

        table: dict[tuple, dict[str, _Delivery]] = defaultdict(dict)

        pending = [s for s in explicit if s.demand]
        for sub in pending:
            self._put(table, sub)
        while pending:
            sub = pending.pop()
            if sub.kind == "raw":
                continue
            producer = self._producer_for(sub)
            proc = self._procs[producer]
            cadence = "always" if getattr(proc, "continuous", False) else sub.cadence
            for kind_in in self._inputs(proc, sub):
                derived = Subscription(
                    producer, sub.experiment, sub.channel, kind_in, cadence, True
                )
                if self._put(table, derived):
                    pending.append(derived)

        for sub in explicit:
            if sub.demand:
                continue
            if sub.kind != "raw" and not self._is_produced(table, sub):
                continue
            self._put(table, sub)

        self._table = {key: list(d.values()) for key, d in table.items()}

        root = self._producers["raw"]
        upstreams: dict[str, set[str]] = defaultdict(set)
        for (_exp, _ch, kind), deliveries in self._table.items():
            for d in deliveries:
                upstreams[d.consumer].add(self._producers[base_kind(kind)])

        for name, proc in self._procs.items():
            active = name in upstreams or getattr(proc, "always_active", False)
            if not active:
                continue
            self._active.append(name)
            if name == root:
                self._upstreams[name] = set()
            else:
                self._upstreams[name] = set(upstreams.get(name, ())) | {root}
            self._ended[name] = set()
            inbox: Queue = Queue()
            self._inboxes[name] = inbox
            proc.attach(self, inbox)

        self.resolved = True
        logger.info(f"routing resolved: {self.as_dict()}")
        return self

    @staticmethod
    def _put(table, sub: Subscription) -> bool:
        """Add or widen a delivery. Returns True if the producer must be (re)visited."""
        delivery = table[sub.key].get(sub.consumer)
        if delivery is None:
            table[sub.key][sub.consumer] = _Delivery(
                sub.consumer, sub.cadence, sub.demand
            )
            return True
        return delivery.merge(sub.cadence, sub.demand)

    def _inputs(self, proc: Any, sub: Subscription) -> tuple[str, ...]:
        """The kinds ``proc`` consumes to produce ``sub.kind`` for that experiment channel."""
        fn = getattr(proc, "inputs", None)
        if fn is not None:
            experiment = self.plan.schedule.experiments[sub.experiment]
            return tuple(fn(sub.kind, experiment, sub.channel))
        return tuple(proc.produces[base_kind(sub.kind)])

    def _is_produced(self, table, sub: Subscription) -> bool:
        producer = self._producers.get(base_kind(sub.kind))
        if producer is None:
            return False
        inputs = self._inputs(self._procs[producer], sub)
        return all(
            producer in table.get((sub.experiment, sub.channel, kind_in), {})
            for kind_in in inputs
        )

    def _producer_for(self, sub: Subscription) -> str:
        where = f"{sub.kind!r} of {sub.experiment}/{sub.channel}"
        name = self._producers.get(base_kind(sub.kind))
        if name is None:
            raise RoutingError(
                f"{sub.consumer!r} needs {where} but no registered process "
                f"produces {sub.kind!r}"
            )
        can = getattr(self._procs[name], "can_produce", None)
        experiment = self.plan.schedule.experiments[sub.experiment]
        if can is not None and not can(sub.kind, experiment, sub.channel):
            raise RoutingError(
                f"{sub.consumer!r} needs {where} but {name!r} has no method "
                f"configured for experiment {sub.experiment!r} "
                "(check the experiment TOML)"
            )
        return name

    def _check(self, sub: Subscription) -> None:
        plan = self.plan
        if base_kind(sub.kind) not in KINDS:
            raise RoutingError(f"{sub.consumer!r}: unknown data kind {sub.kind!r}")
        if sub.cadence not in CADENCES:
            raise RoutingError(f"{sub.consumer!r}: unknown cadence {sub.cadence!r}")
        if sub.experiment not in plan.experiments:
            raise RoutingError(
                f"{sub.consumer!r}: unknown experiment {sub.experiment!r}"
            )
        if sub.channel not in plan.channels(sub.experiment):
            raise RoutingError(
                f"{sub.consumer!r}: experiment {sub.experiment!r} has no channel "
                f"{sub.channel!r} (channels: {plan.channels(sub.experiment)})"
            )
        if sub.cadence == "pattern":
            for t in range(plan.timepoints):
                if plan.pattern_due(sub.experiment, t) and not plan.is_scheduled(
                    sub.experiment, sub.channel, t
                ):
                    raise RoutingError(
                        f"{sub.consumer!r} wants {sub.kind!r} of "
                        f"{sub.experiment}/{sub.channel} when the pattern is due, "
                        f"but at t={t} the pattern is due and the channel is not "
                        "acquired"
                    )

    # ------------------------------------------------------------- running
    def publish(self, data: Any) -> int:
        """Deliver ``data`` to every subscriber due at its timepoint. Returns the count."""
        if not self.resolved:
            raise RoutingError("publish() before resolve()")
        event = data.event
        key = (event.experiment_name, event.index.get("c"), data.kind)
        deliveries = self._table.get(key)
        if not deliveries:
            self.undeliverable += 1
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(f"no subscriber for {key}; dropping it")
            return 0

        due = None
        n = 0
        for d in deliveries:
            if d.cadence == "pattern":
                if due is None:
                    due = self.plan.pattern_due(key[0], event.t_index)
                if not due:
                    continue
            self._inboxes[d.consumer].put(data)
            n += 1
        return n

    def end_stream(self, producer: str) -> list[str]:
        """
        Record that ``producer`` will publish nothing more. Every consumer
        whose last upstream this was gets one StreamCloseMessage. Returns the
        consumers closed. Idempotent per producer.
        """
        closed = []
        with self._lock:
            for consumer, ups in self._upstreams.items():
                if producer not in ups:
                    continue
                ended = self._ended[consumer]
                if producer in ended:
                    continue
                ended.add(producer)
                if ended == ups:
                    closed.append(consumer)
        for consumer in closed:
            self._inboxes[consumer].put(StreamCloseMessage())
        if closed:
            logger.info(f"{producer} ended its stream; closing {closed}")
        return closed

    # ------------------------------------------------------------- queries
    def is_active(self, name: str) -> bool:
        return name in self._active

    def active_processes(self) -> list[Any]:
        """Registered processes that will receive data (or are always active), in registration order."""
        return [self._procs[name] for name in self._active]

    def inbox(self, name: str) -> Queue:
        return self._inboxes[name]

    def upstreams(self, name: str) -> set[str]:
        return set(self._upstreams.get(name, ()))

    def demanded(self, kind: str) -> set[tuple[str, str]]:
        """``(experiment, channel)`` pairs for which some consumer demands ``kind``."""
        return {
            (exp, ch)
            for (exp, ch, k), deliveries in self._table.items()
            if k == kind and any(d.demand for d in deliveries)
        }

    def demanded_kinds(self, base: str) -> set[tuple[str, str, str]]:
        """``(experiment, channel, kind)`` demanded for every kind whose base is ``base`` (e.g. ``seg``, ``seg:nuclei``)."""
        return {
            (exp, ch, k)
            for (exp, ch, k), deliveries in self._table.items()
            if base_kind(k) == base and any(d.demand for d in deliveries)
        }

    def produced_by(
        self, producer: str
    ) -> dict[tuple[str, str], list[tuple[str, str]]]:
        """
        What ``producer`` must make: ``{(experiment, channel): [(kind, cadence), ...]}``
        where ``cadence`` is ``"always"`` if any *demanding* subscriber wants
        every frame (a record-only subscriber never widens production).
        """
        out: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for (exp, ch, kind), deliveries in sorted(self._table.items()):
            if self._producers.get(base_kind(kind)) != producer:
                continue
            cadence = (
                "always"
                if any(d.cadence == "always" and d.demand for d in deliveries)
                else "pattern"
            )
            out[(exp, ch)].append((kind, cadence))
        return dict(out)

    def deliveries_to(self, consumer: str) -> dict[str, set[tuple[str, str]]]:
        """``{kind: {(experiment, channel), ...}}`` that ``consumer`` will receive."""
        out: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for (exp, ch, kind), deliveries in self._table.items():
            if any(d.consumer == consumer for d in deliveries):
                out[kind].add((exp, ch))
        return dict(out)

    def receives(self, consumer: str, experiment: str, channel: str, kind: str) -> bool:
        return any(
            d.consumer == consumer
            for d in self._table.get((experiment, channel, kind), ())
        )

    def as_dict(self) -> dict:
        """Provenance: producers, the routes per experiment/channel/kind, fan-in, active processes."""
        routes: dict = {}
        for (exp, ch, kind), deliveries in sorted(self._table.items()):
            routes.setdefault(exp, {}).setdefault(ch, {})[kind] = [
                d.label() for d in deliveries
            ]
        return {
            "producers": dict(self._producers),
            "routes": routes,
            "upstreams": {k: sorted(v) for k, v in self._upstreams.items()},
            "active": list(self._active),
        }
