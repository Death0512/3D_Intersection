"""Phase 8 — origin/destination routing helpers."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Route:
    origin: str
    destination: str
    nodes: List[str]
    movements: List[str] = field(default_factory=list)


def shortest_path(graph: Dict[str, Iterable[str]], origin: str, destination: str) -> List[str]:
    if origin == destination:
        return [origin]
    q = deque([(origin, [origin])])
    seen = {origin}
    while q:
        node, path = q.popleft()
        for nxt in graph.get(node, []):
            if nxt in seen:
                continue
            if nxt == destination:
                return path + [nxt]
            seen.add(nxt)
            q.append((nxt, path + [nxt]))
    raise ValueError(f"no route from {origin!r} to {destination!r}")


def route_from_od(graph: Dict[str, Iterable[str]], origin: str, destination: str) -> Route:
    nodes = shortest_path(graph, origin, destination)
    movements = [f"{a}->{b}" for a, b in zip(nodes, nodes[1:])]
    return Route(origin=origin, destination=destination, nodes=nodes, movements=movements)


def single_intersection_route(approach: str, turn: str) -> Route:
    opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
    left = {"N": "W", "W": "S", "S": "E", "E": "N"}
    right = {"N": "E", "E": "S", "S": "W", "W": "N"}
    tables = {"straight": opposite, "left": left, "right": right}
    if turn not in tables:
        raise ValueError(f"unknown turn {turn!r}")
    dest = tables[turn][approach]
    return Route(
        origin=f"{approach}_in",
        destination=f"{dest}_out",
        nodes=[f"{approach}_in", "intersection", f"{dest}_out"],
        movements=[f"{approach}_{turn}"],
    )


__all__ = ["Route", "shortest_path", "route_from_od", "single_intersection_route"]
