"""Phase 4: intersection as a shared conflict-resource model.

The intersection box is modeled as a collection of conflict zones.  Each
movement (approach + turn) reserves a set of zones for the duration of its box
traversal.  A vehicle may only enter when:

    1. its signal phase is green (checked by the engine, not here);
    2. the zones its movement needs are free of conflicting reservations;
    3. the downstream exit lane has enough space to accept the vehicle
       (``downstream blocking``).

Zone assignment per movement:

    straight moves share only the central zone with opposing left turns
    that cross them; compatible movements (permissive same-direction, or
    protected non-crossing movements) coexist.  Conflict rules and phase
    helpers are drawn from the canonical ``traffic_signal`` module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from traffic_signal import _ALL_COMBOS, _movement_to_phase


def _phase_for(approach, turn) -> int:
    """Return NEMA phase for (approach, turn) — delegates to the canonical
    ``traffic_signal._movement_to_phase``."""
    return _movement_to_phase(approach, turn)


def movements_conflict(mv_a: Tuple, mv_b: Tuple) -> bool:
    """Two movements conflict if they cross and are not a compatible combo.

    Same movement, same approach, or opposing approaches never conflict
    (permissive opposing flows); two cross-street movements conflict unless
    they belong to the same protected NEMA combo.
    """
    if mv_a == mv_b:
        return False
    ap_a, _ = mv_a
    ap_b, _ = mv_b
    if ap_a == ap_b:
        return False
    opp = {"N": "S", "S": "N", "E": "W", "W": "E"}
    a = ap_a.value if hasattr(ap_a, "value") else ap_a
    b = ap_b.value if hasattr(ap_b, "value") else ap_b
    if opp.get(a) == b:
        return False
    ph_a = _phase_for(*mv_a)
    ph_b = _phase_for(*mv_b)
    for p1, p2 in _ALL_COMBOS:
        if {ph_a, ph_b} <= {p1, p2}:
            return False
    return True


@dataclass
class Reservation:
    """A box-traversal reservation held by one vehicle."""
    vehicle_id: str
    approach: object
    turn: object
    entry_frame: int
    clear_frame: int
    zones: Set[str] = field(default_factory=set)

    @property
    def movement(self) -> Tuple:
        return (self.approach, self.turn)


def _zones_for_movement(approach, turn) -> Set[str]:
    """Zones occupied by a movement.

    All crossings reserve an approach-specific box zone and the central box
    zone.  Compatible movements may share a zone; the conflict predicate
    decides permissibility, not the zone set.
    """
    a = approach.value if hasattr(approach, "value") else approach
    return {f"{a}_box", "center"}


class IntersectionModel:
    """Shared conflict-resource manager for the intersection box.

    Tracks active reservations (with their zones), exposes a
    ``can_enter(movement, entry_frame, clear_frame, downstream_space)``
    gate, and expires reservations after vehicles clear the box.
    """

    def __init__(self) -> None:
        self.reservations: List[Reservation] = []

    def active_zones(self, tick: int) -> Set[str]:
        occ: Set[str] = set()
        for r in self.reservations:
            if r.entry_frame <= tick < r.clear_frame:
                occ |= r.zones
        return occ

    def reservations_for(self, tick: int) -> List[Reservation]:
        return [r for r in self.reservations
                if r.entry_frame <= tick < r.clear_frame]

    def conflicts(self, movement: Tuple, entry_frame: int,
                  clear_frame: int) -> bool:
        """True if a conflicting reservation overlaps the requested window."""
        for r in self.reservations:
            if r.clear_frame <= entry_frame or r.entry_frame >= clear_frame:
                continue
            if movements_conflict(movement, r.movement):
                return True
        return False

    def can_enter(self, movement: Tuple, entry_frame: int,
                  clear_frame: int, vehicle_length: float,
                  downstream_space: Optional[float] = None) -> bool:
        """Gate combining conflict safety and downstream-space availability.

        ``downstream_space`` is the free space (m) on the receiving exit
        lane at ``entry_frame``.  ``None`` disables downstream blocking
        (preserves the original behavior).
        """
        if self.conflicts(movement, entry_frame, clear_frame):
            return False
        if downstream_space is not None and downstream_space < vehicle_length:
            return False
        return True

    def reserve(self, vehicle_id: str, approach, turn: object,
                entry_frame: int, clear_frame: int) -> Reservation:
        r = Reservation(
            vehicle_id=vehicle_id, approach=approach, turn=turn,
            entry_frame=entry_frame, clear_frame=clear_frame,
            zones=_zones_for_movement(approach, turn),
        )
        self.reservations.append(r)
        return r

    def expire(self, tick: int) -> None:
        """Drop reservations that have cleared the box by ``tick``."""
        self.reservations[:] = [r for r in self.reservations
                                if r.clear_frame > tick]

    def occupancy_count(self, tick: int) -> int:
        return len(self.reservations_for(tick))
