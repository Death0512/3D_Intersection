"""IDM (Intelligent Driver Model) car-following module.

Implements the standard Treiber–Hennecke–Helbing IDM for longitudinal vehicle
dynamics. Red lights and stop lines are modelled as virtual stopped leaders so
that a *single* acceleration function handles both car-following and signal
compliance.

Reference:
  Treiber, M., Hennecke, A. & Helbing, D. (2000).
  "Congested traffic states in empirical observations and microscopic
  simulations." Physical Review E, 62(2), 1805.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Default IDM calibration (typical urban passenger car)
# ---------------------------------------------------------------------------
DEFAULT_DESIRED_SPEED = 15.0   # m/s  (~54 km/h)
DEFAULT_MAX_ACCEL     = 2.5    # m/s²
DEFAULT_COMFORT_DECEL = 2.0    # m/s²  (comfortable braking)
DEFAULT_MIN_GAP       = 2.0    # m     (jam-distance s0)
DEFAULT_TIME_HEADWAY  = 1.5    # s     (desired time headway T)
DEFAULT_DELTA         = 4      # acceleration exponent δ


@dataclass
class IDMParams:
    """Calibration parameters for one vehicle's IDM instance."""
    desired_speed: float      = DEFAULT_DESIRED_SPEED
    max_accel: float          = DEFAULT_MAX_ACCEL
    comfortable_decel: float  = DEFAULT_COMFORT_DECEL
    min_gap: float            = DEFAULT_MIN_GAP
    time_headway: float       = DEFAULT_TIME_HEADWAY
    delta: int                = DEFAULT_DELTA

    def __post_init__(self):
        if self.desired_speed <= 0:
            raise ValueError("desired_speed must be > 0")
        if self.max_accel <= 0:
            raise ValueError("max_accel must be > 0")
        if self.comfortable_decel <= 0:
            raise ValueError("comfortable_decel must be > 0")
        if self.min_gap < 0:
            raise ValueError("min_gap must be >= 0")
        if self.time_headway < 0:
            raise ValueError("time_headway must be >= 0")


# ---------------------------------------------------------------------------
# Core IDM acceleration
# ---------------------------------------------------------------------------

def idm_acceleration(speed: float,
                     desired_speed: float,
                     gap: float,
                     delta_v: float,
                     params: IDMParams) -> float:
    """Compute IDM longitudinal acceleration.

    Args:
        speed:         current vehicle speed v  (m/s, ≥ 0)
        desired_speed: free-flow desired speed v₀  (m/s, > 0)
        gap:           net gap s to the leader's rear bumper  (m)
        delta_v:       speed difference v − v_leader  (m/s, positive = closing)
        params:        IDM calibration parameters

    Returns:
        acceleration a  (m/s²; negative = deceleration)

    The IDM acceleration formula:

        a = a_max · [ 1 − (v / v₀)^δ − (s*(v, Δv) / s)² ]

    where the desired gap s* is:

        s* = s₀ + v·T + v·Δv / (2·√(a_max · b))

    On a free road (gap → ∞) the second fraction vanishes and the vehicle
    simply accelerates toward v₀.  When the gap is small or the vehicle is
    closing on the leader, s* dominates and braking occurs naturally.
    """
    a     = params.max_accel
    b     = params.comfortable_decel
    s0    = params.min_gap
    T     = params.time_headway
    delta = params.delta
    v0    = desired_speed

    # Desired dynamical gap
    interaction = max(0.0, speed * delta_v) / (2.0 * math.sqrt(a * b))
    s_star = s0 + speed * T + interaction

    # Free-road term  (v / v0)^δ
    if v0 > 0:
        free_road = (speed / v0) ** delta
    else:
        free_road = 1.0

    # Gap term  (s* / s)²
    if gap > 0:
        gap_term = (s_star / gap) ** 2
    else:
        # Touching or overlapping: maximum braking
        gap_term = 1e6

    accel = a * (1.0 - free_road - gap_term)

    # Physical clamp: don't exceed comfortable-decel by more than a safety
    # factor (allows emergency braking at ~2× comfortable decel).
    max_decel = -2.0 * b
    return max(accel, max_decel)


def free_road_acceleration(speed: float, params: IDMParams) -> float:
    """IDM acceleration on an empty road (no leader)."""
    return idm_acceleration(
        speed=speed,
        desired_speed=params.desired_speed,
        gap=1e6,       # effectively infinite gap
        delta_v=0.0,
        params=params,
    )


# ---------------------------------------------------------------------------
# Virtual leader for traffic signal
# ---------------------------------------------------------------------------

def virtual_signal_leader(vehicle_s: float,
                          stop_line_s: float,
                          vehicle_speed: float,
                          is_green: bool) -> Optional[Tuple[float, float]]:
    """Return (leader_position, leader_speed) of a virtual stopped vehicle
    at the stop line when the signal is red.

    Returns None when the signal is green (no virtual obstacle).

    The virtual leader sits at ``stop_line_s`` with speed 0.  The calling
    code should compute the gap as ``stop_line_s − vehicle_s − vehicle_length``
    and feed it into ``idm_acceleration`` alongside the physical leader,
    using whichever produces the lower (more constraining) gap.
    """
    if is_green:
        return None
    # Only constrain vehicles that haven't yet crossed the stop line.
    if vehicle_s >= stop_line_s:
        return None
    return (stop_line_s, 0.0)


def effective_leader(vehicle_s: float,
                     vehicle_speed: float,
                     vehicle_length: float,
                     physical_leader: Optional[Tuple[float, float, float]],
                     signal_leader: Optional[Tuple[float, float]],
                     ) -> Tuple[float, float]:
    """Choose the most constraining leader (physical vs signal).

    Args:
        vehicle_s:       longitudinal position of ego vehicle's front bumper
        vehicle_speed:   ego speed (m/s)
        vehicle_length:  ego length (m) — used only for gap computation
        physical_leader: (s, speed, length) of the vehicle ahead, or None
        signal_leader:   (stop_line_s, 0.0) from virtual_signal_leader, or None

    Returns:
        (gap, delta_v) to feed into idm_acceleration.
        gap = distance from ego front bumper to nearest obstacle rear bumper.
        delta_v = ego_speed − leader_speed (positive = closing).
    """
    candidates = []

    if physical_leader is not None:
        leader_s, leader_speed, leader_length = physical_leader
        gap = leader_s - leader_length - vehicle_s
        dv = vehicle_speed - leader_speed
        candidates.append((gap, dv))

    if signal_leader is not None:
        sig_s, sig_speed = signal_leader
        # Virtual leader has zero length — gap is just distance to stop line
        gap = sig_s - vehicle_s
        dv = vehicle_speed - sig_speed
        candidates.append((gap, dv))

    if not candidates:
        # Free road: infinite gap, no closing speed
        return (1e6, 0.0)

    # Pick the candidate with the smallest gap (most constraining)
    candidates.sort(key=lambda c: c[0])
    return candidates[0]
