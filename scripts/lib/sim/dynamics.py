"""Vehicle dynamics models for the microscopic simulator.

Phase 2 formalizes the dynamics layer:

* ``DriverProfile`` presets (aggressive / normal / cautious) capture
  heterogeneity in acceleration, deceleration, gaps, headway, and reaction.
* ``build_idm_params`` turns a profile + desired speed into calibrated
  IDM parameters, with optional driver-to-driver noise.
* ``IntegrationMethod`` selects how acceleration is integrated to speed/position.
* ``IDMDynamics`` computes IDM acceleration from leader/signal state and can
  apply a perceptual reaction-time delay (the command issued at frame ``t`` is
  only applied ``reaction_frames`` later).
"""
from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import car_following as CF

from .config import SimulationConfig
from .state import VehicleState


@dataclass
class IntegrationResult:
    speed: float
    position: float


class IntegrationMethod(enum.Enum):
    EULER = "euler"
    SEMI_IMPLICIT = "semi_implicit"


@dataclass
class DriverProfile:
    """Calibration preset for one behavioral class of driver."""

    name: str
    max_accel: float
    comfortable_decel: float
    min_gap: float
    time_headway: float
    delta: int = 4
    reaction_time_s: float = 0.6
    desired_speed_factor: float = 1.0
    speed_noise_std: float = 0.0

    def reaction_frames(self, fps: int) -> int:
        return int(round(self.reaction_time_s * fps))


# ponytail: three behavioral classes are enough for heterogeneity studies.
DRIVER_PROFILES: Dict[str, DriverProfile] = {
    "aggressive": DriverProfile(
        name="aggressive",
        max_accel=3.0,
        comfortable_decel=2.5,
        min_gap=1.0,
        time_headway=0.8,
        delta=4,
        reaction_time_s=0.4,
        desired_speed_factor=1.1,
        speed_noise_std=0.5,
    ),
    "normal": DriverProfile(
        name="normal",
        max_accel=2.5,
        comfortable_decel=2.0,
        min_gap=2.0,
        time_headway=1.5,
        delta=4,
        reaction_time_s=0.6,
        desired_speed_factor=1.0,
        speed_noise_std=0.4,
    ),
    "cautious": DriverProfile(
        name="cautious",
        max_accel=1.8,
        comfortable_decel=1.8,
        min_gap=2.5,
        time_headway=2.0,
        delta=4,
        reaction_time_s=1.0,
        desired_speed_factor=0.9,
        speed_noise_std=0.3,
    ),
}

# Default mix used when a scenario requests heterogeneous drivers.
DEFAULT_DRIVER_MIX: Dict[str, float] = {
    "aggressive": 0.25,
    "normal": 0.60,
    "cautious": 0.15,
}


def build_idm_params(profile: DriverProfile,
                     desired_speed: float,
                     rng: Optional[random.Random] = None) -> CF.IDMParams:
    """Build IDM params from a driver profile and a target desired speed.

    Optional Gaussian noise on the desired speed makes individual drivers
    heterogeneous even within the same profile.  Deterministic when ``rng`` is
    None or the profile has zero noise.
    """
    speed = profile.desired_speed_factor * desired_speed
    if rng is not None and profile.speed_noise_std > 0.0:
        speed = max(1.0, speed + rng.gauss(0.0, profile.speed_noise_std))
    return CF.IDMParams(
        desired_speed=speed,
        max_accel=profile.max_accel,
        comfortable_decel=profile.comfortable_decel,
        min_gap=profile.min_gap,
        time_headway=profile.time_headway,
        delta=profile.delta,
    )


def pick_profile(mix: Dict[str, float],
                 rng: random.Random) -> DriverProfile:
    """Sample a driver profile according to a weight dict."""
    names = list(mix.keys())
    weights = [mix[n] for n in names]
    total = sum(weights)
    if total <= 0:
        return DRIVER_PROFILES["normal"]
    weights = [w / total for w in weights]
    choice = rng.choices(names, weights=weights, k=1)[0]
    return DRIVER_PROFILES[choice]


def integrate(vehicle: VehicleState, dt: float,
              method: IntegrationMethod) -> IntegrationResult:
    """Integrate one vehicle's speed and position for one step.

    EULER uses the pre-step speed for the position update; SEMI_IMPLICIT uses the
    updated speed.  Both clamp speed to be non-negative.  SEMI_IMPLICIT is the
    default because it is more stable for car-following at fixed frame rates.
    """
    new_speed = max(0.0, vehicle.speed + vehicle.accel * dt)
    if method == IntegrationMethod.EULER:
        new_s = vehicle.s + vehicle.speed * dt
    else:
        new_s = vehicle.s + new_speed * dt
    return IntegrationResult(speed=new_speed, position=new_s)


class IDMDynamics:
    """IDM-based longitudinal dynamics with optional reaction-time delay."""

    def __init__(self, apply_reaction: bool = False):
        self.apply_reaction = apply_reaction

    def acceleration(self,
                     vehicle: VehicleState,
                     physical_leader: Optional[VehicleState],
                     signal_green: bool,
                     config: SimulationConfig) -> float:
        phys: Optional[Tuple[float, float, float]] = None
        if physical_leader is not None:
            phys = (physical_leader.s, physical_leader.speed, physical_leader.length)
            vehicle.leader_id = physical_leader.vid
        else:
            vehicle.leader_id = None

        sig = CF.virtual_signal_leader(
            vehicle.s, config.stop_line_s, vehicle.speed, signal_green)
        gap, delta_v = CF.effective_leader(
            vehicle.s, vehicle.speed, vehicle.length, phys, sig)
        vehicle.gap = gap
        return CF.idm_acceleration(
            vehicle.speed,
            vehicle.desired_speed,
            gap,
            delta_v,
            vehicle.idm_params,
        )

    def apply(self, vehicle: VehicleState, target: float) -> None:
        """Store the freshly computed target accel and expose the reaction-delayed
        command as ``vehicle.accel``.

        With ``apply_reaction=False`` the latest command is used immediately.
        With ``apply_reaction=True`` the command issued ``reaction_frames`` ago
        is applied, modeling perceptual/actuation delay.
        """
        vehicle._accel_history.append(target)
        if not self.apply_reaction or vehicle.reaction_frames <= 0:
            vehicle.accel = target
            return
        idx = len(vehicle._accel_history) - 1 - vehicle.reaction_frames
        if idx < 0:
            # Not enough history yet: apply the oldest available command.
            vehicle.accel = vehicle._accel_history[0]
        else:
            vehicle.accel = vehicle._accel_history[idx]
