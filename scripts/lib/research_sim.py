"""Research-grade state-based simulator compatibility wrapper.

This module exposes the same public ``simulate`` contract as ``micro_sim`` and
``intersection_sim`` while delegating to the new formal ``sim`` package.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from sim.engine import SimulationEngine, simulate as _simulate


def simulate(vehicles: list,
             approach_visible_length: float,
             fps: int,
             signal_plan=None,
             seed: int = 42,
             record_trajectories: bool = False,
             driver_mix: Optional[Dict[str, float]] = None,
             integration: str = "semi_implicit",
             apply_reaction: bool = False) -> Tuple[list, Dict]:
    """Research simulator entry point.

    Unlike the generic ``sim.engine.simulate`` compatibility function, the
    research wrapper opts into the Phase 5 signal FSM for adaptive plans.
    """
    return _simulate(
        vehicles,
        approach_visible_length,
        fps,
        signal_plan=signal_plan,
        seed=seed,
        record_trajectories=record_trajectories,
        driver_mix=driver_mix,
        integration=integration,
        apply_reaction=apply_reaction,
        signal_engine="fsm",
    )

__all__ = ["SimulationEngine", "simulate"]
