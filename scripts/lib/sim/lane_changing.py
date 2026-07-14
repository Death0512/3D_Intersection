"""Phase 8 — MOBIL lane-changing decision model.

This module is intentionally independent from the engine for now.  It provides
the standard incentive/safety calculations needed before lane changing is wired
into the multi-lane state update loop.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MOBILParams:
    politeness: float = 0.3
    threshold: float = 0.2
    safe_decel: float = 4.0
    right_bias: float = 0.0


@dataclass(frozen=True)
class LaneChangeContext:
    current_accel: float
    target_accel: float
    old_follower_accel_before: float = 0.0
    old_follower_accel_after: float = 0.0
    new_follower_accel_before: float = 0.0
    new_follower_accel_after: float = 0.0
    target_is_right: bool = False


def mobil_incentive(ctx: LaneChangeContext,
                    params: MOBILParams = MOBILParams()) -> float:
    ego_gain = ctx.target_accel - ctx.current_accel
    old_follower_gain = ctx.old_follower_accel_after - ctx.old_follower_accel_before
    new_follower_gain = ctx.new_follower_accel_after - ctx.new_follower_accel_before
    bias = params.right_bias if ctx.target_is_right else 0.0
    return ego_gain + params.politeness * (old_follower_gain + new_follower_gain) + bias


def mobil_safe(ctx: LaneChangeContext,
               params: MOBILParams = MOBILParams()) -> bool:
    return ctx.new_follower_accel_after >= -abs(params.safe_decel)


def should_change_lane(ctx: LaneChangeContext,
                       params: MOBILParams = MOBILParams()) -> bool:
    return mobil_safe(ctx, params) and mobil_incentive(ctx, params) > params.threshold


__all__ = [
    "MOBILParams",
    "LaneChangeContext",
    "mobil_incentive",
    "mobil_safe",
    "should_change_lane",
]
