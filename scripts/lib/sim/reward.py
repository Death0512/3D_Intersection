"""Phase 9 — reward functions for traffic-control research."""
from __future__ import annotations


def delay_reward(mean_wait_s: float, throughput_vph: float = 0.0,
                 throughput_weight: float = 0.001) -> float:
    """Reward high throughput and penalize delay."""
    return -float(mean_wait_s) + throughput_weight * float(throughput_vph)


def queue_reward(total_queue: float, average_speed: float = 0.0,
                 speed_weight: float = 0.1) -> float:
    """Reward low queue length and non-zero movement."""
    return -float(total_queue) + speed_weight * float(average_speed)


def observation_reward(observation) -> float:
    """Generic reward from a ``SimulationObservation``-like object."""
    lanes = getattr(observation, "lanes", {})
    total_queue = sum(l.queue_length for l in lanes.values())
    avg_speed = 0.0
    if lanes:
        avg_speed = sum(l.average_speed for l in lanes.values()) / len(lanes)
    return queue_reward(total_queue, avg_speed)


__all__ = ["delay_reward", "queue_reward", "observation_reward"]
