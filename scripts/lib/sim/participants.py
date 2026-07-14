"""Phase 8 — additional traffic participant abstractions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentType(str, Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    BUS = "bus"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class ParticipantSpec:
    agent_type: AgentType
    length_m: float
    width_m: float
    desired_speed_mps: float
    priority: int = 0


DEFAULT_PARTICIPANTS = {
    AgentType.VEHICLE: ParticipantSpec(AgentType.VEHICLE, 4.5, 1.8, 13.9, 0),
    AgentType.PEDESTRIAN: ParticipantSpec(AgentType.PEDESTRIAN, 0.5, 0.5, 1.4, 1),
    AgentType.CYCLIST: ParticipantSpec(AgentType.CYCLIST, 1.8, 0.6, 5.5, 1),
    AgentType.BUS: ParticipantSpec(AgentType.BUS, 12.0, 2.6, 11.0, 0),
    AgentType.EMERGENCY: ParticipantSpec(AgentType.EMERGENCY, 5.0, 2.0, 20.0, 10),
}


def participant_spec(agent_type: str | AgentType) -> ParticipantSpec:
    at = agent_type if isinstance(agent_type, AgentType) else AgentType(agent_type)
    return DEFAULT_PARTICIPANTS[at]


__all__ = ["AgentType", "ParticipantSpec", "DEFAULT_PARTICIPANTS", "participant_spec"]
