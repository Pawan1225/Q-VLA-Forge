from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ExperimentRecord:
    """Common record format for all Q-VLA Forge experiments."""

    experiment_id: str
    domain: str
    method: str
    seed: int

    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert the experiment record into a serializable dictionary."""
        return asdict(self)
