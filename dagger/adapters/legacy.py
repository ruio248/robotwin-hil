"""Legacy HG-DAgger callback contract, separate from the canonical runner adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Adapter:
    task_name: str
    snapshot: Any
    classify: Any
    recovery: Any
    quality: Any
    stage_labels: dict[int, str]
