"""Task semantics required by the shared interactive runner."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import Any


class TaskAdapter(ABC):
    """Describe stages and construct recovery from the current live scene.

    Recovery must never reset the environment. Each iterator advance executes
    one stage and yields a progress event; terminal events use
    {"phase": "done", "result": {"success": ..., "plan_success": ...}}.
    The runner owns policy calls, control-source labels, user input and I/O.
    Privileged recovery state must not be added to the policy observation.
    """

    name: str
    prompt: str
    stage_labels: Mapping[int, str]
    default_task_config: str
    default_checkpoint_name: str
    default_frequency: int
    supports_one_stage: bool = False
    supports_sampling: bool = True
    requires_training_gate: bool = False
    default_episode_protocol: str = "legacy-fixed"
    reserved_seed_ranges: tuple = ()

    def initialize_episode(self, env: Any) -> None:
        """Initialize adapter metadata after setup, without moving the scene."""

    def validate_action_chunk(self, chunk: Any) -> None:
        """Validate decoded absolute joint vectors before executing any action.

        The runner converts the policy wire format using the execution adapter;
        this hook never receives XPolicyLab's per-arm action dictionaries.
        """

    def recovery_stage_rejection(self, state: Mapping[str, Any], stage_id: int | None) -> str | None:
        """Optional motion-free preflight reason; None permits confirmation.

        Adapters must still recheck prerequisites when the iterator advances.
        The default preserves adapters whose manual overrides have own logic.
        """
        return None

    @abstractmethod
    def infer_recovery_state(self, env: Any) -> dict[str, Any]:
        """Read live state, including stage/branch and recoverability."""

    @abstractmethod
    def recovery_stage_iterator(
        self, env: Any, start_stage_id: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Create an expert suffix; construction itself must not move arms."""

    @abstractmethod
    def success_metrics(self, env: Any) -> dict[str, Any]:
        """Task-specific terminal diagnostics, alongside env.check_success()."""

    def describe_recovery_state(self, state: Mapping[str, Any]) -> Sequence[str]:
        return [f"{key}={value}" for key, value in state.items()]

    def play_prefix_until_placement(self, env: Any) -> bool:
        raise NotImplementedError(f"{self.name} has no one-stage expert prefix")

    def handoff_record(self, env: Any) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} has no paired one-stage handoff")
