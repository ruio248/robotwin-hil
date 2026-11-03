"""Adapters admitted to the canonical runner.

Other historical collectors use adapters.legacy and keep their own protocols
until they have been checked against this runner's lifecycle.
"""
from importlib import import_module

from .base import TaskAdapter

_ADAPTERS = {
    "handover_to_tray": (".handover_to_tray", "HandoverToTrayAdapter"),
}


def registered_tasks() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def get_adapter(task_name: str) -> TaskAdapter:
    try:
        module, name = _ADAPTERS[task_name]
    except KeyError as exc:
        raise ValueError(
            f"No shared-runner adapter for {task_name!r}; supported: {registered_tasks()}"
        ) from exc
    return getattr(import_module(module, __name__), name)()
