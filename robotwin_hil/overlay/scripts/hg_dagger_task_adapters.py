"""Compatibility exports and registry for legacy HG-DAgger task adapters.

These callbacks are separate from the canonical dagger runner's adapters.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _add_repository_root() -> None:
    # Both source overlay scripts and assembled runtimes live beneath the repo.
    for ancestor in Path(__file__).resolve().parents:
        if (ancestor / "dagger").is_dir():
            root = str(ancestor)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


_add_repository_root()

from dagger.adapters.legacy import Adapter
from dagger.adapters.pick_dual_bottles import get_adapter as _pick_dual_bottles_adapter
from dagger.adapters.place_can_basket import get_adapter as _place_basket_adapter
from dagger.adapters.put_bottles_dustbin import get_adapter as _put_bottles_dustbin_adapter
from dagger.adapters.rotate_qrcode import get_adapter as _rotate_qrcode_adapter
from dagger.adapters.stack_blocks_two import get_adapter as _stack_blocks_two_adapter


_ADAPTER_FACTORIES = {
    "place_can_basket": _place_basket_adapter,
    "rotate_qrcode": _rotate_qrcode_adapter,
    "pick_dual_bottles": _pick_dual_bottles_adapter,
    "put_bottles_dustbin": _put_bottles_dustbin_adapter,
    "stack_blocks_two": _stack_blocks_two_adapter,
}


def _new_task_adapter(task_name: str) -> Adapter:
    if task_name in {"pick_dual_bottles", "put_bottles_dustbin", "stack_blocks_two"}:
        return _ADAPTER_FACTORIES[task_name]()
    raise ValueError(f"no HG-DAgger adapter registered for {task_name!r}")


def get_adapter(task_name: str) -> Adapter:
    if task_name in _ADAPTER_FACTORIES:
        return _ADAPTER_FACTORIES[task_name]()
    raise ValueError(f"no HG-DAgger adapter registered for {task_name!r}")
