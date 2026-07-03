from __future__ import annotations

from importlib import import_module
import sys
from typing import Any


def export(module_name: str, module_globals: dict[str, Any]) -> None:
    impl = import_module(f"ppwam.{module_name}")
    current_name = module_globals.get("__name__")
    names = getattr(
        impl,
        "__all__",
        [name for name in dir(impl) if not name.startswith("_")],
    )
    for name in names:
        module_globals[name] = getattr(impl, name)
    module_globals["__all__"] = list(names)
    if current_name and current_name != "__main__":
        sys.modules[current_name] = impl


def run_main(module_name: str) -> None:
    impl = import_module(f"ppwam.{module_name}")
    main = getattr(impl, "main", None)
    if main is None:
        raise SystemExit(f"ppwam.{module_name} has no CLI entrypoint")
    main()
