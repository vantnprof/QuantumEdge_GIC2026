"""One-shot data cache service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from quantumedge.config import ensure_runtime_dirs
from quantumedge.data.loaders import load_all


def _row_count(value: Any) -> int | None:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return int(len(value))
    return None


def build_manifest() -> dict[str, Any]:
    settings = ensure_runtime_dirs()
    data = load_all()
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(settings.data_dir),
        "datasets": {
            "sp500": {"rows": _row_count(data["sp500"])},
            "vix": {"rows": _row_count(data["vix"])},
            "oxford_man": {
                "rows": _row_count(data["oxford_man"]),
                "source": data["oxford_source"],
            },
            "mackey_glass": {"rows": _row_count(data["mackey_glass"])},
            "lorenz": {"rows": _row_count(data["lorenz"])},
        },
    }
    manifest_path = settings.data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[data-service] manifest written to {manifest_path}")
    return manifest


def main() -> None:
    build_manifest()


if __name__ == "__main__":
    main()
