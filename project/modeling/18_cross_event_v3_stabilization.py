#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_main(rel_path: str):
    target = Path(__file__).resolve().parent / rel_path
    spec = importlib.util.spec_from_file_location(target.stem.replace('-', '_'), target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


if __name__ == '__main__':
    main = _load_main('pipelines/02_cross_event_pipeline.py')
    raise SystemExit(main(['stabilize-v3', *sys.argv[1:]]))
