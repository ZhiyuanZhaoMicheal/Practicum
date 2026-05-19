#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

from pipeline_lib import RunContext, build_pixel_panel, load_json, CONFIG_DEFAULTS, PANEL_PATH, save_issue_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-threshold", type=float, default=None)
    parser.add_argument("--damage-threshold", type=float, default=None)
    parser.add_argument("--exclude-types", type=str, default="")
    parser.add_argument("--output", type=str, default=str(PANEL_PATH))
    args = parser.parse_args()

    defaults = load_json(CONFIG_DEFAULTS)
    pre_thr = args.pre_threshold if args.pre_threshold is not None else float(defaults["pre_ntl_threshold"])
    dmg_thr = args.damage_threshold if args.damage_threshold is not None else float(defaults["damage_threshold"])
    excludes = [x.strip() for x in args.exclude_types.split(",") if x.strip()] or None

    ctx = RunContext(issues=[])
    panel = build_pixel_panel(
        ctx=ctx,
        pre_threshold=pre_thr,
        damage_threshold=dmg_thr,
        exclude_types=excludes,
        output_path=Path(args.output),
    )
    save_issue_log(ctx)
    print(f"saved panel rows={len(panel)} path={args.output}")


if __name__ == "__main__":
    main()
