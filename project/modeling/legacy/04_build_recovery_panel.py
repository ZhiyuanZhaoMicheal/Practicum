#!/usr/bin/env python3
import argparse
import pandas as pd

from pipeline_lib import RunContext, PANEL_PATH, RECOVERY_PATH, build_recovery_panel, load_json, CONFIG_DEFAULTS, save_issue_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(PANEL_PATH))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=str, default=str(RECOVERY_PATH))
    args = parser.parse_args()

    defaults = load_json(CONFIG_DEFAULTS)
    thr = args.threshold if args.threshold is not None else float(defaults["recovery_threshold"])

    df = pd.read_parquet(args.input)
    ctx = RunContext(issues=[])
    rec = build_recovery_panel(ctx=ctx, panel=df, threshold=thr, output_path=args.output)
    save_issue_log(ctx)
    print(f"saved recovery rows={len(rec)} threshold={thr}")


if __name__ == "__main__":
    main()
