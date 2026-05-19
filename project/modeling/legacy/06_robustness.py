#!/usr/bin/env python3
import pandas as pd

from pipeline_lib import RunContext, PANEL_PATH, CONFIG_DEFAULTS, load_json, run_robustness, save_issue_log


def main() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    defaults = load_json(CONFIG_DEFAULTS)
    ctx = RunContext(issues=[])
    out = run_robustness(ctx=ctx, base_panel=panel, defaults=defaults, include_land_use=False)
    save_issue_log(ctx)
    print(f"saved robustness rows={len(out)}")


if __name__ == "__main__":
    main()
