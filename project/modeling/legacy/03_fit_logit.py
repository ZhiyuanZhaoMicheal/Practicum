#!/usr/bin/env python3
import argparse
import pandas as pd

from pipeline_lib import (
    RunContext,
    PANEL_PATH,
    PANEL_NLCD_PATH,
    OUTPUT_DIR,
    fit_logit,
    load_json,
    CONFIG_DEFAULTS,
    save_issue_log,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-land-use", action="store_true")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--damage-threshold", type=float, default=None)
    args = parser.parse_args()

    defaults = load_json(CONFIG_DEFAULTS)
    dthr = args.damage_threshold if args.damage_threshold is not None else float(defaults["damage_threshold"])

    in_path = args.input
    if not in_path:
        in_path = str(PANEL_NLCD_PATH if args.with_land_use else PANEL_PATH)

    variant = args.variant or ("with_nlcd" if args.with_land_use else "no_nlcd")
    df = pd.read_parquet(in_path)

    ctx = RunContext(issues=[])
    result = fit_logit(ctx, df=df, variant=variant, include_land_use=args.with_land_use, damage_threshold=dthr)

    out = OUTPUT_DIR / "logit_results.csv"
    if out.exists() and not result["coef"].empty:
        old = pd.read_csv(out)
        pd.concat([old, result["coef"]], ignore_index=True).to_csv(out, index=False)
    elif not result["coef"].empty:
        result["coef"].to_csv(out, index=False)

    if not result["marginal"].empty:
        result["marginal"].to_csv(OUTPUT_DIR / f"logit_marginal_effects_{variant}.csv", index=False)
    if not result["roc"].empty:
        result["roc"].to_csv(OUTPUT_DIR / f"logit_roc_{variant}.csv", index=False)
    if not result["calibration"].empty:
        result["calibration"].to_csv(OUTPUT_DIR / f"logit_calibration_{variant}.csv", index=False)

    save_issue_log(ctx)
    print(f"saved logit outputs for variant={variant}")


if __name__ == "__main__":
    main()
