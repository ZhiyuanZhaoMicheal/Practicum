#!/usr/bin/env python3
import argparse
import pandas as pd

from pipeline_lib import (
    RunContext,
    PANEL_PATH,
    PANEL_NLCD_PATH,
    OUTPUT_DIR,
    fit_ols_and_mixed,
    save_issue_log,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-land-use", action="store_true")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--variant", type=str, default="")
    args = parser.parse_args()

    in_path = args.input
    if not in_path:
        in_path = str(PANEL_NLCD_PATH if args.with_land_use else PANEL_PATH)

    variant = args.variant or ("with_nlcd" if args.with_land_use else "no_nlcd")
    df = pd.read_parquet(in_path)

    ctx = RunContext(issues=[])
    result = fit_ols_and_mixed(ctx, df=df, variant=variant, include_land_use=args.with_land_use)
    ols_path = OUTPUT_DIR / "ols_results.csv"
    mixed_path = OUTPUT_DIR / "mixedlm_results.csv"

    if ols_path.exists():
        old = pd.read_csv(ols_path)
        pd.concat([old, result["ols_coef"]], ignore_index=True).to_csv(ols_path, index=False)
    else:
        result["ols_coef"].to_csv(ols_path, index=False)

    if mixed_path.exists():
        old = pd.read_csv(mixed_path)
        pd.concat([old, result["mixed_coef"]], ignore_index=True).to_csv(mixed_path, index=False)
    else:
        result["mixed_coef"].to_csv(mixed_path, index=False)

    save_issue_log(ctx)
    print(f"saved OLS/Mixed results for variant={variant}")


if __name__ == "__main__":
    main()
