#!/usr/bin/env python3
import argparse
import pandas as pd

from pipeline_lib import RunContext, RECOVERY_PATH, OUTPUT_DIR, fit_cox, save_issue_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(RECOVERY_PATH))
    parser.add_argument("--variant", type=str, default="no_nlcd")
    parser.add_argument("--with-land-use", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    ctx = RunContext(issues=[])
    result = fit_cox(ctx, recovery_df=df, variant=args.variant, include_land_use=args.with_land_use)

    if not result["coef"].empty:
        out = OUTPUT_DIR / "cox_results.csv"
        if out.exists():
            old = pd.read_csv(out)
            pd.concat([old, result["coef"]], ignore_index=True).to_csv(out, index=False)
        else:
            result["coef"].to_csv(out, index=False)

    if not result["km"].empty:
        result["km"].to_csv(OUTPUT_DIR / f"cox_km_{args.variant}.csv", index=False)
    if not result["ph"].empty:
        result["ph"].to_csv(OUTPUT_DIR / f"cox_ph_test_{args.variant}.csv", index=False)

    save_issue_log(ctx)
    print(f"saved cox outputs for variant={args.variant}")


if __name__ == "__main__":
    main()
