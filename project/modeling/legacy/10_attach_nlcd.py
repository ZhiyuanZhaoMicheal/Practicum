#!/usr/bin/env python3
import argparse

from pipeline_lib import RunContext, PANEL_PATH, PANEL_NLCD_PATH, attach_nlcd, save_issue_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(PANEL_PATH))
    parser.add_argument("--output", type=str, default=str(PANEL_NLCD_PATH))
    parser.add_argument("--nlcd-dir", type=str, default="")
    args = parser.parse_args()

    ctx = RunContext(issues=[])
    attach_nlcd(ctx=ctx, panel_path=args.input, output_path=args.output, nlcd_dir=args.nlcd_dir or None)
    save_issue_log(ctx)
    print("NLCD attachment completed")
