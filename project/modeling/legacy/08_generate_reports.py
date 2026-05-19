#!/usr/bin/env python3
from pipeline_lib import build_model_summary_for_report, generate_reports


if __name__ == "__main__":
    build_model_summary_for_report()
    generate_reports()
    print("reports generated")
