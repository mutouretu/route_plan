# run_experiments.py
# CLI runner for the UAV experiment framework
import argparse
from framework import run_from_config
from make_tex_table import make_latex_rows
def main():
    parser = argparse.ArgumentParser(description="Run UAV spraying experiments from config")
    parser.add_argument("--config", "-c", required=True, help="Path to config JSON/YAML")
    args = parser.parse_args()
    df = run_from_config(args.config)
    print(df.head())
    print("Saved results to path set in config.")

    tex_str = make_latex_rows("results.csv", "results_rows.tex")
    print(tex_str)

if __name__ == "__main__":
    main()
