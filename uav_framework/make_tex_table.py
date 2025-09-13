#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert results.csv into LaTeX table rows grouped by experiment map (file name + tree count).
-------------------------------------------------------------------------------------------
Usage:
    python make_tex_table.py --csv results.csv --out results_rows.tex

Import:
    from make_tex_table import make_latex_rows
    tex_str = make_latex_rows("results.csv", "results_rows.tex")
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
import ast, os

def _fmt(v, ndigits=2) -> str:
    if v is None:
        return r"--"
    try:
        f = float(v)
        s = f"{f:.{ndigits}f}".rstrip("0").rstrip(".")
        return s if s != "" else "0"
    except Exception:
        return str(v)

def _latex_escape(s: str) -> str:
    repl = {
        '_': r'\_',
        '%': r'\%',
        '#': r'\#',
        '&': r'\&',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
    }
    return "".join(repl.get(ch, ch) for ch in s)

def _extract_map_info(val):
    d = None
    if isinstance(val, str):
        try:
            d = ast.literal_eval(val)
        except Exception:
            d = None
    elif isinstance(val, dict):
        d = val
    return d if isinstance(d, dict) else None

def make_latex_rows(csv_path: str, out_path: str = None) -> str:
    df = pd.read_csv(csv_path)

    # find method col
    method_col = None
    for c in df.columns:
        if c.lower() == "method":
            method_col = c
            break
    if method_col is None:
        raise ValueError("Cannot find 'Method' column in CSV.")

    if "map" not in df.columns:
        raise ValueError("CSV must contain 'map' column.")

    if "N" in df.columns:
        df["_N_"] = df["N"]
    else:
        df["_N_"] = 0

    # build group labels = file basename + (N=...)
    labels = []
    for i, row in df.iterrows():
        info = _extract_map_info(row["map"])
        label = "map"
        if info:
            if info.get("type") == "csv":
                bn = os.path.basename(info.get("path", ""))
                label = f"{bn} (N={int(row['_N_'])})"
            elif info.get("type") == "random":
                N = row["_N_"]
                seed = info.get("seed", None)
                if seed is not None:
                    label = f"random_seed{seed} (N={int(N)})"
                else:
                    label = f"random (N={int(N)})"
        labels.append(label)
    df["_LABEL_"] = labels

    # sort by label then method
    df = df.sort_values(["_LABEL_", method_col])

    lines = []
    for idx, (label, block) in enumerate(df.groupby("_LABEL_")):
        first = True
        for _, row in block.iterrows():
            left = _latex_escape(str(label)) if first else "    "
            first = False
            mname = str(row[method_col])
            if ':' in mname:
                mname = mname.split(':')[-1]
            elif '.' in mname:
                mname = mname.split('.')[-1]
            lines.append(
                f"{left} & {mname} & {_fmt(row['TCR'], 2)} & {_fmt(row['T_op_min'], 2)} & {_fmt(row['L_total_m'], 1)} \\\\")
        # 在每个分组结束时加一次 \hline
        lines.append(r"\hline")

    tex = "\n".join(lines)
    if out_path:
        Path(out_path).write_text(tex, encoding="utf-8")
    return tex

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Convert results.csv to LaTeX rows grouped by map (filename+N)")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=False)
    args = p.parse_args()
    tex = make_latex_rows(args.csv, args.out)
    print(tex)
