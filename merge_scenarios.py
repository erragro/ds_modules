#!/usr/bin/env python3
"""
merge_step3_outputs.py
Purpose: Pre-merge Step-3 outputs so Step-4 has units/R²/beta baked in.
Inputs:
  - raw/clean_outputs/out_refined/scenarios_refined.csv
  - raw/clean_outputs/out_refined/elasticity_segments.csv
Output:
  - clean_outputs/out_refined/scenarios_merged.csv
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

BASE = Path("raw/clean_outputs/out_refined")
SCN_PATH = BASE / "scenarios_refined.csv"
FIT_PATH = BASE / "elasticity_segments.csv"
OUT_PATH = BASE / "scenarios_merged.csv"

def norm(s):
    if pd.isna(s): return s
    return str(s).strip().lower()

def main():
    assert SCN_PATH.exists(), f"Missing: {SCN_PATH}"
    assert FIT_PATH.exists(), f"Missing: {FIT_PATH}"

    scn = pd.read_csv(SCN_PATH)
    fit = pd.read_csv(FIT_PATH)

    # Normalize column names
    scn.columns = [str(c).strip() for c in scn.columns]
    fit.columns = [str(c).strip() for c in fit.columns]

    # Keep raw for debugging
    for col in ("Category", "SKU"):
        if col in scn.columns: scn[f"{col}_raw"] = scn[col]
        if col in fit.columns: fit[f"{col}_raw"] = fit[col]

    # Normalize CELL VALUES on keys
    for col in ("Category", "SKU"):
        if col in scn.columns: scn[col] = scn[col].map(norm)
        if col in fit.columns: fit[col] = fit[col].map(norm)

    # What to bring from fit
    attach_cols = ["fit_confidence", "elasticity_beta", "r2",
                   "total_units_observed", "n_points", "unique_prices",
                   "price_min", "price_median", "price_max"]
    attach_cols = [c for c in attach_cols if c in fit.columns]

    merged = scn.copy()

    # Prefer join on SKU if SKU present in both and not all null
    can_join_sku = ("SKU" in scn.columns and "SKU" in fit.columns
                    and scn["SKU"].notna().any() and fit["SKU"].notna().any())
    if can_join_sku:
        merged = merged.merge(
            fit[[c for c in (["SKU"] + attach_cols) if c in fit.columns]].drop_duplicates(),
            on="SKU", how="left", validate="m:1"
        )

    # For remaining rows missing units, try Category join
    if "Category" in scn.columns and "Category" in fit.columns:
        need_cat = merged["total_units_observed"].isna() if "total_units_observed" in merged.columns else (merged["Category"].notna())
        if need_cat.any():
            merged = merged.merge(
                fit[[c for c in (["Category"] + attach_cols) if c in fit.columns]].drop_duplicates(),
                on="Category", how="left", suffixes=("", "_by_cat"), validate="m:1"
            )
            # Fill any missing from _by_cat columns
            for c in attach_cols:
                src = c
                byc = f"{c}_by_cat"
                if byc in merged.columns:
                    if src not in merged.columns:
                        merged[src] = merged[byc]
                    else:
                        merged[src] = merged[src].fillna(merged[byc])
                    merged.drop(columns=[byc], inplace=True, errors="ignore")

    # Final safety defaults
    if "total_units_observed" not in merged.columns:
        merged["total_units_observed"] = 1.0
    merged["total_units_observed"] = pd.to_numeric(merged["total_units_observed"], errors="coerce").fillna(1.0)
    if "r2" not in merged.columns: merged["r2"] = 0.0
    merged["r2"] = pd.to_numeric(merged["r2"], errors="coerce").fillna(0.0)

    # Diagnostics
    have_units = (merged["total_units_observed"] > 1).sum()
    have_r2 = (merged["r2"] > 0).sum()
    print(f"Rows: {len(merged)} | with units>1: {have_units} | with r2>0: {have_r2}")

    # Drop duplicate rows on key fields (if any)
    dedupe_keys = [k for k in ["SKU", "Category", "scenario_price_change", "margin_assumption"] if k in merged.columns]
    merged = merged.drop_duplicates(subset=dedupe_keys)

    # Write
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Wrote → {OUT_PATH}")

if __name__ == "__main__":
    main()