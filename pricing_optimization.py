#!/usr/bin/env python3
# pricing_optimization.py
# Step 4 — Pricing optimization aligned with Step 3 (Category + SKU).
# Supports pre-merged scenarios (preferred) and raw scenarios+fit.
# Adds strict business guardrails: Δp cap, fit floors, β sanity, and in-range enforcement.

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ============================ CONFIG ============================
# Point to the PRE-MERGED scenarios by default:
SCENARIOS_PATH = Path("raw/clean_outputs/out_refined/scenarios_merged.csv")
# Keep FIT_PATH available (used only if scenarios are NOT merged):
FIT_PATH       = Path("raw/clean_outputs/out_refined/elasticity_segments.csv")
OUT_DIR        = SCENARIOS_PATH.parent / "out_opt"

# Guardrails (tightened)
INIT_PREFER_CONFIDENCE = "high"      # "high" | "low" | "any"
INIT_ALLOW_LOW_IF_UNITS = 20000.0    # admit low_conf only if very large volume
INIT_MIN_PROFIT_PCT = 0.02           # require ≥2% profit uplift

# R² handling: down-weight low-fit segments, but don’t zero them out
USE_R2_IN_SCORE = True
R2_FLOOR = 0.05                       # stronger push toward better fits

# Cap extreme price moves at Step-4 (set to None to disable)
MAX_ABS_DELTA = 0.15                  # |Δp| ≤ 15%

TOPN_PRINT = 10
WRITE_EXCEL = True
EXCEL_NAME  = "pricing_recommendations.xlsx"
# ===============================================================

def section(title: str):
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")

def ensure_file(path: Path):
    if not path.exists():
        print(f"ERROR: file not found → {path}", file=sys.stderr)
        sys.exit(1)

def load_csv(path: Path, required: list[str] | None = None) -> pd.DataFrame:
    ensure_file(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"ERROR: columns missing in {path.name}: {missing}", file=sys.stderr)
            sys.exit(1)
    return df

def norm_val(s):
    if pd.isna(s): return s
    return str(s).strip().lower()

def is_premerged(scn: pd.DataFrame) -> bool:
    need = {"total_units_observed", "r2", "elasticity_beta"}
    return need.issubset(set(scn.columns))

def dedupe_scenarios(df: pd.DataFrame, seg_cols: list[str]) -> pd.DataFrame:
    keys = [c for c in seg_cols if c in df.columns] + \
           [c for c in ["scenario_price_change", "margin_assumption"] if c in df.columns]
    return df.drop_duplicates(subset=keys) if keys else df.drop_duplicates()

def attach_units_and_fit(df: pd.DataFrame, fit_df: pd.DataFrame | None, seg_cols: list[str]) -> pd.DataFrame:
    """
    If pre-merged (units/r2/beta already present), return as-is.
    Else, merge from fit on SKU first, then Category.
    """
    left = df.copy();  left.columns = [str(c).strip() for c in left.columns]

    if is_premerged(left) or fit_df is None:
        # Ensure required columns exist
        if "total_units_observed" not in left.columns: left["total_units_observed"] = 1.0
        if "r2" not in left.columns: left["r2"] = 0.0
        if "elasticity_beta" not in left.columns: left["elasticity_beta"] = np.nan
        if "fit_confidence" not in left.columns: left["fit_confidence"] = ""
        if "in_hist_range" not in left.columns: left["in_hist_range"] = True
        return left

    right = fit_df.copy(); right.columns = [str(c).strip() for c in right.columns]
    # Normalize cell values for join keys
    for col in ("Category", "SKU"):
        if col in left.columns:  left[col]  = left[col].map(norm_val)
        if col in right.columns: right[col] = right[col].map(norm_val)

    # Accept aliases for units
    aliases_units = ["total_units_observed", "total_units", "units_total", "total_qty", "units"]
    units_col = next((a for a in aliases_units if a in right.columns), None)

    want = seg_cols + ["fit_confidence", "elasticity_beta", "r2"]
    have = all(c in right.columns for c in want) and (units_col is not None)

    if have:
        take = right[seg_cols + [units_col, "fit_confidence", "elasticity_beta", "r2"]].drop_duplicates().copy()
        if units_col != "total_units_observed":
            take = take.rename(columns={units_col: "total_units_observed"})
        merged = left.merge(take, on=seg_cols, how="left", validate="m:1")
    else:
        merged = left.copy()

    # safety nets
    if "total_units_observed" not in merged.columns:
        merged["total_units_observed"] = 1.0
    merged["total_units_observed"] = pd.to_numeric(merged["total_units_observed"], errors="coerce").fillna(1.0)

    if "fit_confidence" not in merged.columns:
        merged["fit_confidence"] = ""
    if "elasticity_beta" not in merged.columns:
        merged["elasticity_beta"] = np.nan
    if "r2" not in merged.columns:
        merged["r2"] = 0.0
    if "in_hist_range" not in merged.columns:
        merged["in_hist_range"] = True

    return merged

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def apply_confidence_filters(df: pd.DataFrame,
                             seg_cols: list[str],
                             prefer_conf: str,
                             allow_low_if_units: float | None) -> pd.DataFrame:
    d = df.copy()
    if "in_hist_range" in d.columns:
        d = d[d["in_hist_range"] == True].copy()

    if "confidence" not in d.columns:
        return d

    prefer_conf = (prefer_conf or "high").lower()
    if prefer_conf not in ("high","low","any"): prefer_conf = "high"
    if prefer_conf == "any": return d

    conf = d["confidence"].astype(str).fillna("")
    high_mask = conf.str.contains("high_conf")
    low_mask  = conf.str.contains("low_conf")

    d_high, d_low = d[high_mask].copy(), d[low_mask].copy()
    if prefer_conf == "high":
        if not len(d_low) or not allow_low_if_units:
            return d_high if len(d_high) else d_low
        d_low_ok = d_low[pd.to_numeric(d_low["total_units_observed"], errors="coerce").fillna(0.0) >= float(allow_low_if_units)]
        return pd.concat([d_high, d_low_ok], ignore_index=True) if len(d_high) or len(d_low_ok) else d_low
    return d_low if len(d_low) else d_high

def pick_best_moves(df: pd.DataFrame,
                    seg_cols: list[str],
                    min_profit_pct: float,
                    use_r2_in_score: bool,
                    r2_floor: float) -> pd.DataFrame:
    """
    Pick argmax per (segment × margin) using R²-weighted uplift score.
    Robust against NaNs/inf and empty groups.
    """
    required_cols = ["profit_pct_change", "margin_assumption"]
    if not all(c in df.columns for c in required_cols):
        return pd.DataFrame()

    d = df.copy()
    d = coerce_numeric(d, ["profit_pct_change", "revenue_pct_change",
                           "margin_assumption", "total_units_observed", "r2",
                           "scenario_price_change"])

    # Step-4 Δp cap
    if MAX_ABS_DELTA is not None and "scenario_price_change" in d.columns:
        d = d[d["scenario_price_change"].abs() <= float(MAX_ABS_DELTA)]

    # Drop rows missing critical keys or margin
    d = d.dropna(subset=seg_cols + ["margin_assumption"])

    # Profit guard
    d = d[d["profit_pct_change"] > float(min_profit_pct)].copy()
    if d.empty: return pd.DataFrame()

    # Defaults/sanitization
    if "total_units_observed" not in d.columns:
        d["total_units_observed"] = 1.0
    d["total_units_observed"] = d["total_units_observed"].astype(float).fillna(1.0)
    d["r2"] = (d["r2"].astype(float).fillna(0.0) if "r2" in d.columns else 0.0)

    # Build score
    if use_r2_in_score:
        d["score_weight"] = np.maximum(d["r2"], float(r2_floor))
        d["uplift_score"] = d["profit_pct_change"] * d["total_units_observed"] * d["score_weight"]
    else:
        d["uplift_score"] = d["profit_pct_change"] * d["total_units_observed"]

    # Remove non-finite scores
    d = d[np.isfinite(d["uplift_score"])].copy()
    if d.empty: return pd.DataFrame()

    # Group & argmax safely
    grp_cols = seg_cols + ["margin_assumption"]
    d = d.dropna(subset=grp_cols)
    if d.empty: return pd.DataFrame()

    try:
        idx = d.groupby(grp_cols, dropna=False)["uplift_score"].idxmax()
    except ValueError:
        return pd.DataFrame()

    if hasattr(idx, "isna"):
        idx = idx[~idx.isna()]
    if len(idx) == 0:
        return pd.DataFrame()

    picks = d.loc[idx.values].copy().reset_index(drop=True)

    # Column ordering (keep what exists)
    keep = list(dict.fromkeys(
        seg_cols + [
            "margin_assumption", "scenario_price_change", "scenario_price_new",
            "profit_pct_change", "revenue_pct_change", "confidence",
            "fit_confidence", "total_units_observed", "elasticity_beta", "r2",
            "uplift_score"
        ]
    ))
    keep = [c for c in keep if c in picks.columns]
    return picks[keep]

def portfolio_rollup(picks: pd.DataFrame) -> pd.DataFrame:
    if picks.empty or "margin_assumption" not in picks.columns:
        return pd.DataFrame(columns=["margin_assumption","segments_count","vw_revenue_pct","vw_profit_pct"])

    d = coerce_numeric(picks, ["revenue_pct_change","profit_pct_change","margin_assumption","total_units_observed"])
    def agg(g: pd.DataFrame) -> pd.Series:
        w  = g["total_units_observed"].astype(float).fillna(1.0)
        rv = g["revenue_pct_change"].astype(float).fillna(0.0)
        pf = g["profit_pct_change"].astype(float).fillna(0.0)
        wsum = float(w.sum()) if float(w.sum()) > 0 else 1.0
        return pd.Series({
            "segments_count": int(g.shape[0]),
            "vw_revenue_pct": float((rv * w).sum() / wsum),
            "vw_profit_pct":  float((pf * w).sum() / wsum),
        })
    out = d.groupby("margin_assumption", dropna=False).apply(agg, include_groups=False).reset_index()
    return out.sort_values("margin_assumption").reset_index(drop=True)

def detect_modes(df: pd.DataFrame) -> dict:
    return {"category": "Category" in df.columns and df["Category"].notna().any(),
            "sku": "SKU" in df.columns and df["SKU"].notna().any()}

def try_write_excel(tables: dict[str, pd.DataFrame], out_path: Path):
    try:
        import openpyxl  # noqa: F401
    except Exception:
        print("Note: openpyxl not available — skipping Excel export.")
        return
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print(f"Wrote Excel → {out_path}")

def optimize_mode(name, scenarios, fit, seg_cols,
                  prefer_conf, allow_low_if_units, min_profit_pct,
                  use_r2_in_score, r2_floor, topn):
    section(f"{name.upper()} — FILTER, SCORE, PICK")
    scn = scenarios.dropna(subset=seg_cols).copy()
    scn = dedupe_scenarios(scn, seg_cols)

    # If pre-merged, skip join; else attach from fit
    merged = attach_units_and_fit(scn, None if is_premerged(scn) else fit, seg_cols)

    # initial strict → auto-relax ladder
    used_mode, picks = "", pd.DataFrame()
    for stage, pref, minp in [
        ("initial",            prefer_conf, min_profit_pct),
        ("allow_low_big",      "high",      min_profit_pct),
        ("any_conf",           "any",       min_profit_pct),
        ("any_conf_looser_p",  "any",       -0.01),
    ]:
        scn_f = apply_confidence_filters(merged, seg_cols, pref, allow_low_if_units)
        picks = pick_best_moves(scn_f, seg_cols, minp, use_r2_in_score, r2_floor)
        if not picks.empty:
            used_mode = f"{stage}|{pref}|minp>{minp}"
            break

    if not picks.empty:
        print(f"Relaxation used: {used_mode}")
        show = picks.sort_values(["total_units_observed"] + seg_cols,
                                 ascending=[False] + [True]*len(seg_cols))
        print("\nTop-N moves (weighted by units × R²):")
        print(show.head(topn).to_string(index=False))
    else:
        print("No valid picks after all relaxations.")
    return picks, used_mode

def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    section("LOAD")
    scenarios = load_csv(SCENARIOS_PATH, ["scenario_price_change","revenue_pct_change"])
    fit = load_csv(FIT_PATH) if FIT_PATH.exists() else None
    if fit is not None:
        print("Fit columns detected:", fit.columns.tolist())
    else:
        print("Fit file not present; assuming scenarios are pre-merged.")

    # Normalize CELL VALUES for join keys
    for col in ("Category", "SKU"):
        if col in scenarios.columns: scenarios[col] = scenarios[col].map(norm_val)
        if fit is not None and col in fit.columns: fit[col] = fit[col].map(norm_val)

    # ---- Business sanity filters BEFORE any optimization ----
    # 1) filter out pathological positive elasticities (or keep if NaN)
    if "elasticity_beta" in scenarios.columns:
        scenarios = scenarios[(scenarios["elasticity_beta"].isna()) | (scenarios["elasticity_beta"] <= 0)]

    # 2) enforce historical price band if available
    need = {"scenario_price_new","price_min","price_max"}
    if need.issubset(scenarios.columns):
        scenarios = scenarios[
            (scenarios["scenario_price_new"] >= scenarios["price_min"]) &
            (scenarios["scenario_price_new"] <= scenarios["price_max"])
        ]

    modes = detect_modes(scenarios)
    print(f"Detected modes → Category: {modes['category']} | SKU: {modes['sku']}")

    out_cat = OUT_DIR / "recommended_moves_category.csv"
    out_sku = OUT_DIR / "recommended_moves_sku.csv"
    out_port = OUT_DIR / "portfolio_summary.csv"

    excel_tabs: dict[str, pd.DataFrame] = {}
    all_portfolio = []
    relax_notes = []

    # CATEGORY
    if modes["category"]:
        picks_cat, used_mode_cat = optimize_mode(
            "Category", scenarios, fit, ["Category"],
            INIT_PREFER_CONFIDENCE, INIT_ALLOW_LOW_IF_UNITS, INIT_MIN_PROFIT_PCT,
            USE_R2_IN_SCORE, R2_FLOOR, TOPN_PRINT
        )
        picks_cat.to_csv(out_cat, index=False)
        excel_tabs["category_moves"] = picks_cat
        relax_notes.append(("Category", used_mode_cat))
        port_cat = portfolio_rollup(picks_cat)
        if not port_cat.empty:
            port_cat["mode"] = "Category"
            all_portfolio.append(port_cat)

    # SKU
    if modes["sku"]:
        picks_sku, used_mode_sku = optimize_mode(
            "SKU", scenarios, fit, ["SKU"],
            INIT_PREFER_CONFIDENCE, INIT_ALLOW_LOW_IF_UNITS, INIT_MIN_PROFIT_PCT,
            USE_R2_IN_SCORE, R2_FLOOR, TOPN_PRINT
        )
        picks_sku.to_csv(out_sku, index=False)
        excel_tabs["sku_moves"] = picks_sku
        relax_notes.append(("SKU", used_mode_sku))
        port_sku = portfolio_rollup(picks_sku)
        if not port_sku.empty:
            port_sku["mode"] = "SKU"
            all_portfolio.append(port_sku)

    section("PORTFOLIO SUMMARY")
    if all_portfolio:
        portfolio = pd.concat(all_portfolio, ignore_index=True)
        portfolio = portfolio[["mode","margin_assumption","segments_count","vw_revenue_pct","vw_profit_pct"]]
        portfolio = portfolio.sort_values(["mode","margin_assumption"])
        portfolio.to_csv(out_port, index=False)
        print(f"Wrote → {out_port}")
        print("\nPortfolio (mode × margin):")
        print(portfolio.to_string(index=False))
        excel_tabs["portfolio_summary"] = portfolio
    else:
        print("No portfolio rows.")

    # add relaxation notes
    relax_df = pd.DataFrame(relax_notes, columns=["mode", "relaxation_used"])
    excel_tabs["relaxation_notes"] = relax_df

    if WRITE_EXCEL:
        try:
            with pd.ExcelWriter(OUT_DIR / EXCEL_NAME, engine="openpyxl") as xw:
                for name, df in excel_tabs.items():
                    df.to_excel(xw, sheet_name=name[:31], index=False)
            print(f"Wrote Excel → {OUT_DIR / EXCEL_NAME}")
        except Exception as e:
            print(f"Excel export skipped: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()