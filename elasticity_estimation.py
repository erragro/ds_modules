#!/usr/bin/env python3
# elasticity_estimation.py (multi-margin; keep-all; confidence-tiered)

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# -------------------- Utils --------------------

def section(msg: str):
    bar = "=" * 80
    print(f"\n{bar}\n{msg}\n{bar}")

def safe_log(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0)
    return np.log(s)

def ols_slope_r2(x: np.ndarray, y: np.ndarray):
    """Fit y = a + b x via OLS; return (b, a, r2, n)."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta[0], beta[1]
    y_hat = a + b * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return b, a, r2, len(y)

def aggregate_price_points(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """(segment..., Unit_Price) → Qty_at_price."""
    return (df.groupby(group_cols + ["Unit_Price"], dropna=False)["Qty"]
              .sum()
              .rename("Qty_at_price")
              .reset_index())

def fit_elasticity(agg_df: pd.DataFrame, id_cols: list[str],
                   min_points: int, min_prices: int) -> pd.DataFrame:
    """Fit ln(Q)=α+β ln(P) per segment; return β, R², counts, price bounds."""
    rows = []
    for key, sub in agg_df.groupby(id_cols):
        up = sub["Unit_Price"].dropna().unique()
        if len(up) < min_prices:
            continue

        lnP = safe_log(sub["Unit_Price"])
        lnQ = safe_log(sub["Qty_at_price"])
        mask = lnP.notna() & lnQ.notna()
        x, y = lnP[mask].to_numpy(), lnQ[mask].to_numpy()
        if len(x) < min_points:
            continue

        beta, alpha, r2, n = ols_slope_r2(x, y)

        p_min = float(np.nanmin(sub["Unit_Price"]))
        p_max = float(np.nanmax(sub["Unit_Price"]))
        p_med = float(np.nanmedian(sub["Unit_Price"]))
        total_units = int(np.nansum(sub["Qty_at_price"]))
        uniq_prices = int(sub["Unit_Price"].nunique())

        row_key = list(key) if isinstance(key, tuple) else [key]
        rows.append(row_key + [beta, r2, n, uniq_prices, total_units, p_min, p_med, p_max])

    cols = id_cols + [
        "elasticity_beta","r2","n_points","unique_prices","total_units_observed",
        "price_min","price_median","price_max"
    ]
    return pd.DataFrame(rows, columns=cols)

def scenario_deltas_from_range(p_min: float, p_med: float, p_max: float,
                               max_abs_dp: float = 0.30) -> list[float]:
    """
    Targets: [p_min, 0.9*median, median, 1.1*median, p_max]
    Δp = (target/median) - 1, clipped to ±max_abs_dp; ensure ±5% exists.
    """
    targets = []
    if p_med and p_med > 0:
        targets = [p_min, 0.9 * p_med, p_med, 1.1 * p_med, p_max]
    deltas = []
    for t in targets:
        if t and p_med and p_med > 0:
            dp = (t / p_med) - 1.0
            deltas.append(dp)
    out = []
    for dp in deltas:
        dp = float(np.clip(dp, -max_abs_dp, max_abs_dp))
        if all(abs(dp - d) > 1e-9 for d in out):
            out.append(dp)
    for dp in (-0.05, 0.05):
        if all(abs(dp - d) > 1e-9 for d in out):
            out.append(dp)
    return sorted(out)

def parse_list_of_floats(value, default=None) -> list[float]:
    """Accepts comma/space-separated string or list/tuple; returns list[float]."""
    if value is None:
        return [] if default is None else default
    if isinstance(value, (list, tuple)):
        vals = []
        for v in value:
            try:
                vals.append(float(v))
            except:
                pass
        return vals
    s = str(value).strip()
    if not s:
        return [] if default is None else default
    parts = []
    for tok in s.replace(",", " ").split():
        try:
            parts.append(float(tok))
        except:
            pass
    return parts if parts else ([] if default is None else default)

def simulate_rows_for_margins(seg_row: pd.Series,
                              deltas: list[float],
                              margins: list[float]) -> list[dict]:
    """
    Emit one scenario row per Δp per margin (long format).
    Revenue:  (1 + Δp)^(β + 1)
    Profit :  ((m + Δp)/m) * (1 + Δp)^β; invalid if m+Δp<=0
    """
    beta = float(seg_row["elasticity_beta"])
    pmin = float(seg_row["price_min"])
    pmed = float(seg_row["price_median"])
    pmax = float(seg_row["price_max"])

    out = []
    for dp in deltas:
        p_new = pmed * (1.0 + dp) if pmed and pmed > 0 else np.nan
        in_range = bool((not np.isnan(p_new)) and (p_new >= pmin - 1e-9) and (p_new <= pmax + 1e-9))
        rev_mult = (1.0 + dp) ** (beta + 1.0)

        if not margins:
            # still emit revenue-only row
            out.append({
                **{k: seg_row[k] for k in seg_row.index},
                "scenario_price_change": float(dp),
                "scenario_price_new": float(p_new) if not np.isnan(p_new) else np.nan,
                "in_hist_range": in_range,
                "revenue_multiplier": float(rev_mult),
                "revenue_pct_change": float((rev_mult - 1.0) * 100.0),
                "margin_assumption": np.nan,
                "profit_multiplier": np.nan,
                "profit_pct_change": np.nan,
                "profit_flag": "no_margin_provided"
            })
            continue

        for m in margins:
            if m is None:
                continue
            m = float(m)
            row = {
                **{k: seg_row[k] for k in seg_row.index},
                "scenario_price_change": float(dp),
                "scenario_price_new": float(p_new) if not np.isnan(p_new) else np.nan,
                "in_hist_range": in_range,
                "revenue_multiplier": float(rev_mult),
                "revenue_pct_change": float((rev_mult - 1.0) * 100.0),
                "margin_assumption": m
            }
            if (m + dp) <= 0:
                row["profit_multiplier"] = np.nan
                row["profit_pct_change"] = np.nan
                row["profit_flag"] = "invalid_margin_at_dp"
            else:
                prof_mult = ((m + dp) / m) * ((1.0 + dp) ** beta)
                row["profit_multiplier"] = float(prof_mult)
                row["profit_pct_change"] = float((prof_mult - 1.0) * 100.0)
                row["profit_flag"] = ""
            out.append(row)
    return out

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(
        description="Elasticity with data-driven & business scenarios; multi-margin profit; keep-all with confidence tiers."
    )
    ap.add_argument("--path", required=True, help="Path to cleaned CSV (from Step 2)")
    ap.add_argument("--segment_by", default="Category",
                    help="Comma-separated segment cols (e.g., 'Category' or 'Category,Sales Channel ')")
    ap.add_argument("--min_points", type=int, default=5, help="Min (price,demand) points per segment")
    ap.add_argument("--min_prices", type=int, default=3, help="Min unique price points per segment")
    ap.add_argument("--max_abs_dp", type=float, default=0.30, help="Max |Δp| from data-driven range")
    ap.add_argument("--business_scenarios", default="-0.20,-0.10,0.05,0.10",
                    help="Comma/space-separated Δp values (default: -20%,-10%,+5%,+10%)")
    # NEW: multiple margins
    ap.add_argument("--assumed_margins", default="0.10,0.20,0.30",
                    help="Comma/space-separated margin assumptions (default: 0.10,0.20,0.30)")

    # Deprecated arg for compatibility; ignored (no filtering).
    ap.add_argument("--r2_min", type=float, default=0.0,
                    help="(Deprecated) R² cutoff; script keeps all segments and tags confidence.")
    args = ap.parse_args()

    csv_path = Path(args.path)
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = csv_path.parent / "out_refined"
    out_dir.mkdir(exist_ok=True)

    # Load
    section("LOAD CLEANED DATA")
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    required = ["Unit_Price", "Qty"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: required column missing: {col}", file=sys.stderr)
            sys.exit(1)

    seg_cols = [c.strip() for c in str(args.segment_by).split(",") if c.strip()]
    for c in seg_cols:
        if c not in df.columns:
            print(f"ERROR: segment column missing: {c}", file=sys.stderr)
            sys.exit(1)
    print("Loaded:", df.shape, "| Segmentation:", seg_cols)

    # Aggregate & fit
    section("AGGREGATE PRICE POINTS & FIT ELASTICITY")
    agg = aggregate_price_points(df, seg_cols)
    fit = fit_elasticity(agg, seg_cols, args.min_points, args.min_prices)

    # Interpretation + fit tiers (no filtering)
    def interp(beta: float) -> str:
        if pd.isna(beta):
            return "Insufficient data"
        b = float(beta)
        if b <= -1.2:
            return "Highly elastic — discount drives volume"
        if b <= -0.8:
            return "Moderately elastic — careful pricing"
        if b <= -0.3:
            return "Relatively inelastic — margin increase possible"
        if b < 0:
            return "Inelastic — strong pricing power"
        return "Positive slope — premium effect or noise"

    def fit_conf_tier(r2: float) -> str:
        r2 = 0.0 if pd.isna(r2) else float(r2)
        return "high_fit" if r2 >= 0.05 else "low_fit"

    if fit.empty:
        print("No segments had sufficient price variation/points. Writing empty outputs and exiting.")
        fit_cols = seg_cols + [
            "elasticity_beta","r2","n_points","unique_prices","total_units_observed",
            "price_min","price_median","price_max","business_interpretation","fit_confidence"
        ]
        (out_dir / "elasticity_segments.csv").write_text(",".join(fit_cols) + "\n")
        scn_cols = seg_cols + [
            "elasticity_beta","r2","price_min","price_median","price_max",
            "scenario_price_change","scenario_price_new","in_hist_range",
            "revenue_multiplier","revenue_pct_change",
            "margin_assumption","profit_multiplier","profit_pct_change","profit_flag","confidence"
        ]
        (out_dir / "scenarios_refined.csv").write_text(",".join(scn_cols) + "\n")
        return

    fit["business_interpretation"] = fit["elasticity_beta"].apply(interp)
    fit["fit_confidence"] = fit["r2"].apply(fit_conf_tier)

    fit_path = out_dir / "elasticity_segments.csv"
    fit.to_csv(fit_path, index=False)
    print(f"Wrote: {fit_path}")

    # Scenarios (multi-margin)
    section("BUILD SCENARIOS (data-driven + business; multi-margin)")
    extra_dps = parse_list_of_floats(args.business_scenarios, default=[-0.20, -0.10, 0.05, 0.10])
    margins = parse_list_of_floats(args.assumed_margins, default=[0.10, 0.20, 0.30])

    all_rows = []
    for _, r in fit.iterrows():
        auto_dps = scenario_deltas_from_range(
            r["price_min"], r["price_median"], r["price_max"], max_abs_dp=args.max_abs_dp
        )
        deltas = sorted(set(auto_dps + extra_dps))
        all_rows.extend(simulate_rows_for_margins(r, deltas, margins))

    scn_path = out_dir / "scenarios_refined.csv"
    if not all_rows:
        print("No scenarios generated. Writing empty scenarios and exiting.")
        scn_cols = seg_cols + [
            "elasticity_beta","r2","price_min","price_median","price_max",
            "scenario_price_change","scenario_price_new","in_hist_range",
            "revenue_multiplier","revenue_pct_change",
            "margin_assumption","profit_multiplier","profit_pct_change","profit_flag","confidence"
        ]
        (out_dir / "scenarios_refined.csv").write_text(",".join(scn_cols) + "\n")
        return

    scenarios = pd.DataFrame(all_rows)

    # Rounding
    for c in ["elasticity_beta","r2","price_min","price_median","price_max",
              "scenario_price_change","scenario_price_new",
              "revenue_multiplier","revenue_pct_change",
              "margin_assumption","profit_multiplier","profit_pct_change"]:
        if c in scenarios.columns:
            scenarios[c] = scenarios[c].astype(float).round(6)

    # Scenario confidence (fit tier + in-range)
    def scenario_conf(in_range: bool, r2: float) -> str:
        high_fit = (0.0 if pd.isna(r2) else float(r2)) >= 0.05
        if in_range and high_fit:
            return "in_range_high_conf"
        if in_range and not high_fit:
            return "in_range_low_conf"
        if (not in_range) and high_fit:
            return "extrapolated_high_conf"
        return "extrapolated_low_conf"

    scenarios["confidence"] = [
        scenario_conf(ir, r) for ir, r in zip(scenarios["in_hist_range"], scenarios["r2"])
    ]

    scenarios.to_csv(scn_path, index=False)
    print(f"Wrote: {scn_path}")

    # Exec summary (±10% rows if exist, prefer margin 0.20)
    section("EXEC SUMMARY (console)")
    prefer_margin = 0.20

    if scenarios.empty:
        print("Scenarios table is empty. See scenarios_refined.csv for details.")
    else:
        key = scenarios[
            (scenarios["scenario_price_change"].isin([-0.10, 0.10])) &
            (np.isclose(scenarios.get("margin_assumption", np.nan), prefer_margin))
        ].copy()

        if key.empty:
            print("No ±10% scenarios at margin 0.20. See scenarios_refined.csv for the full grid.")
        else:
            # Attach total_units_observed & fit_confidence safely
            cols_attach = seg_cols + ["total_units_observed", "fit_confidence"]
            attach = fit[cols_attach].drop_duplicates() if all(c in fit.columns for c in cols_attach) else pd.DataFrame()

            if not attach.empty:
                key = key.merge(attach, on=seg_cols, how="left", validate="m:1")
            else:
                # Fallback mapping if needed
                if "total_units_observed" not in key.columns and "total_units_observed" in fit.columns:
                    mapping = fit.drop_duplicates(subset=seg_cols).set_index(seg_cols)["total_units_observed"].to_dict()
                    key["total_units_observed"] = [mapping.get(tuple(row[seg_cols]), np.nan) for _, row in key.iterrows()]
                if "fit_confidence" not in key.columns and "fit_confidence" in fit.columns:
                    mapping_fc = fit.drop_duplicates(subset=seg_cols).set_index(seg_cols)["fit_confidence"].to_dict()
                    key["fit_confidence"] = [mapping_fc.get(tuple(row[seg_cols]), "") for _, row in key.iterrows()]

            # Choose safe sort keys
            if "total_units_observed" in key.columns:
                sort_by = ["total_units_observed"] + seg_cols
                asc = [False] + [True] * len(seg_cols)
            else:
                sort_by = seg_cols
                asc = [True] * len(seg_cols)

            key = key.sort_values(sort_by, ascending=asc)

            # Columns to print (only those that exist)
            cols_show = []
            for c in (seg_cols + ["elasticity_beta","r2","scenario_price_change",
                                  "revenue_pct_change","profit_pct_change",
                                  "in_hist_range","confidence","fit_confidence",
                                  "margin_assumption","total_units_observed"]):
                if c in key.columns:
                    cols_show.append(c)

            print(key[cols_show].head(20).to_string(index=False))

if __name__ == "__main__":
    main()