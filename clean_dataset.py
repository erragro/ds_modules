#!/usr/bin/env python3
# clean_dataset.py
# Purpose: Produce a clean, analysis-ready dataset from the raw Kaggle file.
# Scope: Deterministic cleaning with clear, commented steps. Safe defaults.

import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# ------------------------ Helpers ------------------------

def section(msg: str):
    bar = "=" * 80
    print(f"\n{bar}\n{msg}\n{bar}")

def pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first existing column from candidates (handles schema variance)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_str_series(s: pd.Series) -> pd.Series:
    """Trim whitespace, collapse internal spaces, and title/upper-case categories."""
    if s.dtype != "O":
        return s
    # strip outer spaces, collapse internal runs of whitespace to single space
    s2 = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return s2

def to_int_nullable(series: pd.Series) -> pd.Series:
    """Convert float-ish postal codes to nullable integer without .0 artifacts."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")

# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Clean the ecommerce sales dataset.")
    ap.add_argument("--path", required=True, help="Path to raw CSV file")
    ap.add_argument("--sep", default=",", help="CSV delimiter (default ,)")
    ap.add_argument("--encoding", default="utf-8", help="File encoding")
    args = ap.parse_args()

    csv_path = Path(args.path)
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = csv_path.parent / "clean_outputs"
    out_dir.mkdir(exist_ok=True)

    # ------------------------ Load ------------------------
    section("LOAD")
    df = pd.read_csv(csv_path, sep=args.sep, encoding=args.encoding, low_memory=False)
    orig_shape = df.shape
    print(f"Loaded: {orig_shape[0]:,} rows, {orig_shape[1]:,} cols")

    # ------------------------ Column aliases ------------------------
    # Map common names so script is robust to minor schema changes.
    C = {
        "order_id":  ["Order ID", "order_id", "OrderID"],
        "date":      ["Date", "Order Date", "order_date", "Purchase Date"],
        "status":    ["Status", "Order Status"],
        "fulfilment":["Fulfilment", "Fulfillment", "fulfillment_method"],
        "sales_ch":  ["Sales Channel", "SalesChannel", "Channel"],
        "service":   ["ship-service-level", "Ship Service Level"],
        "style":     ["Style", "style"],
        "sku":       ["SKU", "sku", "Sku"],
        "category":  ["Category", "category", "Product Category"],
        "size":      ["Size", "size"],
        "asin":      ["ASIN", "asin"],
        "courier":   ["Courier Status", "Courier_Status"],
        "qty":       ["Qty", "Quantity", "quantity", "Order Quantity"],
        "currency":  ["currency", "Currency"],
        "amount":    ["Amount", "amount", "Total Amount", "Order Amount", "Revenue"],
        "promo":     ["promotion-ids", "Promotion Ids"],
        "fulfilled_by": ["fulfilled-by", "Fulfilled By"],
        "index_col":["index", "Index"],
        "ship_city": ["ship-city", "Ship City"],
        "ship_state":["ship-state", "Ship State"],
        "ship_post": ["ship-postal-code", "Ship Postal Code", "Pincode"],
        "ship_ctry": ["ship-country", "Ship Country"],
        "misc":      ["Unnamed: 22", "Unnamed: 0", "Unnamed: 1"],
        "b2b":       ["B2B", "is_b2b"],
    }

    col = {k: pick(df, v) for k, v in C.items()}
    print("Detected columns:", json.dumps(col, indent=2))

    # ------------------------ Drop junk/sparse columns ------------------------
    # Based on EDA: 'misc' is a parse artifact; 'promo' & 'fulfilled_by' largely sparse;
    # 'currency' & 'ship_ctry' constant (INR/IN) -> drop for modeling.
    drop_candidates = [col["misc"], col["promo"], col["fulfilled_by"], col["currency"], col["ship_ctry"]]
    drop_cols = [c for c in drop_candidates if c and c in df.columns]
    if drop_cols:
        section("DROP JUNK/SPARSE/CONSTANT COLUMNS")
        print("Dropping:", drop_cols)
        df = df.drop(columns=drop_cols)

    # ------------------------ Normalize text fields ------------------------
    section("NORMALIZE TEXT FIELDS (trim/collapse whitespace)")
    text_like = [k for k in [col["sku"], col["category"], col["style"], col["status"],
                             col["fulfilment"], col["sales_ch"], col["service"],
                             col["ship_city"], col["ship_state"]] if k]
    for t in text_like:
        df[t] = normalize_str_series(df[t])

    # ------------------------ Parse Date ------------------------
    if col["date"]:
        section("PARSE DATE")
        # Handles patterns like 04-30-22 etc.; coerce invalid
        df[col["date"]] = pd.to_datetime(df[col["date"]], errors="coerce", infer_datetime_format=True)
        invalid_dates = df[col["date"]].isna().sum()
        print(f"Parsed dates. Invalid/NaT rows: {invalid_dates:,}")

    # ------------------------ Remove cancelled orders ------------------------
    # Business logic: Qty==0 indicates cancelled/refunded and should not flow into elasticity.
    if col["qty"]:
        section("REMOVE CANCELLED ORDERS (Qty == 0)")
        before = len(df)
        df = df[df[col["qty"]] > 0]
        print(f"Removed {before - len(df):,} cancelled rows; remaining {len(df):,}")

    # ------------------------ Ensure numeric Amount ------------------------
    if col["amount"]:
        section("SANITIZE AMOUNT")
        df[col["amount"]] = pd.to_numeric(df[col["amount"]], errors="coerce")
        amt_nulls = df[col["amount"]].isna().sum()
        print(f"Amount nulls after coercion: {amt_nulls:,}")

    # ------------------------ Derive Unit_Price safely ------------------------
    section("DERIVE Unit_Price = Amount / Qty (safe)")
    if col["amount"] and col["qty"]:
        # avoid division by zero; Qty already filtered > 0 above
        df["Unit_Price"] = df[col["amount"]] / df[col["qty"]]
    else:
        df["Unit_Price"] = pd.NA
    # enforce numeric
    df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")

    # ------------------------ Basic validity filters ------------------------
    section("VALIDITY FILTERS (drop rows missing critical fields)")
    required = [x for x in [col["sku"], col["category"], col["amount"], "Unit_Price", col["qty"]] if x]
    before_req = len(df)
    df = df.dropna(subset=required)
    print(f"Dropped {before_req - len(df):,} rows with missing criticals.")

    # Remove nonsensical monetary values (negative) or zero Unit_Price
    before_val = len(df)
    df = df[(df[col["amount"]] > 0) & (df["Unit_Price"] > 0)]
    print(f"Dropped {before_val - len(df):,} rows with non-positive Amount/Unit_Price.")

    # ------------------------ Postal code fix ------------------------
    if col["ship_post"] and col["ship_post"] in df.columns:
        section("POSTAL CODE TYPE FIX")
        df[col["ship_post"]] = to_int_nullable(df[col["ship_post"]])
        print(f"Postal code nulls after fix: {df[col['ship_post']].isna().sum():,}")

    # ------------------------ Deduplicate ------------------------
    # Conservative: If an explicit unique key exists (Order ID + SKU + Date), use it.
    section("DEDUPLICATE CONSERVATIVELY")
    key_cols = [c for c in [col["order_id"], col["sku"], col["date"]] if c]
    before_dupe = len(df)
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    print(f"Removed {before_dupe - len(df):,} duplicate rows.")

    # ------------------------ Curated column ordering ------------------------
    section("REORDER/RENAME OUTPUT COLUMNS (for readability)")
    nice_cols = []
    for k in ["order_id", "date", "status", "fulfilment", "sales_ch", "service",
              "sku", "style", "size", "asin", "category", "qty", "amount",
              "Unit_Price", "ship_city", "ship_state", "ship_post"]:
        cname = col.get(k) if k != "Unit_Price" else "Unit_Price"
        if cname and cname in df.columns and cname not in nice_cols:
            nice_cols.append(cname)
    # keep anything else at the end
    rest = [c for c in df.columns if c not in nice_cols]
    df = df[nice_cols + rest]

    # ------------------------ Quick post-clean checks ------------------------
    section("POST-CLEAN CHECKS")
    print("Shape:", df.shape)
    print("Sample rows:")
    with pd.option_context("display.max_columns", 120):
        print(df.head(5).to_string(index=False))
    print("\nNulls in key fields:")
    print(df[[c for c in ["Unit_Price", col["amount"], col["qty"], col["category"], col["sku"]] if c]].isna().sum())

    # ------------------------ Outputs ------------------------
    section("WRITE OUTPUTS")
    cleaned_csv = out_dir / "ecommerce_sales_cleaned.csv"
    cleaned_parquet = out_dir / "ecommerce_sales_cleaned.parquet"
    df.to_csv(cleaned_csv, index=False)
    try:
        df.to_parquet(cleaned_parquet, index=False)
        wrote_parquet = True
    except Exception as e:
        print(f"Parquet write failed (install pyarrow or fastparquet to enable). Error: {e}")
        wrote_parquet = False

    # Cleaning report (for your MD)
    report = {
        "input_file": str(csv_path.name),
        "input_shape": {"rows": orig_shape[0], "cols": orig_shape[1]},
        "output_shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "dropped_columns": drop_cols,
        "date_invalid_rows": int(df[col["date"]].isna().sum()) if col["date"] else None,
        "cancelled_orders_removed": None,  # computed earlier but not stored; optional to add
        "critical_nulls_dropped": None,
        "non_positive_rows_removed": None,
        "postal_nulls": int(df[col["ship_post"]].isna().sum()) if col["ship_post"] else None,
        "outputs": {
            "csv": str(cleaned_csv),
            "parquet": str(cleaned_parquet) if wrote_parquet else None
        }
    }
    with open(out_dir / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved cleaned CSV → {cleaned_csv}")
    if wrote_parquet:
        print(f"Saved cleaned Parquet → {cleaned_parquet}")
    print(f"Saved cleaning report → {out_dir / 'cleaning_report.json'}")
    print("\nDone.")

if __name__ == "__main__":
    main()