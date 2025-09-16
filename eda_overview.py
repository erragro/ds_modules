#!/usr/bin/env python3
# eda_overview.py
# Purpose: First-pass, read-only EDA to understand dataset structure & health.
# Notes:
#  - Includes column alias detection and derives Unit Price if no explicit Price.
#  - No cleaning or modeling performed—observation only.
#  - Explanations inline (why each report exists per EDA best practices).

import argparse
from pathlib import Path
import sys
import pandas as pd

# ---------- Utilities ----------
def human_bytes(n: int) -> str:
    """Human-friendly memory size display."""
    for unit in ["B","KB","MB","GB","TB","PB","EB"]:
        if n < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}ZB"

def section(title: str):
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")

def pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first column name present from candidates, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None

def safe_has(df: pd.DataFrame, cols) -> bool:
    return all(c in df.columns for c in cols)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Read-only EDA with explanations + column aliasing."
    )
    parser.add_argument("--path", required=True, help="Path to CSV file")
    parser.add_argument("--sample", type=int, default=0,
                        help="Read only first N rows (0 = all)")
    parser.add_argument("--sep", default=",", help="CSV delimiter (default ,)")
    parser.add_argument("--encoding", default="utf-8", help="File encoding")
    args = parser.parse_args()

    csv_path = Path(args.path)
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # 0) LOAD — controlled, reproducible read; --sample allows quick dry runs
    nrows = None if args.sample == 0 else args.sample
    # low_memory=False -> better type inference pass
    df = pd.read_csv(csv_path, nrows=nrows, sep=args.sep,
                     encoding=args.encoding, low_memory=False)

    out_dir = csv_path.parent / "eda_outputs"
    out_dir.mkdir(exist_ok=True)

    # Column alias resolution — handle common schema variations
    CANDIDATES = {
        "sku": ["SKU", "sku", "Sku", "ItemSKU"],
        "category": ["Category", "category", "Product Category"],
        "qty": ["Qty", "Quantity", "quantity", "Order Quantity"],
        "amount": ["Amount", "amount", "Total Amount", "Order Amount", "Revenue"],
        "price": ["Price", "Unit Price", "Unit_Price", "Selling Price", "selling_price"],
        "date": ["Date", "Order Date", "order_date", "Purchase Date"]
    }

    SKU_COL = pick(df, CANDIDATES["sku"])
    CAT_COL = pick(df, CANDIDATES["category"])
    QTY_COL = pick(df, CANDIDATES["qty"])
    AMT_COL = pick(df, CANDIDATES["amount"])
    PRICE_COL = pick(df, CANDIDATES["price"])
    DATE_COL = pick(df, CANDIDATES["date"])

    # Derive Unit Price if needed and possible (Amount / Qty for Qty>0)
    derived_price_col = None
    if PRICE_COL is None and AMT_COL and QTY_COL:
        df = df.copy()
        derived_price_col = "__Unit_Price__"
        df[derived_price_col] = pd.Series(pd.NA, index=df.index)
        mask = df[QTY_COL] > 0
        # Protect against null Amounts
        df.loc[mask, derived_price_col] = df.loc[mask, AMT_COL] / df.loc[mask, QTY_COL]
        PRICE_COL = derived_price_col

    print("\n[Column detection]")
    print(f"SKU: {SKU_COL} | Category: {CAT_COL} | Qty: {QTY_COL} | "
          f"Amount: {AMT_COL} | Price/Unit_Price: {PRICE_COL} | Date: {DATE_COL}")

    # 1) SHAPE & MEMORY — feasibility check; switch to Spark only if huge
    section("1) BASIC SHAPE & MEMORY (sanity check)")
    rows, cols = df.shape
    mem = df.memory_usage(index=True, deep=True).sum()
    print(f"Rows: {rows:,} | Columns: {cols:,}")
    print(f"In-memory size (approx): {human_bytes(mem)}")

    # 2) DTYPES — correctness of numeric/categorical/datetime is foundational
    section("2) DATA TYPES (ensure correct numeric/categorical/datetime)")
    print(df.dtypes.sort_index())

    # 3) HEAD — quick “eyeball” parse/structure validation
    section("3) HEAD (first 5 rows) — eyeball parsing/structure quickly")
    with pd.option_context("display.max_columns", 200):
        print(df.head(5).to_string(index=False))

    # 4) NULL COUNTS — quantify missingness to plan imputations/drops later
    section("4) NULL COUNTS (top 20 by nulls) — plan imputations/drops later")
    nulls = df.isnull().sum().sort_values(ascending=False)
    print(nulls.head(20).to_string())
    nulls.to_csv(out_dir / "null_counts.csv", header=["null_count"])

    # 5) UNIQUE COUNTS — cardinality guides IDs, categories, segmentation
    section("5) UNIQUE COUNTS PER COLUMN — identify IDs vs categories")
    nunique = df.nunique(dropna=True).sort_values(ascending=False)
    print(nunique.to_string())
    nunique.to_csv(out_dir / "unique_counts.csv", header=["unique_values"])

    # 6) NUMERIC SUMMARY — detect outliers / impossible values
    section("6) NUMERIC SUMMARY (describe) — outliers/scale sanity")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().transpose()
        print(desc.to_string())
        desc.to_csv(out_dir / "numeric_summary.csv")
    else:
        print("No numeric columns detected.")

    # 7) DOMAIN SANITY PIVOTS — align with pricing/elasticity prep
    section("7) DOMAIN SANITY PIVOTS — align with pricing/elasticity questions")

    # 7a) SKU-level quick stats (txn_count, unique price points)
    if SKU_COL and PRICE_COL:
        sku_price_points = (
            df.groupby(SKU_COL)[PRICE_COL]
            .nunique(dropna=True)
            .rename("unique_price_points")
        )
        sku_counts = df.groupby(SKU_COL).size().rename("txn_count")
        sku_quick = (
            pd.concat([sku_counts, sku_price_points], axis=1)
            .sort_values("txn_count", ascending=False)
        )
        print("\nTop 15 SKUs by transaction count (with unique price points):")
        print(sku_quick.head(15).to_string())
        sku_quick.to_csv(out_dir / "sku_quick_stats.csv")
    else:
        print("SKU/Price not both present after alias resolution; skipping SKU-level pivot.")

    # 7b) Category-level price summary (mean/median/count)
    if CAT_COL and PRICE_COL:
        cat_sum = (
            df.groupby(CAT_COL)[PRICE_COL]
              .agg(["count", "mean", "median"])
              .sort_values("count", ascending=False)
        )
        print("\nTop 15 Categories by row count with price stats:")
        print(cat_sum.head(15).to_string())
        cat_sum.to_csv(out_dir / "category_price_summary.csv")
    else:
        print("Category/Price not both present after alias resolution; skipping category price summary.")

    # 8) Column list artifact — quick reference for downstream scripts
    (out_dir / "columns.csv").write_text(",".join(df.columns) + "\n", encoding="utf-8")

    section("ARTIFACTS")
    print(f"Saved to: {out_dir.resolve()}")
    if derived_price_col:
        print(f"Note: Derived unit price column written in-memory as '{derived_price_col}' for diagnostics only.")
    print("Done. Observation-only EDA complete.")

if __name__ == "__main__":
    main()