# Dynamic Pricing Analysis for a Medical E-Commerce Platform

### Submission Artifact
Comprehensive **README** for the Data Scientist Interview Challenge  

---

## Scope Covered
- **Data Preprocessing & Feature Engineering**  
- **Demand Curve Estimation**  
- **Dynamic Pricing Simulation**  
- **Segmentation**  
- **Evaluation & Scenario Testing**  
- **Optimization**  
- **Packaging & Next Steps**

## Executive Summary
	•	We built an end‑to‑end, reproducible pricing analysis pipeline that ingests sales data (~129k rows), cleans/standardizes it (~113.7k rows retained), estimates price elasticities at Category and SKU levels, simulates price‑change scenarios, and recommends guardrailed price moves with a portfolio roll‑up of expected impact.
	•	Findings indicate the highest, most reliable leverage sits in the Set and Kurta categories, which behave inelastically. After strict sanity checks (historical price bounds, fit thresholds, Δp caps), the recommender favors a +5–10% price increase pilot in these categories.
	•	Estimated portfolio impact at +10% price: ~+6% revenue and ~+28–93% profit (depending on assumed gross margin). Caveat: model fits (R²) are low; treat magnitudes as directional; validate via A/B tests with rollback triggers.


## Repository Layout & Reproducibility

| Path / File | Description |
|-------------|-------------|
| **ds_modules/** | Core analysis scripts |
| ├─ `eda_overview.py` | Lightweight sanity EDA |
| ├─ `clean_dataset.py` | Canonical cleaning & standardization |
| ├─ `elasticity_estimation.py` | Step 3: elasticities + scenarios (refined) |
| ├─ `merge_scenarios.py` | Optional: pre-merge scenarios + fits |
| ├─ `pricing_optimization.py` | Step 4: guarded move selection + portfolio roll-up |
| └─ `utils/` | Small helper functions (if needed) |
| **clean_outputs/** | Generated outputs |
| ├─ `ecommerce_sales_cleaned.csv` | Standardized dataset (Step 2 output) |
| └─ **out_refined/** | Refined outputs (Step 3–4) |
| &nbsp;&nbsp; ├─ `elasticity_segments.csv` | Per-segment β, R², volumes (Step 3) |
| &nbsp;&nbsp; ├─ `scenarios_refined.csv` | Δp × margin simulations (Step 3) |
| &nbsp;&nbsp; ├─ `scenarios_merged.csv` | Merged for optimizer (optional) |
| &nbsp;&nbsp; └─ **out_opt/** | Optimizer outputs (Step 4) |
| &nbsp;&nbsp;&nbsp;&nbsp; ├─ `recommended_moves_category.csv` | Recommended moves at category level |
| &nbsp;&nbsp;&nbsp;&nbsp; ├─ `recommended_moves_sku.csv` | Recommended moves at SKU level (may be empty if not applicable) |
| &nbsp;&nbsp;&nbsp;&nbsp; ├─ `portfolio_summary.csv` | Volume-weighted portfolio roll-up |
| &nbsp;&nbsp;&nbsp;&nbsp; └─ `pricing_recommendations.xlsx` | Combined Excel (category, SKU, portfolio) |
| **README.md** | This document |
| **requirements.txt** | Pinned package versions for reproducibility |

Determinism & Idempotency
All scripts write to clean_outputs/… without overwriting raw data. Re‑running steps with the same inputs produces the same artifacts.

Environment
	•	Python ≥ 3.10
	•	pandas, numpy, scipy/statsmodels (or sklearn), openpyxl (Excel export), pyarrow (optional Parquet)

## Quick Start

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
# Step 1: EDA (sanity)
```bash
python ds_modules/eda_overview.py --path raw/ecommerce_sales.csv
```
# Step 2: Clean
```bash
python ds_modules/clean_dataset.py --path raw/ecommerce_sales.csv \
  
```
# Step 3: Elasticities + scenarios (refined)
```bash
python ds_modules/elasticity_estimation.py --path raw/clean_outputs/ecommerce_sales_cleaned.csv 
```
# Pre-merge scenarios with fits for Step 4
```bash
python ds_modules/merge_scenarios.py 
```
# Step 4: Guarded move selection + portfolio roll-up (consumes merged if present)
```bash
python ds_modules/pricing_optimization.py
```

⸻

## 2) Business Problem & Design Principles

### Business Question  
Are we leaving money on the table by not adjusting prices dynamically?  
If so, which products/categories can bear **price increases (+Δp)** without hurting units,  
and where do **discounts (−Δp)** actually destroy profit?

---

### Design Principles  

1. **Explainability first**  
   - Simple, transparent models over black-boxes.  
   - A pricing decision must have a narrative:  
     *“This category is inelastic; a small price increase raises profit.”*

2. **Guardrails everywhere**  
   - Only recommend within observed price ranges.  
   - Cap absolute price changes (|Δp|).  
   - Penalize low-fit regressions.  
   - Require positive profit uplift before recommending.  

3. **Measure twice, cut once**  
   - Use scenario simulations.  
   - Roll up to a portfolio view to balance local vs. global effects.  

4. **Iterate quickly**  
   - Modular scripts.  
   - Each step produces auditable artifacts for review.
  
## 3) Data Understanding & EDA (Step 1)

### Raw Data Size  
- **Initial**: ~128,975 rows × 24 columns  
- **After cleaning**: 113,698 rows × 20 columns  

---

### Key Columns (post-standardization)
- **Identifiers**: `Order ID`, `SKU`, `ASIN`, `Style`  
- **Attributes**: `Category`, `Size`, `Fulfilment`, `Sales Channel`, `ship-service-level`  
- **Geography**: `ship-city`, `ship-state`, `ship-postal-code`  
- **Commerce**: `Qty`, `Amount`, `Unit_Price (derived)`, `Date`, `Status`  

---

### Initial Observations (Sanity Checks)
- `Qty` centered at **1**, but ranges up to **15** (bulk orders rare).  
- `Amount` median ≈ **₹605**; long tail up to **₹5,584**.  
- `Status` includes *Cancelled*; many rows had `Qty = 0` → **removed**.  
- Columns such as `promotion-ids`, `fulfilled-by`, `currency`, and `Unnamed: 22` were **sparse or constant** → **dropped**.  

---

### Column Count Anomaly (24 → 25)
- Occasionally during diagnostics, we added a temporary helper column (`__Unit_Price__`) to compare computed vs. provided price.  
- This explains why EDA sometimes showed **25 columns**.  
- Final cleaned dataset has **20 stable columns**.  

---

### Top Categories (by count & median price)
- **Set**: ~45k rows, median price ≈ ₹788 (largest share)  
- **Kurta**: ~44k rows, median price ≈ ₹435  
- **Western Dress**: ~13.9k rows, median price ≈ ₹744  
- **Top**: ~9.8k rows, median price ≈ ₹522  

---

### Early Red Flags
- Many segments had **too few unique price points**, making elasticity estimation weak.  
- Inconsistent category naming/spaces → standardized to **lowercase and trimmed**.
  

## 4) Cleaning & Standardization (Step 2)

### Goals  
The cleaning pipeline was designed to:  
- Remove cancellations and irrelevant rows.  
- Normalize datatypes for consistency.  
- Derive usable economic variables (e.g., `Unit_Price`).  
- Preserve only decision-relevant columns for elasticity modeling.  

---

### Operations (Deterministic Rules)
1. **Drop sparse/constant columns**: `Unnamed: 22`, `promotion-ids`, `fulfilled-by`, `currency`, `ship-country`.  
2. **Normalize text**: trimmed whitespace, lowercased categories/SKUs, collapsed duplicate spacing.  
3. **Parse dates**: converted `Date` → pandas datetime (`YYYY-MM-DD`).  
4. **Remove cancellations**: dropped rows with `Qty == 0` (~12,807 rows removed).  
5. **Coerce numerics**: ensured `Amount` is numeric; quantified nulls after coercion.  
6. **Derive `Unit_Price`**: computed as `Amount / Qty`; guarded against division by zero; retained only positive values.  
7. **Validity filters**: removed rows missing `Amount`, `Qty`, `Unit_Price`, `Category`, or `SKU`; dropped non-positive amounts.  
8. **Postal codes**: cast to string to retain leading zeros; quantified remaining nulls.  
9. **Deduplication (conservative)**: removed exact duplicates only (3 rows found).  
10. **Column order**: re-ordered for readability; outputs written to CSV and JSON report.  

---

### Outputs
- `clean_outputs/ecommerce_sales_cleaned.csv` → canonical cleaned dataset (113,698 rows × 20 columns).  
- `clean_outputs/cleaning_report.json` → structured metadata (row counts, nulls, drops, sample records).  

---

### Why CSV Over Parquet?  
- **CSV** is universally portable and simple for interview review.  
- **Parquet** is supported (if `pyarrow` or `fastparquet` is installed) and should be used at scale for:  
  - Faster I/O.  
  - Lower storage footprint.  
  - Richer schema preservation.
  - 
## 5) Demand Curve Estimation (Step 3)

### Objective
Estimate how **units sold respond to price changes** — measured as the **price elasticity β** — at both **Category** and **SKU** aggregation levels.  

---

### Econometric Form
We use a **log–log regression model**:  

\[
\log(Q) = \alpha + \beta \,\log(P) + \varepsilon
\]

- **β** is elasticity: a 1% change in price → β% change in units sold.  
- β < 0 → demand falls as price rises (*normal goods*).  
- |β| < 1 → **inelastic** demand (units less sensitive to price).  
- |β| > 1 → **elastic** demand (units highly sensitive to price).  

---

### Identification Constraints
To ensure reliable elasticity estimates, we require:  
- **Minimum observations per segment**: `min_points = 5`.  
- **Minimum unique price points**: `min_prices = 3`.  
- **Minimum fit quality (R² floor)**: configurable (default `0.00`); R² is reported and later used as a weighting factor.  

⚠️ Note: R² is often **low (<0.1)** in this dataset because:  
- Limited variation in prices.  
- Missing demand drivers (e.g., seasonality, promotions, competition).  

---

### Implementation Notes
- Aggregated data at **(segment, price)** level to avoid duplicated price points skewing regression fits.  
- Computed for each segment:  
  - Elasticity coefficient (β)  
  - Fit quality (R²)  
  - Number of observations (n_points)  
  - Number of unique prices (unique_prices)  
  - Total units sold (total_units_observed)  

- **Category-level**: generally more stable estimates (more data).  
- **SKU-level**: noisier but can reveal pockets of opportunity or risk.  

---

### Illustrative Outcomes (Dataset Examples)
- **Set**: β ≈ −0.18 → very inelastic → safe to increase price.  
- **Kurta**: β ≈ −0.61 → relatively inelastic.  
- **Western Dress**: β ≈ +1.18 → elastic/anomalous (price ↑ linked to higher units; may reflect promotions, so treat as low confidence).  
- **R² values**: frequently ≪ 0.1 → transparency: flagged as low confidence, not discarded outright.  

---

### Outputs
- `clean_outputs/out_refined/elasticity_segments.csv`  
  - One row per segment.  
  - Includes β, R², total_units_observed, price_min/median/max, and a human-readable **business_interpretation**.  

---

### Example CLI Command
```bash
python ds_modules/elasticity_estimation.py \
  --path clean_outputs/ecommerce_sales_cleaned.csv \
  --segment_by Category \
  --min_points 5 --min_prices 3 --r2_min 0.00 \
  --business_scenarios "-0.20,-0.10,0.05,0.10" \
  --assumed_margin 0.30
```

## 6) Dynamic Pricing Simulation (Step 3, continued)

### Purpose
Translate elasticity estimates (β) into **revenue and profit forecasts** under candidate price changes.

---

### Scenario Grid
Δp ∈ {−20%, −10%, +5%, +10%}  

**Rationale:**
- Covers **promotions** (−20%, −10%) and **gentle price increases** (+5%, +10%) commonly used in B2B/retail.  
- Avoids unrealistic extrapolation — larger moves (> ±20%) are often outside the range of historical data.  

---

### Core Formulas
Given baseline **Price (P)**, **Units (Q)**, and **Elasticity (β):**

- **Units ratio:**  
  \[
  \frac{Q'}{Q} = (1 + \Delta p)^{\beta}
  \]  

- **Revenue ratio:**  
  \[
  \frac{R'}{R} = (1 + \Delta p)^{\beta+1}
  \]  

- **Profit ratio (assuming gross margin m):**  
  \[
  \frac{\Pi'}{\Pi} = \left(\frac{R'}{R}\right) \times m
  \]  

In practice, we report **profit% changes** under **margin bands (10%, 20%, 30%)** to reflect typical retail/wholesale ranges.  

---

### Guardrails at Simulation Time
1. **Historical price band check**  
   - Simulated new prices must lie within each segment’s observed range:  
     \[
     [\text{price\_min}, \text{price\_max}]
     \]  

2. **Elasticity sanity**  
   - Drop β > 0 (since most goods should have β < 0).  
   - If kept (e.g., promotion-driven anomalies), mark as **low confidence**.  

3. **Volume weighting**  
   - Later portfolio roll-ups are weighted by `total_units_observed` to reflect business impact.  

---

### Outputs
- **`clean_outputs/out_refined/scenarios_refined.csv`**  
  - Long format table: [segment keys] × Δp × margin → revenue% and profit% changes.  
  - Includes **confidence tags**: `in_hist_range`, `confidence`.  

- **(Optional)** `scenarios_merged.csv`  
  - Pre-joined with elasticity fits for one-shot consumption by the optimizer.
    
## 7) Optimization & Guardrails (Step 4)

### Goal  
Transform simulated scenarios into **business-ready pricing moves** with rigorous **risk controls**.

---

### Inputs
- **Preferred:** `scenarios_merged.csv`  
- **Alternative:** `scenarios_refined.csv` + `elasticity_segments.csv`

---

### Risk Controls Applied
1. **Historical price range**  
   - Keep only scenarios where `in_hist_range == True`.  
2. **Δp cap**  
   - Restrict absolute price changes: \|Δp\| ≤ 10–15% (we used 10%).  
3. **Elasticity sign sanity**  
   - Drop β > 0 (anomalous for most goods; treat as promo-driven noise).  
4. **Confidence gating**  
   - Prefer `*_high_conf` scenarios.  
   - Allow low-confidence scenarios only if **volume ≥ 20k units**.  
5. **Profit positivity**  
   - Require **profit uplift ≥ 2%** (configurable).  
6. **Fit weighting**  
   - Compute **uplift score**:  
     \[
     \text{Uplift Score} = (\text{Profit\% Change}) \times (\text{Units}) \times \max(R^2, \text{floor})
     \]  
   - Default R² floor = 0.05 (avoids zeroing low-fit regressions).  

---

### Selection Rule
- For each **segment × margin band**, choose the Δp that **maximizes uplift score (argmax)**.  
- Then compute **portfolio roll-ups**:  
  - Volume-weighted **Revenue%** and **Profit%** across all recommended moves.  

---

### Observed Recommendations (This Dataset)
- **Survivors (post-guardrails):**  
  - **Set** and **Kurta** categories (high volume, inelastic β).  
- **Recommended Δp:** +10% (also +5% if risk appetite is lower).  
- **Portfolio summary (@ +10%):**  
  - Revenue: ~ **+6%**  
  - Profit: **+93%** @ 10% margin; **+45%** @ 20% margin; **+28%** @ 30% margin  

**Intuition:**  
In inelastic categories, raising prices slightly reduces unit sales, but the **margin expansion outweighs the loss in volume**, leading to significant profit growth.  

---

### Outputs
- `recommended_moves_category.csv`  
- `recommended_moves_sku.csv` (if SKU analysis was possible)  
- `portfolio_summary.csv`  
- `pricing_recommendations.xlsx` (Category + SKU tabs, portfolio roll-up, relaxation notes)  

---

### Console Transparency
- Script prints the **relaxation mode used** (e.g., `initial|high|minp>0.02`).  
- Ensures reviewers can see if constraints had to be loosened to yield recommendations.  

## 8) Evaluation & Scenario Testing (Step 5)

### Baseline vs. Simulation
- **Baseline:** observed revenue from cleaned sales data.  
- **Simulation:** scenario outcomes computed via elasticity (β) under Δp grid with multiple margin bands.  
- **Comparison:** R’ (simulated revenue) vs R (baseline) at both segment and portfolio levels.  

---

### Stress Tests Performed
1. **Raised R² floor / banned β near zero**  
   - Still concentrated recommendations in **Set** and **Kurta** → robustness.  
2. **Capped |Δp| at 10%**  
   - Results remain positive.  
   - Prevents extreme, non-business-friendly price changes.  
3. **Historical price band enforcement**  
   - Ensures simulations remain realistic.  
   - Avoids extrapolating to prices never seen in market.  

---

### Business Interpretation
- **Price decreases (−10% / −20%)**  
  - Typically **destroy profit** in inelastic categories.  
  - Should be reserved for **short-term promotions** only.  
- **Price increases (+5% / +10%)**  
  - Consistently deliver **revenue stability** and **profit uplift** in high-volume, inelastic categories like Set & Kurta.  

---

### Pilot Recommendation
- **Run A/B pilot** with +5% and +10% price changes in **Set** and **Kurta** for **2–4 weeks**.  
- **Rollback triggers:**  
  - Units drop > **8–12%**, OR  
  - Net revenue falls below baseline.  
- **Monitor metrics:**  
  - Units sold  
  - Conversion rate  
  - Revenue  
  - Profit  
  - Cancellation / return rates  

---

### Key Takeaway
Controlled pilots with rollback safeguards ensure the **model’s simulated gains translate into real-world profit uplift** without risking business continuity.

## 9) Why These Choices (Defensibility)

- **Δp grid (−20%, −10%, +5%, +10%)**  
  Matches common retail/B2B price-change brackets. Keeps simulations **realistic** and avoids extrapolation into price ranges not seen historically.  

- **Log-log regression**  
  A standard and **interpretable** method for elasticity estimation.  
  - β has a clear business meaning: a 1% price change → β% unit change.  
  - Robust to multiplicative noise in sales data.  

- **R² weighting, not filtering**  
  Instead of discarding segments with low R² (which is common when price variation is limited), we:  
  - Flag them as low-confidence.  
  - Down-weight their influence in portfolio roll-ups.  

- **Historical price band enforcement**  
  Ensures **business realism**. We do not recommend prices outside the observed min–max range for each segment.  

- **Profit uplift threshold**  
  Filters out scenarios where profit changes are too small or negative.  
  - Protects against “false positives” caused by noise.  
  - Guarantees recommendations are **economically meaningful**.  

- **Volume-weighted portfolio roll-up**  
  Ensures the portfolio perspective emphasizes where the business actually makes money.  
  - High-volume categories (e.g., Set, Kurta) carry more weight than niche segments.
    
## 10) Limitations & Risk

- **Low R² fits**  
  Price alone explains little of the variation in demand.  
  - Missing drivers likely include: seasonality, promotions, competitor actions, stock-outs, product placement, shipping promises, and B2B negotiated pricing.  
  - Result: elasticities should be treated as **directional signals**, not precise forecasts.  

- **Elasticity instability**  
  With few unique price points per SKU/category, β can be unstable and sensitive to outliers.  
  - Mitigation: enforce minimum thresholds for unique prices and apply conservative Δp caps (±10–20%).  

- **Assumed margins**  
  We used margin bands (10%, 20%, 30%) as placeholders because SKU-level cost of goods (COGS) data was not available.  
  - True optimization requires actual cost data to refine profit projections.  

- **No causal claims**  
  This is an **observational analysis**, not a causal inference model.  
  - Elasticities are reduced-form estimates from historical correlations.  
  - Deployment decisions must be validated through controlled **A/B experiments** with rollback triggers.  

## 11) Concrete Next Steps (Productization Path)

1. **Experimentation**  
   - Run a **pilot program** (e.g., +5% and +10% price changes) using feature flags and A/B testing in production.  
   - Collect real-world outcomes to recalibrate elasticity (β).  
   - Define **rollback triggers** if unit sales or revenue dip below thresholds.  

2. **Feature Enrichment**  
   - Integrate external and internal signals:  
     - Competitor price feeds  
     - Promotion flags (seasonal sales, discounts)  
     - Inventory/stock levels  
     - Clickstream data (sessions → add-to-cart → conversion)  
   - Goal: improve explanatory power (R²) of elasticity estimates.  

3. **Hierarchical Modeling**  
   - Apply **Bayesian or mixed-effects models** for partial pooling.  
   - Stabilize elasticity estimates across SKUs by sharing statistical strength within categories.  

4. **Time-Series & Panel Approaches**  
   - Model elasticity as **dynamic**, varying over time.  
   - Control for seasonality, promotions, and external shocks (e.g., supply chain disruptions).  

5. **Databricks / Delta Lake Infrastructure**  
   - Batch & streaming ingestion into **bronze/silver/gold layers**.  
   - Use MLflow for experiment tracking (hyperparameters, metrics, artifacts).  
   - Automate pipelines with Databricks Jobs for orchestration.  

6. **Real-Time Pricing API**  
   - Serve per-SKU dynamic price recommendations to the storefront.  
   - Add **guardrails**:  
     - Rate limits  
     - Kill-switches  
     - Circuit breakers (auto-revert if anomaly detected).
     - 


## 12) Data Dictionary (Core Fields)

### Raw & Cleaned Data Fields
| **Field**                  | **Description**                                                                 |
|-----------------------------|---------------------------------------------------------------------------------|
| **Order ID**                | Unique order identifier                                                        |
| **Date**                    | Order date (UTC normalized), `YYYY-MM-DD`                                      |
| **Status**                  | Fulfillment status; cancellations removed (`Qty=0`)                            |
| **Fulfilment, ship-service-level** | Shipping method indicators                                              |
| **SKU, ASIN, Style**        | Product identifiers (stock keeping unit, Amazon ID, style code)                |
| **Category, Size**          | Product grouping (e.g., Set, Kurta, Saree) and variant (size)                  |
| **Qty**                     | Units sold in the line                                                         |
| **Amount**                  | Line revenue (currency normalized)                                             |
| **Unit_Price**              | Derived as `Amount / Qty`                                                      |
| **ship-city, ship-state, ship-postal-code** | Destination geography for shipping                             |

---

### Analysis Artifacts (Step 3–4 Outputs)
| **Field**                | **Description**                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| **elasticity_beta**       | Estimated price elasticity coefficient (β)                                      |
| **r2**                    | Model fit quality (R² of log-log regression)                                    |
| **n_points**              | Number of aggregated (price, demand) points per segment                         |
| **unique_prices**         | Number of distinct observed price points per segment                            |
| **total_units_observed**  | Total sales volume across the segment                                           |
| **price_min/median/max**  | Historical price range (guardrail for simulations)                              |
| **in_hist_range**         | Boolean flag: whether scenario price lies within historical bounds              |
| **confidence**            | Scenario-level confidence (high/low, in-range)                                  |
| **fit_confidence**        | Fit-level confidence derived from R² and sample thresholds                      |
| **business_interpretation** | Human-readable label of elasticity (e.g., *very inelastic — strong pricing power*) |


---

## 13) AI-Assisted Development (Disclosure)

We used AI assistance as a **coding copilot and reviewer** to accelerate:

- Scaffolding of **data-cleaning utilities** and CLI plumbing.  
- Generating **boilerplate for regression, grouping, and table outputs**.  
- Drafting **human-readable business reviews** for each step.  

🔎 **Important:** All **critical logic** (filters, formulas, guardrails, and final recommendations) was **validated manually** and cross-checked with console outputs/artifacts.  
This ensures that while AI accelerated development, the decisions and interpretations remain sound and defensible.

---

## 14) Conclusion

This project delivers a **transparent, defensible, and extensible dynamic pricing analysis**.  

Key takeaways:

- It clearly surfaces where the business has **pricing power** (Set, Kurta).  
- Provides **conservative, testable recommendations** (+5–10%).  
- Includes **guardrails** to de-risk rollout (historical price checks, profit uplift filters, fit thresholds).  
- With richer features (promotions, competitors, seasonality) and an **experimentation loop**, this can evolve into a **production-grade pricing engine** deployed atop **Databricks/Delta** and monitored end-to-end.  

---

## 15) Appendix — Command Reference

Quick reference for running each step of the pipeline:

 **EDA**  
  ```bash
  python ds_modules/eda_overview.py --path raw/ecommerce_sales.csv
  ```
 **Clean**
  
  ```bash
  python ds_modules/clean_dataset.py --path raw/ecommerce_sales.csv 
  
  ```
**Elasticities**

```bash
python elasticity_estimation.py --path clean_outputs/ecommerce_sales_cleaned.csv 
 
```
**Merge**
```bash
python ds_modules/merge_scenarios.py 
```
**Optimize**
```bash
python ds_modules/pricing_optimization.py
```

