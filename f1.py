import pandas as pd
from pathlib import Path

# -----------------------
# Paths
# -----------------------
ROOT = Path(".").resolve()
OUTDIR = ROOT / "outputs"

long_path = OUTDIR / "enplanements_long.csv"
top30_path = OUTDIR / "airports_rank_pool_topN.csv"
out_csv = OUTDIR / "fig1_aggregate_totals_2000_2024.csv"

assert long_path.exists(), f"Missing: {long_path}"
assert top30_path.exists(), f"Missing: {top30_path}"

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(long_path)

# Expected columns check (adjust if your column names differ)
need_cols = {"airport_code_std", "year", "enplanements"}
missing = need_cols - set(df.columns)
if missing:
    raise ValueError(f"enplanements_long.csv missing columns: {sorted(missing)}")

top30 = pd.read_csv(top30_path)
if "airport_code_std" not in top30.columns:
    raise ValueError("airports_rank_pool_topN.csv must contain column: airport_code_std")
top30_set = set(top30["airport_code_std"].astype(str).str.strip().str.upper())

# -----------------------
# Clean / type cast
# -----------------------
df["airport_code_std"] = df["airport_code_std"].astype(str).str.strip().str.upper()
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["enplanements"] = pd.to_numeric(df["enplanements"], errors="coerce")

# Keep 2000–2024 only (Figure 1 range)
df = df[df["year"].between(2000, 2024, inclusive="both")].copy()

# Drop missing enplanements (summing ignores NaN but we keep it explicit)
df = df.dropna(subset=["year", "enplanements"])

# -----------------------
# Aggregate totals
# -----------------------
all_total = (
    df.groupby("year", as_index=False)["enplanements"]
      .sum()
      .rename(columns={"enplanements": "all_total"})
)

top30_total = (
    df[df["airport_code_std"].isin(top30_set)]
      .groupby("year", as_index=False)["enplanements"]
      .sum()
      .rename(columns={"enplanements": "top30_total"})
)

# Merge and compute share
agg = all_total.merge(top30_total, on="year", how="left")
agg["top30_total"] = agg["top30_total"].fillna(0.0)
agg["top30_share"] = agg["top30_total"] / agg["all_total"]

# Optional: convert to millions for plotting convenience (keep raw too)
agg["all_total_million"] = agg["all_total"] / 1e6
agg["top30_total_million"] = agg["top30_total"] / 1e6

# -----------------------
# Pre-pandemic share (2018–2019 avg)
# -----------------------
pre = agg[agg["year"].isin([2018, 2019])].copy()
share_18_19 = (pre["top30_total"].mean() / pre["all_total"].mean()) if len(pre) == 2 else None

print("=== Figure 1 Prep ===")
print(f"Years in table: {agg['year'].min()}–{agg['year'].max()}  (n={len(agg)})")
if share_18_19 is not None:
    print(f"Top-30 share (avg 2018–2019): {share_18_19*100:.2f}%")
else:
    print("Top-30 share (avg 2018–2019): NA (missing 2018/2019 in agg table)")

# -----------------------
# Save output table
# -----------------------
agg = agg.sort_values("year")
agg.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")