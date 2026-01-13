import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- Config ----------------
base_dir = Path(__file__).resolve().parent
output_dir = base_dir

B = 5000
PI_LEVELS = [0.80, 0.95]

# 与 model2.py 一致
PRED_FLOOR_FRAC = 0.05

# 核心增强：Block bootstrap + 分层残差
BLOCK_BOOTSTRAP = True      # 按年抽“残差块”，捕捉共同冲击
STRATIFIED = True           # 按机场规模分层抽残差
N_STRATA = 3                # 规模分层数（3=三分位）

USE_TOP30_RESIDUALS = True  # 不确定性残差分布只用 Top30（更贴题）
MAKE_PLOTS = True

# ---------------- Load ----------------
pred = pd.read_csv(output_dir / "predictions_2025_detailed.csv")
val = pd.read_csv(output_dir / "validation_results_2023_2024.csv")
rank_pool = pd.read_csv(output_dir / "airports_rank_pool_topN.csv")["airport_code_std"].tolist()

pred = pred.rename(columns={"enplanements_pred_2025": "yhat_2025"})
pred = pred[pred["airport_code_std"].isin(rank_pool)].copy()

# 基本 dict
airports = pred["airport_code_std"].tolist()
yhat = pred.set_index("airport_code_std")["yhat_2025"].to_dict()
cf = pred.set_index("airport_code_std")["enplanements_counterfactual"].to_dict()

# ---------------- Residuals (relative) ----------------
val = val.dropna(subset=["enplanements", "enplanements_pred", "year"]).copy()
val = val[val["enplanements_pred"] > 0].copy()
val["rel_resid"] = (val["enplanements"] - val["enplanements_pred"]) / val["enplanements_pred"]
val["rel_resid"] = val["rel_resid"].replace([np.inf, -np.inf], np.nan).dropna()

# 只用 Top30 的误差分布（更贴 Problem C）
if USE_TOP30_RESIDUALS:
    val = val[val["airport_code_std"].isin(rank_pool)].copy()

resid_all = val["rel_resid"].values
if len(resid_all) < 50:
    raise RuntimeError("残差样本太少（<50），无法做 bootstrap。请检查 validation_results 文件。")

# ---------------- Stratify by size (based on yhat_2025) ----------------
if STRATIFIED:
    # 用 yhat_2025 做规模分层
    pred_tmp = pred[["airport_code_std", "yhat_2025"]].copy()
    # qcut 可能因重复值导致报错，duplicates='drop' 更稳
    pred_tmp["size_bin"] = pd.qcut(pred_tmp["yhat_2025"], q=N_STRATA, labels=False, duplicates="drop")
    airport_to_bin = dict(zip(pred_tmp["airport_code_std"], pred_tmp["size_bin"]))
else:
    airport_to_bin = {a: "all" for a in airports}

# 把 size_bin merge 到 val 里（用于构造分层残差池）
val["size_bin"] = val["airport_code_std"].map(airport_to_bin)

# ---------------- Build residual pools ----------------
# 我们构造两类池：
# 1) (year, size_bin)  -> residual array  (block+strat)
# 2) ('all', size_bin) -> residual array  (fallback)
# 3) ('all','all')     -> residual array  (ultimate fallback)

resid_pools = {}

# 年份列表（用于 block bootstrap 抽年）
years_avail = sorted(val["year"].dropna().unique().tolist())
if len(years_avail) == 0:
    years_avail = [2023, 2024]

# year + bin pools
if STRATIFIED:
    for yy in years_avail:
        for bb in sorted(val["size_bin"].dropna().unique().tolist()):
            arr = val[(val["year"] == yy) & (val["size_bin"] == bb)]["rel_resid"].values
            if len(arr) > 0:
                resid_pools[(yy, bb)] = arr

    # all-year bin pools
    for bb in sorted(val["size_bin"].dropna().unique().tolist()):
        arr = val[val["size_bin"] == bb]["rel_resid"].values
        if len(arr) > 0:
            resid_pools[("all", bb)] = arr
else:
    for yy in years_avail:
        arr = val[val["year"] == yy]["rel_resid"].values
        if len(arr) > 0:
            resid_pools[(yy, "all")] = arr

# ultimate fallback
resid_pools[("all", "all")] = resid_all

def sample_resid(rng, year_key, size_key):
    """Sample one residual from the best available pool with fallbacks."""
    # best: (year, bin)
    if (year_key, size_key) in resid_pools and len(resid_pools[(year_key, size_key)]) > 0:
        return float(rng.choice(resid_pools[(year_key, size_key)]))
    # fallback: ('all', bin)
    if ("all", size_key) in resid_pools and len(resid_pools[("all", size_key)]) > 0:
        return float(rng.choice(resid_pools[("all", size_key)]))
    # fallback: (year, 'all')
    if (year_key, "all") in resid_pools and len(resid_pools[(year_key, "all")]) > 0:
        return float(rng.choice(resid_pools[(year_key, "all")]))
    # ultimate
    return float(rng.choice(resid_pools[("all", "all")]))

# ---------------- Bootstrap simulation ----------------
sim_rank = {a: [] for a in airports}
sim_y = {a: [] for a in airports}

rng = np.random.default_rng(42)

for _ in range(B):
    # block bootstrap: 抽一个“年份块”，让所有机场共享同一年残差分布（共同冲击）
    sampled_year = rng.choice(years_avail) if BLOCK_BOOTSTRAP else "all"

    y_sim = {}
    for a in airports:
        bb = airport_to_bin.get(a, "all") if STRATIFIED else "all"
        rr = sample_resid(rng, sampled_year, bb)

        y = yhat[a] * (1.0 + rr)
        # floor + non-negativity（与 model2 一致思想）
        y = max(y, PRED_FLOOR_FRAC * cf[a])
        y_sim[a] = y
        sim_y[a].append(y)

    # ranks within Top30 (1 = highest)
    ordered = sorted(y_sim.items(), key=lambda x: x[1], reverse=True)
    for rank, (a, _) in enumerate(ordered, start=1):
        sim_rank[a].append(rank)

# ---------------- Summaries ----------------
interval_rows = []
rank_rows = []

for a in airports:
    ys = np.array(sim_y[a])
    ranks = np.array(sim_rank[a])

    row_i = {"airport_code_std": a, "yhat_2025": yhat[a]}
    for lvl in PI_LEVELS:
        lo = np.quantile(ys, (1 - lvl) / 2)
        hi = np.quantile(ys, 1 - (1 - lvl) / 2)
        row_i[f"PI{int(lvl*100)}_low"] = lo
        row_i[f"PI{int(lvl*100)}_high"] = hi
    interval_rows.append(row_i)

    row_r = {
        "airport_code_std": a,
        "rank_p05": float(np.quantile(ranks, 0.05)),
        "rank_p50": float(np.quantile(ranks, 0.50)),
        "rank_p95": float(np.quantile(ranks, 0.95)),
        "p_top10": float((ranks <= 10).mean()),
        "p_top20": float((ranks <= 20).mean()),
        "p_top30": float((ranks <= 30).mean()),
    }
    row_r["rank_stability"] = float(1.0 - (row_r["rank_p95"] - row_r["rank_p05"]) / 30.0)
    rank_rows.append(row_r)

df_int = pd.DataFrame(interval_rows).sort_values("yhat_2025", ascending=False)
df_rank = pd.DataFrame(rank_rows).merge(df_int[["airport_code_std", "yhat_2025"]], on="airport_code_std", how="left")
df_rank = df_rank.sort_values("yhat_2025", ascending=False)

df_int.to_csv(output_dir / "uncertainty_intervals_2025.csv", index=False)
df_rank.to_csv(output_dir / "rank_probabilities_2025.csv", index=False)

# 论文级汇总表（Top-30）
summary_cols = [
    "airport_code_std","yhat_2025",
    "PI80_low","PI80_high","PI95_low","PI95_high"
]
df_summary = df_rank.merge(df_int[summary_cols], on=["airport_code_std","yhat_2025"], how="inner")

# 排序并输出
df_summary = df_summary.sort_values("yhat_2025", ascending=False)
df_summary.to_csv(output_dir / "top30_summary_2025.csv", index=False)

print("Saved:", output_dir / "uncertainty_intervals_2025.csv")
print("Saved:", output_dir / "rank_probabilities_2025.csv")
print("Saved:", output_dir / "top30_summary_2025.csv")
print("model3 finished successfully.")

# ---------------- Optional plots ----------------
if MAKE_PLOTS:
    import matplotlib.pyplot as plt

    plot_df = df_summary.copy().head(30).reset_index(drop=True)

    # 1) PI95 errorbar plot
    plt.figure(figsize=(12, 8))
    x = np.arange(len(plot_df))
    y = plot_df["yhat_2025"].values
    yerr_low = y - plot_df["PI95_low"].values
    yerr_high = plot_df["PI95_high"].values - y
    plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt='o', capsize=3)
    plt.xticks(x, plot_df["airport_code_std"].values, rotation=90)
    plt.title("Top-30 2025 Enplanements Forecast with 95% Prediction Intervals")
    plt.tight_layout()
    plt.savefig(output_dir / "Figure_ModelIII_Top30_PI95.png", dpi=300)
    plt.close()

    # 2) p_top10 bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["airport_code_std"].values, plot_df["p_top10"].values)
    plt.xticks(rotation=90)
    plt.ylim(0, 1.0)
    plt.title("Top-30: Probability of Being in Top-10 (p_top10)")
    plt.tight_layout()
    plt.savefig(output_dir / "Figure_ModelIII_Top30_pTop10.png", dpi=300)
    plt.close()

    print("Saved:", output_dir / "Figure_ModelIII_Top30_PI95.png")
    print("Saved:", output_dir / "Figure_ModelIII_Top30_pTop10.png")
