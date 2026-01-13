import pandas as pd
import numpy as np
from pathlib import Path
import ruptures as rpt

ROOT = Path(".").resolve()
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

long_path = OUT / "enplanements_long.csv"
pool_path = OUT / "airports_model_pool.csv"   # 如果你想严格按 model pool 聚合
out_csv = OUT / "fig3_pelt_aggregate_signal.csv"

df = pd.read_csv(long_path)

# 统一类型
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["enplanements"] = pd.to_numeric(df["enplanements"], errors="coerce")
df = df.dropna(subset=["year", "enplanements"])
df["year"] = df["year"].astype(int)

# 可选：只用 model pool（与你 Model I 更一致）
if pool_path.exists():
    pool = pd.read_csv(pool_path)["airport_code_std"].astype(str).str.strip().str.upper()
    df["airport_code_std"] = df["airport_code_std"].astype(str).str.strip().str.upper()
    df = df[df["airport_code_std"].isin(set(pool))].copy()

# 限定范围
df = df[df["year"].between(2000, 2024)]

# 年度聚合
ts = (df.groupby("year", as_index=False)["enplanements"]
        .sum()
        .sort_values("year")
        .reset_index(drop=True))

# PELT 变点检测（与你论文/代码一致）
signal = ts["enplanements"].values
algo = rpt.Pelt(model="rbf").fit(signal)
bkps = algo.predict(pen=10)  # pen=10 你当前设定

# bkps 是“索引终点”，最后一个是 len(signal)，我们只取前面的断点
break_years = [int(ts["year"].iloc[i-1]) for i in bkps[:-1]]  # i-1 对应断点位置
print("Detected change points (years):", break_years)

# 输出表给画图用
ts["total_enplanements_million"] = ts["enplanements"] / 1e6
ts["is_disruption"] = ts["year"].between(2020, 2022).astype(int)

ts.to_csv(out_csv, index=False)
print("Saved:", out_csv)