import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import ruptures as rpt  # 用于变点检测: pip install ruptures

# ==========================================
# 1. 配置与加载数据
# ==========================================
# 读取你整理好的数据
base_dir = Path(__file__).resolve().parent
root_dir = base_dir.parent

output_dir = base_dir  # keep all artifacts inside outputs/
output_dir.mkdir(parents=True, exist_ok=True)

# 宏观特征（请确保包含 year 列）
df_macro = pd.read_csv(root_dir / 'macro_features.csv')
required_cols = {
    'year', 'gdp_real_trillion', 'fuel_price_dollar', 'population_million',
    'income_capita_real', 'covid_stringency'
}
missing_cols = required_cols - set(df_macro.columns)
if missing_cols:
    raise ValueError(f"macro_features.csv 缺少列: {sorted(missing_cols)}")

df_long = pd.read_csv(base_dir / 'enplanements_long.csv')
pool_list = pd.read_csv(base_dir / 'airports_model_pool.csv')['airport_code_std'].tolist()
rank_pool = pd.read_csv(base_dir / 'airports_rank_pool_topN.csv')['airport_code_std'].tolist()

# 过滤出主分析集合
df = df_long[df_long['airport_code_std'].isin(pool_list)].copy()

# 合并宏观特征
df = df.merge(df_macro, on='year', how='left')

# Phase I 反事实基线特征：不使用 covid_stringency（它本身是疫情干预强度，会污染“无疫情”基线）
CF_FEATURES = ['year', 'gdp_real_trillion', 'fuel_price_dollar', 'population_million', 'income_capita_real']
# 若后续要做“疫情影响解释/恢复”可使用该特征
AUX_FEATURES = ['covid_stringency']

# ==========================================
# 2. 变点检测 (PELT) - 论文 Figure 素材
# ==========================================
print("正在执行变点检测（ruptures PELT）...")

# 聚合全美总客流量 (Total Enplanements)
total_ts = df.groupby('year')['enplanements'].sum().reset_index()
signal = total_ts['enplanements'].values

# 使用 Pelt 算法检测断点 (Penalty linear transformation)
# 这里假设它是正态分布的突变
algo = rpt.Pelt(model="rbf").fit(signal)
result = algo.predict(pen=10) # 惩罚项 pen 可以调整，控制断点数量

# 绘图
plt.figure(figsize=(12, 6))
# 绘制原始数据
plt.plot(total_ts['year'], total_ts['enplanements'], label='Total Enplanements', linewidth=2.5, color='#1f77b4')

# 绘制断点
detected_years = [total_ts['year'].iloc[min(i, len(total_ts)-1)] for i in result[:-1]] # 转换索引为年份
print(f"检测到的结构性断点年份: {detected_years}")

for j, year in enumerate(detected_years):
    lbl = f'Detected Break(s)' if j == 0 else None
    plt.axvline(x=year, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=lbl)

plt.title('Change Point Detection (PELT) on US Aviation Traffic', fontsize=14, fontweight='bold')
plt.ylabel('Total Enplanements')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(output_dir / 'Figure_Phase1_CPD.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 3. 构建反事实基线 (Counterfactuals)
# ==========================================
print("正在构建反事实基线 (Counterfactual Baseline)...")

# 宏观特征用于预测的年份范围（需要包含 2025）
macro_subset = df_macro[(df_macro['year'] >= 2000) & (df_macro['year'] <= 2025)].copy()
if 2025 not in set(macro_subset['year']):
    raise ValueError("macro_features.csv 需要包含 year=2025 的宏观特征预测值")

# 构建“无疫情”宏观输入：把 2020-2022 的 stringency 置 0（不直接进入 CF_FEATURES，但便于后续一致性/扩展）
macro_subset_cf = macro_subset.copy()
macro_subset_cf.loc[macro_subset_cf['year'].between(2020, 2022), 'covid_stringency'] = 0

results = []

for airport in pool_list:
    # 取出该机场数据
    sub_df = df[df['airport_code_std'] == airport].sort_values('year')
    
    # 训练集: 2000-2019 (Pre-Covid)
    train_mask = (sub_df['year'] >= 2000) & (sub_df['year'] <= 2019)
    test_mask = (sub_df['year'] >= 2000) & (sub_df['year'] <= 2025) # 预测全时段
    
    # 训练用特征（只用 pre-covid 年份且剔除宏观缺失）
    X_train = sub_df.loc[train_mask, CF_FEATURES]
    y_train = sub_df.loc[train_mask, 'enplanements']

    train_pack = pd.concat([X_train, y_train], axis=1).dropna()
    X_train = train_pack[CF_FEATURES]
    y_train = train_pack['enplanements']
    
    # 如果数据不足，跳过
    if len(X_train) < 10:
        continue
        
    # 建立简单的线性回归作为“无疫情基线”
    # 逻辑: 假设 GDP 和人口按照正常趋势发展，客流应该多少？
    model_cf = LinearRegression()
    model_cf.fit(X_train, y_train)
    
    # 预测 Counterfactual（无疫情基线）
    cf_preds = model_cf.predict(macro_subset_cf[CF_FEATURES])

    # 存储结果
    temp_res = macro_subset_cf[['year']].copy()
    temp_res['airport_code_std'] = airport
    temp_res['enplanements_counterfactual'] = cf_preds
    
    results.append(temp_res)

if not results:
    raise RuntimeError("没有生成任何反事实结果：请检查训练数据是否满足 len(X_train) >= 10 且宏观特征无缺失")
df_cf = pd.concat(results, ignore_index=True)

# 将 Counterfactual 拼回原数据
# 注意: 原数据可能只有到 2024，这里会把 2025 的空行拼进来
df_final = pd.merge(df, df_cf, on=['year', 'airport_code_std'], how='left')

# 计算 Gap (Actual - Counterfactual)
# 2025 年 Actual 是空的，这是我们要预测的目标
df_final['gap_to_trend'] = df_final['enplanements'] - df_final['enplanements_counterfactual']
df_final['recovery_ratio'] = df_final['enplanements'] / df_final['enplanements_counterfactual']

# 保存给 Phase II 使用
df_final.to_csv(output_dir / 'data_phase2_input.csv', index=False)
print("Phase I 完成！已生成 data_phase2_input.csv")

# ==========================================
# 4. 可视化反事实 (论文 Figure 素材)
# ==========================================
# 挑几个 Top 机场画图（从 rank_pool 取前 4 个；若不足则回退到 pool_list）
top_airports = (rank_pool[:4] if len(rank_pool) >= 4 else pool_list[:4])
plt.figure(figsize=(15, 10))

for i, airport in enumerate(top_airports):
    plt.subplot(2, 2, i+1)
    data = df_final[df_final['airport_code_std'] == airport]
    
    # 真实值
    plt.plot(data['year'], data['enplanements'], 'b-o', label='Actual')
    # 反事实基线
    plt.plot(data['year'], data['enplanements_counterfactual'], 'r--', label='Counterfactual (No Pandemic)')
    
    # 标注 Gap
    plt.fill_between(data['year'], data['enplanements'], data['enplanements_counterfactual'], 
                     where=(data['year']>=2020), color='gray', alpha=0.3, label='Pandemic Impact')
    
    plt.title(f'{airport}: Actual vs Counterfactual')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'Figure_Phase1_Counterfactuals.png', dpi=300, bbox_inches='tight')
plt.show()
