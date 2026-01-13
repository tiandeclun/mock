import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

base_dir = Path(__file__).resolve().parent
root_dir = base_dir.parent
output_dir = base_dir

# 可调：对 2025 年的防疫强度作情景假设（0=无强制管控）
STRINGENCY_2025 = 0.0

# 可调：建模目标
# - 'absolute': gap = actual - counterfactual
# - 'ratio'   : gap = (actual - counterfactual) / counterfactual   （推荐，跨机场尺度更稳）
TARGET_MODE = 'absolute'  # ratio 在当前实现下会显著劣化 hold-out，先用 absolute 更稳

# 防止极端预测把小机场压成 0（ratio 模式下：最小 -0.95 表示预测至少保留 5% 的基线）
# 防止极端预测把小机场压成 0（ratio 模式下：最小 -0.95 表示预测至少保留 5% 的基线）
RATIO_CLIP = (-0.95, 3.0)

# 预测下限：避免出现大量 0 预测导致 MAPE/APE 爆炸（与基线规模成比例）
PRED_FLOOR_FRAC = 0.05  # 至少保留 5% 的反事实基线

# ------- Metrics helpers -------
def safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = (y_true.notna()) & (y_pred.notna()) & (y_true != 0)
    if mask.sum() == 0:
        return float('nan')
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()) * 100.0

def wmape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = (y_true.notna()) & (y_pred.notna()) & (y_true != 0)
    if mask.sum() == 0:
        return float('nan')
    return (np.abs(y_true[mask] - y_pred[mask]).sum() / np.abs(y_true[mask]).sum()) * 100.0

def median_ape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = (y_true.notna()) & (y_pred.notna()) & (y_true != 0)
    if mask.sum() == 0:
        return float('nan')
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.median(ape) * 100.0)

# ==========================================
# 0. 数据准备与特征工程
# ==========================================
print("Step 0: Loading Data & Feature Engineering...")

# 读取 Phase I 的产出（优先使用根目录的 2000-2025 版本）
phase2_path_root = root_dir / 'data_phase2_input.csv'
phase2_path_out = output_dir / 'data_phase2_input.csv'
df = pd.read_csv(phase2_path_root if phase2_path_root.exists() else phase2_path_out)
# 读取宏观数据 (主要为了确保 covid_stringency 在里面，如果 data_phase2_input 已经有了可以省略)
df_macro = pd.read_csv(root_dir / 'macro_features.csv')

# 读取 Phase I 的反事实序列（包含 2000-2025），用于构造 2025 预测行
cf_path = output_dir / 'counterfactual_by_airport_year.csv'
if cf_path.exists():
    cf = pd.read_csv(cf_path)
else:
    # 从 Phase I 产出中提取反事实序列作为兜底
    cf = df[['year', 'airport_code_std', 'enplanements_counterfactual']].dropna().copy()
    cf.to_csv(cf_path, index=False)

# 读取机场池（与 Phase I 一致）
pool_list = pd.read_csv(output_dir / 'airports_model_pool.csv')['airport_code_std'].tolist()
rank_pool = pd.read_csv(output_dir / 'airports_rank_pool_topN.csv')['airport_code_std'].tolist()

# 重新 merge 一次 macro 确保万无一失（先安全移除可能已存在的宏观列）
macro_cols = ['gdp_real_trillion', 'fuel_price_dollar', 'population_million', 'income_capita_real', 'covid_stringency']
df = df.drop(columns=[c for c in macro_cols if c in df.columns], errors='ignore')
df = df.merge(df_macro, on='year', how='left')

# 2025 的 stringency 若缺失，按情景值填充
if 'covid_stringency' in df.columns:
    df.loc[(df['year'] == 2025) & (df['covid_stringency'].isna()), 'covid_stringency'] = STRINGENCY_2025

# --------- 定义要学习的 gap 目标 ---------
if TARGET_MODE == 'ratio':
    df['gap_target'] = (df['enplanements'] - df['enplanements_counterfactual']) / df['enplanements_counterfactual']
else:
    df['gap_target'] = df['enplanements'] - df['enplanements_counterfactual']

# 清理无穷/异常（例如 counterfactual=0 导致 ratio 无穷大）
df.loc[~np.isfinite(df['gap_target']), 'gap_target'] = np.nan

# 特征工程 1: 滞后特征 (Lag Features) - 非常重要！
# 逻辑: 今年的恢复情况很大程度取决于去年
# 注意: 对于 2025 的预测，我们需要 2024 的真实 Gap。
df = df.sort_values(['airport_code_std', 'year'])
df['gap_lag1'] = df.groupby('airport_code_std')['gap_target'].shift(1)

# 填充 Lag 的缺失值 (2000年没有前一年，填0)
df['gap_lag1'] = df['gap_lag1'].fillna(0)

# 特征工程 2: 机场编码 (Label Encoding)
# 注意：避免信息泄露，只在训练集机场上 fit；未见过机场编码为 -1
le = LabelEncoder()

# 训练集：历史有真实 gap_target 的记录
train_df = df[df['year'] <= 2024].dropna(subset=['gap_target']).copy()

# fit 编码器并写入 train_df 与 df（df 里只有用于 lag/merge；编码主要用于训练/预测）
le.fit(train_df['airport_code_std'])
train_df['airport_idx'] = le.transform(train_df['airport_code_std'])

# 给 df 也生成 airport_idx（供 hold-out 验证直接切片使用）
known_codes_all = set(le.classes_)
df['airport_idx'] = df['airport_code_std'].astype(str).apply(lambda c: int(le.transform([c])[0]) if c in known_codes_all else -1)

# 预测集：用 counterfactual 表构造 2025 行（避免 Phase I left-merge 导致 df 中没有 2025）
predict_df = cf[(cf['year'] == 2025) & (cf['airport_code_std'].isin(pool_list))].copy()
# 合并宏观特征（year=2025）
predict_df = predict_df.merge(df_macro, on='year', how='left')
# 2025 的 stringency 若缺失，按情景值填充
if 'covid_stringency' in predict_df.columns:
    predict_df.loc[predict_df['covid_stringency'].isna(), 'covid_stringency'] = STRINGENCY_2025

# 如果 2025 的 gap_lag1 是空的 (因为 shift 可能会有问题如果 2025 行本来不存在)，我们需要手动把 2024 的 gap 填给 2025
# 确保 predict_df 的 gap_lag1 是准确的
last_year_gaps = df[df['year'] == 2024][['airport_code_std', 'gap_target']].rename(columns={'gap_target': 'gap_lag1'})
predict_df = predict_df.drop(columns=['gap_lag1'], errors='ignore')
predict_df = predict_df.merge(last_year_gaps, on='airport_code_std', how='left')
# 若某机场缺少 2024 gap（极少数情况），用 0 填充以保证可预测
predict_df['gap_lag1'] = predict_df['gap_lag1'].fillna(0)

# 预测集机场编码（与训练保持一致；未见过的机场用 -1）
known_codes = set(le.classes_)
predict_df['airport_idx'] = predict_df['airport_code_std'].apply(
    lambda c: int(le.transform([c])[0]) if c in known_codes else -1
)

print(f"训练集样本数: {len(train_df)}, 预测目标机场数: {len(predict_df)}")

# ==========================================
# Stage 1: SVR (Trend Modeling)
# 目标: 用 疫情指数 和 时间 拟合 Gap 的大趋势
# ==========================================
print("\nStage 1: Training SVR (Recovery Shape Modeling)...")

svr_features = ['covid_stringency', 'year']
X_svr = train_df[svr_features]
y_svr = train_df['gap_target']

# SVR 对尺度敏感，必须标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_svr_scaled = scaler_X.fit_transform(X_svr)
y_svr_scaled = scaler_y.fit_transform(y_svr.values.reshape(-1, 1)).ravel()

# 训练 SVR (RBF Kernel)
svr_model = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
svr_model.fit(X_svr_scaled, y_svr_scaled)

# 预测 SVR
train_df['svr_pred_scaled'] = svr_model.predict(X_svr_scaled)
train_df['svr_gap_pred'] = scaler_y.inverse_transform(train_df['svr_pred_scaled'].values.reshape(-1, 1)).ravel()

# 计算 Residual (SVR 没预测准的部分)
train_df['residual'] = train_df['gap_target'] - train_df['svr_gap_pred']

# 对 2025 进行 SVR 预测
X_2025_svr = predict_df[svr_features]
X_2025_svr_scaled = scaler_X.transform(X_2025_svr)
predict_df['svr_gap_pred'] = scaler_y.inverse_transform(svr_model.predict(X_2025_svr_scaled).reshape(-1, 1)).ravel()

print("SVR Training Done. RMSE:", float(np.sqrt(mean_squared_error(train_df['gap_target'], train_df['svr_gap_pred']))))

# ==========================================
# Stage 2: XGBoost (Residual Correction)
# 目标: 用 宏观经济 + 机场特征 + 滞后项 修正 SVR 的错误
# ==========================================
print("\nStage 2: Training XGBoost (Residual Correction)...")

xgb_features = ['gdp_real_trillion', 'fuel_price_dollar', 'income_capita_real', 
                'population_million', 'gap_lag1', 'airport_idx']

X_xgb = train_df[xgb_features]
y_xgb = train_df['residual'] # 目标是预测残差！

# XGBoost 模型参数 (可以微调)
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_xgb, y_xgb)

# 预测 2025 残差
predict_df['xgb_residual_pred'] = xgb_model.predict(predict_df[xgb_features])

# ==========================================
# 3. 合并结果与最终预测
# ==========================================
print("\nStep 3: Generating Final Forecasts...")

# 最终 Gap = SVR预测 + XGB修正
predict_df['final_gap_pred'] = predict_df['svr_gap_pred'] + predict_df['xgb_residual_pred']

# 最终客流：根据 gap 定义合成
if TARGET_MODE == 'ratio':
    # 约束 ratio 预测，防止极端负值把小机场压成 0
    predict_df['final_gap_pred'] = predict_df['final_gap_pred'].clip(RATIO_CLIP[0], RATIO_CLIP[1])
    predict_df['enplanements_pred_2025'] = predict_df['enplanements_counterfactual'] * (1.0 + predict_df['final_gap_pred'])
else:
    predict_df['enplanements_pred_2025'] = predict_df['enplanements_counterfactual'] + predict_df['final_gap_pred']

predict_df['enplanements_pred_2025'] = predict_df['enplanements_pred_2025'].clip(lower=0)
# 进一步加地板，防止极端负 gap 把小机场压成 0
predict_df['enplanements_pred_2025'] = np.maximum(
    predict_df['enplanements_pred_2025'],
    PRED_FLOOR_FRAC * predict_df['enplanements_counterfactual']
)

# 保存详细结果
output_columns = ['year', 'airport_code_std', 'enplanements_counterfactual', 'svr_gap_pred', 
                  'xgb_residual_pred', 'final_gap_pred', 'enplanements_pred_2025']
predict_df[output_columns].to_csv(output_dir / 'predictions_2025_detailed.csv', index=False)

# 合并回历史数据以便画图
history_df = train_df[['year', 'airport_code_std', 'enplanements', 'enplanements_counterfactual']].copy()
history_df['type'] = 'History'
future_df = predict_df[['year', 'airport_code_std', 'enplanements_pred_2025', 'enplanements_counterfactual']].rename(columns={'enplanements_pred_2025': 'enplanements'})
future_df['type'] = 'Forecast'

full_df = pd.concat([history_df, future_df], ignore_index=True)
full_df.to_csv(output_dir / 'final_dataset_with_forecast.csv', index=False)

print("预测完成！结果已保存至 predictions_2025_detailed.csv")

# ==========================================
# 补充：Hold-out Validation (2023–2024)
# 目的：避免只看训练集误差；用 2000–2022 训练，检验 2023–2024 的泛化能力
# ==========================================
print("\n=== Hold-out Validation: train<=2022, validate on 2023–2024 ===")

# 使用与主流程一致的 df（已合并宏观特征、已构造 gap_lag1）
# 只在主分析集合（model pool）上验证，避免把不在研究范围的小机场拉低指标
train_hold = df[(df['year'] <= 2022) & (df['airport_code_std'].isin(pool_list))].dropna(subset=['gap_target']).copy()
val_hold = df[(df['year'] >= 2023) & (df['year'] <= 2024) & (df['airport_code_std'].isin(pool_list))].dropna(subset=['gap_target']).copy()

if len(val_hold) == 0:
    print("[WARN] 数据集中没有 2023–2024 的 gap_target，可跳过验证。")
else:
    # ---- Stage 1: SVR on GAP ----
    X_tr_svr = train_hold[svr_features]
    y_tr_svr = train_hold['gap_target']

    scaler_X_h = StandardScaler()
    scaler_y_h = StandardScaler()

    X_tr_s = scaler_X_h.fit_transform(X_tr_svr)
    y_tr_s = scaler_y_h.fit_transform(y_tr_svr.values.reshape(-1, 1)).ravel()

    svr_h = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
    svr_h.fit(X_tr_s, y_tr_s)

    # SVR predictions for train/val
    train_hold['svr_gap_pred'] = scaler_y_h.inverse_transform(
        svr_h.predict(X_tr_s).reshape(-1, 1)
    ).ravel()

    X_val_s = scaler_X_h.transform(val_hold[svr_features])
    val_hold['svr_gap_pred'] = scaler_y_h.inverse_transform(
        svr_h.predict(X_val_s).reshape(-1, 1)
    ).ravel()

    # ---- Stage 2: XGBoost on residual ----
    train_hold['residual'] = train_hold['gap_target'] - train_hold['svr_gap_pred']

    # 避免编码泄露：只用训练集 fit；验证集未见机场用 -1 兜底
    le_h = LabelEncoder()
    le_h.fit(train_hold['airport_code_std'])
    train_hold['airport_idx'] = le_h.transform(train_hold['airport_code_std'])

    val_codes = val_hold['airport_code_std'].astype(str)
    known = set(le_h.classes_)
    val_hold['airport_idx'] = val_codes.apply(lambda c: int(le_h.transform([c])[0]) if c in known else -1)

    # 确保验证集 gap_lag1 已有（来自 df 的 shift）；缺失则填 0
    if 'gap_lag1' not in val_hold.columns:
        val_hold = val_hold.sort_values(['airport_code_std', 'year'])
        val_hold['gap_lag1'] = val_hold.groupby('airport_code_std')['gap_target'].shift(1)
    val_hold['gap_lag1'] = val_hold['gap_lag1'].fillna(0)

    X_tr_xgb = train_hold[xgb_features]
    y_tr_xgb = train_hold['residual']

    xgb_h = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_h.fit(X_tr_xgb, y_tr_xgb)

    val_hold['xgb_residual_pred'] = xgb_h.predict(val_hold[xgb_features])
    val_hold['final_gap_pred'] = val_hold['svr_gap_pred'] + val_hold['xgb_residual_pred']

    # 最终预测值（用验证集当年的反事实基线 + 预测的 gap）
    if TARGET_MODE == 'ratio':
        val_hold['final_gap_pred'] = val_hold['final_gap_pred'].clip(RATIO_CLIP[0], RATIO_CLIP[1])
        val_hold['enplanements_pred'] = (val_hold['enplanements_counterfactual'] * (1.0 + val_hold['final_gap_pred'])).clip(lower=0)
    else:
        val_hold['enplanements_pred'] = (val_hold['enplanements_counterfactual'] + val_hold['final_gap_pred']).clip(lower=0)

    # 预测地板：避免 0 预测导致百分比指标爆炸
    val_hold['enplanements_pred'] = np.maximum(
        val_hold['enplanements_pred'],
        PRED_FLOOR_FRAC * val_hold['enplanements_counterfactual']
    )

    # ---- Metrics ----
    y_true = val_hold['enplanements']
    y_pred = val_hold['enplanements_pred']

    mape = safe_mape(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    wm = wmape(y_true, y_pred)
    med = median_ape(y_true, y_pred)

    # Top-30 专项
    val_top30 = val_hold[val_hold['airport_code_std'].isin(rank_pool)].copy()
    y_t30 = val_top30['enplanements']
    yhat_t30 = val_top30['enplanements_pred']
    wm_t30 = wmape(y_t30, yhat_t30) if len(val_top30) else float('nan')
    med_t30 = median_ape(y_t30, yhat_t30) if len(val_top30) else float('nan')

    print("-" * 55)
    print("Hold-out Results (2023–2024):")
    print(f"MAE : {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"WMAPE: {wm:.2f}%")
    print(f"MedAPE: {med:.2f}%")
    print(f"Top30 WMAPE: {wm_t30:.2f}%")
    print(f"Top30 MedAPE: {med_t30:.2f}%")
    print(f"R^2 : {r2:.4f}")
    print("-" * 55)

    # 保存结果
    val_out = val_hold[['year', 'airport_code_std', 'enplanements', 'enplanements_counterfactual', 'enplanements_pred']].copy()
    val_out.to_csv(output_dir / 'validation_results_2023_2024.csv', index=False)
    val_out[val_out['airport_code_std'].isin(rank_pool)].to_csv(output_dir / 'validation_results_top30_2023_2024.csv', index=False)

    metrics_txt = output_dir / 'validation_metrics_2023_2024.txt'
    with open(metrics_txt, 'w', encoding='utf-8') as f:
        f.write("Hold-out Validation (train<=2022, val=2023–2024)\n")
        f.write(f"MAE  : {mae:,.0f}\n")
        f.write(f"RMSE : {rmse:,.0f}\n")
        f.write(f"MAPE : {mape:.2f}%\n")
        f.write(f"WMAPE : {wm:.2f}%\n")
        f.write(f"MedAPE: {med:.2f}%\n")
        f.write(f"Top30 WMAPE: {wm_t30:.2f}%\n")
        f.write(f"Top30 MedAPE: {med_t30:.2f}%\n")
        f.write(f"R^2  : {r2:.4f}\n")
    print(f"已保存验证结果: {output_dir / 'validation_results_2023_2024.csv'}")
    print(f"已保存验证指标: {metrics_txt}")

# ==========================================
# 4. 核心可视化: 混合模型的表现
# ==========================================
# 画 Top 4 机场的 "预测 vs 真实"
top_airports = rank_pool[:4] if len(rank_pool) >= 4 else pool_list[:4]
plt.figure(figsize=(14, 10))

for i, airport in enumerate(top_airports):
    plt.subplot(2, 2, i+1)
    
    # 历史数据
    hist_data = full_df[(full_df['airport_code_std'] == airport) & (full_df['type'] == 'History')]
    # 预测点
    pred_data = full_df[(full_df['airport_code_std'] == airport) & (full_df['type'] == 'Forecast')]
    
    # 画线
    plt.plot(hist_data['year'], hist_data['enplanements'], 'b-o', label='Historical Data')
    plt.plot(hist_data['year'], hist_data['enplanements_counterfactual'], 'k--', alpha=0.3, label='Counterfactual Baseline')
    
    # 画 2025 预测点
    plt.scatter(pred_data['year'], pred_data['enplanements'], color='red', s=150, zorder=5, marker='*', label='2025 Forecast')
    
    # 连接线
    last_hist = hist_data.iloc[-1]
    curr_pred = pred_data.iloc[0]
    plt.plot([last_hist['year'], curr_pred['year']], [last_hist['enplanements'], curr_pred['enplanements']], 'r--', label='Projection')

    plt.title(f"{airport}: Hybrid Model Forecast", fontweight='bold')
    plt.grid(True, alpha=0.3)
    if i == 0: plt.legend()

plt.tight_layout()
plt.savefig(output_dir / 'Figure_Phase2_Forecast_Results.png', dpi=300, bbox_inches='tight')
plt.show()

# 特征重要性图 (XGBoost) - 论文里很有用
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='weight', title='XGBoost Feature Importance (Residual Correction)')
plt.tight_layout()
plt.savefig(output_dir / 'Figure_Phase2_FeatureImportance.png', dpi=300, bbox_inches='tight')
