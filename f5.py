import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

import matplotlib.pyplot as plt

try:
    import shap
except ImportError as e:
    shap = None


def compute_feature_importance_gain(model: XGBRegressor, feature_names):
    """
    用 gain 作为重要性，更适合论文解释（比 split/weight 更常用）。
    """
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")  # dict: {feature: gain}
    # xgboost 可能只返回出现过的特征，补齐为 0
    rows = []
    for f in feature_names:
        rows.append((f, float(score.get(f, 0.0))))
    imp = pd.DataFrame(rows, columns=["feature", "gain_importance"])
    imp = imp.sort_values("gain_importance", ascending=False).reset_index(drop=True)
    return imp


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="both", colors="#333333")
    ax.grid(True, which="major", axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(False, axis="x")


def main():
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir.parent

    fig_dir = ensure_dir(base_dir / "figures")
    table_dir = ensure_dir(base_dir / "tables")

    data_path_out = base_dir / "data_phase2_input.csv"
    data_path_root = root_dir / "data_phase2_input.csv"
    data_path = data_path_out if data_path_out.exists() else data_path_root

    df = pd.read_csv(data_path)
    df = df[df["year"] <= 2024].dropna(subset=["gap_to_trend"]).copy()

    df = df.sort_values(["airport_code_std", "year"])
    df["gap_lag1"] = df.groupby("airport_code_std")["gap_to_trend"].shift(1)
    df["gap_lag1"] = df["gap_lag1"].fillna(0)

    svr_features = ["covid_stringency", "year"]
    X_svr = df[svr_features]
    y_svr = df["gap_to_trend"]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_svr_scaled = scaler_X.fit_transform(X_svr)
    y_svr_scaled = scaler_y.fit_transform(y_svr.values.reshape(-1, 1)).ravel()

    svr_model = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)
    svr_model.fit(X_svr_scaled, y_svr_scaled)
    df["svr_gap_pred"] = scaler_y.inverse_transform(
        svr_model.predict(X_svr_scaled).reshape(-1, 1)
    ).ravel()
    df["residual"] = df["gap_to_trend"] - df["svr_gap_pred"]

    le = LabelEncoder()
    df["airport_idx"] = le.fit_transform(df["airport_code_std"])

    xgb_features = [
        "gdp_real_trillion",
        "fuel_price_dollar",
        "income_capita_real",
        "population_million",
        "gap_lag1",
        "airport_idx",
    ]

    train_df = df.dropna(subset=xgb_features + ["residual"]).copy()
    if len(train_df) < 50:
        raise ValueError("可训练样本太少，请检查数据完整性。")

    X = train_df[xgb_features]
    y = train_df["residual"]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    model.fit(X, y)

    imp = compute_feature_importance_gain(model, xgb_features)
    out_csv = table_dir / "fig5_feature_importance.csv"
    imp.to_csv(out_csv, index=False)

    imp_plot = imp.copy()
    total_gain = imp_plot["gain_importance"].sum()
    imp_plot["gain_pct"] = imp_plot["gain_importance"] / total_gain * 100 if total_gain > 0 else 0.0

    name_map = {
        "gap_lag1": "Lagged gap (t-1)",
        "fuel_price_dollar": "Fuel price ($)",
        "population_million": "Population (M)",
        "income_capita_real": "Real income per capita",
        "gdp_real_trillion": "Real GDP (T)",
        "airport_idx": "Airport index",
    }
    imp_plot["feature_label"] = imp_plot["feature"].map(name_map).fillna(imp_plot["feature"])

    topk = min(15, len(imp_plot))
    imp_top = imp_plot.head(topk).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp_top["feature_label"], imp_top["gain_pct"])
    ax.set_xlabel("Gain importance (%)")
    ax.set_title("Figure 5. XGBoost feature importance (residual correction)")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure5_XGBoost_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_csv)
    print("Saved:", fig_dir / "Figure5_XGBoost_importance.png")

    train_df = train_df.copy()
    train_df["xgb_resid_pred"] = model.predict(train_df[xgb_features])
    train_df["gap_pred_final"] = train_df["svr_gap_pred"] + train_df["xgb_resid_pred"]

    atl_df = train_df[train_df["airport_code_std"] == "ATL"].sort_values("year").copy()
    atl_df = atl_df[(atl_df["year"] >= 2000) & (atl_df["year"] <= 2024)]
    atl_df = atl_df.dropna(subset=["gap_to_trend", "svr_gap_pred", "xgb_resid_pred", "gap_pred_final"])

    fig6_csv = table_dir / "fig6_atl_gap_fit.csv"
    atl_df[["year", "gap_to_trend", "svr_gap_pred", "xgb_resid_pred", "gap_pred_final"]].to_csv(fig6_csv, index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(atl_df["year"], atl_df["gap_to_trend"], linewidth=2.6, label="Actual gap")
    ax1.plot(atl_df["year"], atl_df["svr_gap_pred"], linestyle="--", linewidth=2.2, label="SVR trend")
    ax1.plot(atl_df["year"], atl_df["gap_pred_final"], linewidth=2.4, label="SVR + XGB (final)")
    ax1.axvspan(2020, 2022, alpha=0.15, color="#9e9e9e")
    ax1.set_ylabel("Absolute gap")
    ax1.set_title("Figure 6. ATL gap modeling: SVR trend + XGBoost residual correction")
    ax1.legend(frameon=False, ncol=1, loc="best")
    style_axes(ax1)

    ax2.plot(atl_df["year"], atl_df["xgb_resid_pred"], linewidth=2.2, label="XGB residual correction")
    ax2.axhline(0, color="#666666", linewidth=1.2, alpha=0.9)
    ax2.axvspan(2020, 2022, alpha=0.15, color="#9e9e9e")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("XGB residual correction")
    style_axes(ax2)

    fig.tight_layout()
    fig.savefig(fig_dir / "Figure6_ATL_Gap_Fit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", fig6_csv)
    print("Saved:", fig_dir / "Figure6_ATL_Gap_Fit.png")

    if shap is None:
        raise ImportError("Package 'shap' is required for Figure 7. Install it with: pip install shap")

    X_shap = X.copy()
    if len(X_shap) > 5000:
        X_shap = X_shap.sample(n=5000, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title("Figure 7. SHAP summary for XGBoost residual correction", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure7_SHAP_Summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", fig_dir / "Figure7_SHAP_Summary.png")


if __name__ == "__main__":
    main()
