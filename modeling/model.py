"""聚类 + 回归建模（纯逻辑，不含可视化；可视化在 test_clustering.ipynb）。

流程对应 booklet 的「先 k-means 聚类，再回归」：
1. 载入 cleaned_data.csv，过滤有目标(i_corr)且有成分的行；成分 NaN→0、目标取 log10。
2. 在「有变化的成分元素」上做标准化 + k-means，用轮廓系数选 k。
3. 簇标签(one-hot) + 环境(Cl_M/pH/Temp_C) 作为主模型特征（成分经 k-means 压缩成
   材料族，见 build_features），回归预测 log10(i_corr)。
4. 5 折交叉验证 + 训练/测试切分，给出 R² / RMSE / 特征重要性。

新增数据后只需重跑 data_cleaner/clean_data.py，再重跑本文件（或直接 Run All ipynb）。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split

CODES = Path(__file__).resolve().parent.parent
CLEAN = CODES / "cleaned_data.csv"

ELEMENTS = ["B", "C", "N", "Mg", "Al", "Si", "P", "S", "Ti", "V", "Cr", "Mn",
            "Fe", "Co", "Ni", "Cu", "Y", "Nb", "Mo", "Ce", "Gd", "Ta", "W", "Re"]
ENV_FEATURES = ["Cl_M", "pH", "Temp_C"]
TARGET = "I_corr_uA_per_cm2"
LOG_TARGET = "log_i_corr"


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_clean_data(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or CLEAN, encoding="utf-8")
    df = df[df[TARGET].notna()].copy()
    df = df[df[ELEMENTS].notna().any(axis=1)].copy()  # 至少一个元素已知，成分才有意义
    df[ELEMENTS] = df[ELEMENTS].fillna(0.0)           # 未添加元素 = 0
    df[LOG_TARGET] = np.log10(df[TARGET])
    return df


def variable_elements(df) -> list:
    """数据里真正出现过的元素（max>0），用于聚类/回归，避开零方差列。"""
    return [e for e in ELEMENTS if df[e].max() > 1e-9]


# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------
def cluster_composition(df, k, ve=None):
    """在标准化成分上做 k-means，返回 (labels, km, scaler, X_scaled, ve)。"""
    ve = ve or variable_elements(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[ve].values)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km, scaler, X_scaled, ve


def pick_k(df, k_range=range(2, 10)) -> pd.DataFrame:
    """肘部(惯量) + 轮廓系数，供选 k。"""
    ve = variable_elements(df)
    X = StandardScaler().fit_transform(df[ve].values)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
        })
    return pd.DataFrame(rows)


def cluster_profile(df, labels, ve=None) -> pd.DataFrame:
    """每个簇的平均成分(wt%) + 样本数，用于解释簇（对照 Material class）。"""
    ve = ve or variable_elements(df)
    prof = df.copy()
    prof["cluster"] = labels
    return prof.groupby("cluster")[ve].mean()


# ---------------------------------------------------------------------------
# 回归
# ---------------------------------------------------------------------------
def build_features(df, labels, scaler, ve):
    """构造回归特征 X 与目标 y。

    主模型特征 = 簇标签(one-hot) + 环境(Cl_M/pH/Temp_C)。成分经 k-means 压缩成
    6 个材料族（簇 one-hot），用于降维、加快训练（RF 训练时间约 -20%）并增强可
    解释性；scaler/ve 仅保留接口兼容，主特征不再用连续成分。也不含 E_corr_mV
    （它和 i_corr 同出自一条极化曲线，用了等于偷看答案）。
    """
    cl = pd.get_dummies(pd.Series(labels, index=df.index).astype(int),
                        prefix="cluster").astype(float)
    env = df[ENV_FEATURES].fillna(df[ENV_FEATURES].median())
    X = pd.concat([cl, env], axis=1)
    return X, df[LOG_TARGET]


def build_comp_features(df, scaler, ve):
    """构造成分+环境特征 X 与目标 y（最终模型用）。

    与 build_features（簇 one-hot）相对：直接用标准化连续成分 + 环境，特征更全
    （精度更高，见消融 0.736 vs 0.650），代价是维度更高、训练稍慢。
    """
    Xc = pd.DataFrame(scaler.transform(df[ve].values), columns=ve, index=df.index)
    env = df[ENV_FEATURES].fillna(df[ENV_FEATURES].median())
    X = pd.concat([Xc, env], axis=1)
    return X, df[LOG_TARGET]


def train_eval(df, labels, scaler, ve, make_model, test_size=0.2, seed=42,
               features="cluster") -> dict:
    """训练 + 评估一个回归器，返回模型、预测、指标、特征重要性。

    features="cluster" 用簇 one-hot + 环境（主模型/聚类方法）；
    features="comp" 用连续成分 + 环境（最终结果）。
    """
    if features == "comp":
        X, y = build_comp_features(df, scaler, ve)
    else:
        X, y = build_features(df, labels, scaler, ve)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)

    model = make_model()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    cv = cross_val_score(make_model(), X, y,
                         cv=KFold(5, shuffle=True, random_state=seed), scoring="r2")

    imp = (pd.Series(model.feature_importances_, index=X.columns)
           .sort_values(ascending=False))
    return {
        "model": model,
        "X_test": Xte, "y_test": yte, "pred": pred,
        "test_r2": float(r2_score(yte, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
        "feature_importances": imp,
        "n_samples": len(X),
        "n_features": X.shape[1],
    }


# ---------------------------------------------------------------------------
# 特征消融
# ---------------------------------------------------------------------------
def ablate(df, labels, scaler, ve) -> pd.DataFrame:
    """特征消融：对比不同特征组的 CV R²（5 折，随机森林）。

    分组：
    - 仅 E_corr：E_corr 单独（预期无用/负 R²）→ 说明 E_corr 不适合作为特征；
    - 仅簇 / 仅成分 / 仅环境：单类特征各自的信号；
    - 簇+环境 vs 成分+环境：簇是成分的压缩表示，两者越接近说明聚类压缩几乎不丢信息。
    """
    Xc = pd.DataFrame(scaler.transform(df[ve].values), columns=ve, index=df.index)
    cl = pd.get_dummies(pd.Series(labels, index=df.index).astype(int),
                        prefix="cluster").astype(float)
    env = df[ENV_FEATURES].fillna(df[ENV_FEATURES].median())
    ec = df[["E_corr_mV"]].fillna(df["E_corr_mV"].median())
    y = df[LOG_TARGET]

    groups = {
        "仅 E_corr": ec,
        "仅簇": cl,
        "仅成分": Xc,
        "仅环境": env,
        "簇+环境": pd.concat([cl, env], axis=1),
        "成分+环境": pd.concat([Xc, env], axis=1),
    }

    def _cv_r2(X):
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        s = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
        return float(s.mean()), float(s.std())

    rows = []
    for name, X in groups.items():
        m, s = _cv_r2(X)
        rows.append({"特征组": name, "特征数": X.shape[1], "CV R²": round(m, 3), "±": round(s, 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 一键编排
# ---------------------------------------------------------------------------
def run_pipeline(k=None) -> dict:
    df = load_clean_data()
    ve = variable_elements(df)
    k_range_df = pick_k(df)
    if k is None:
        # 轮廓系数随 k 单调增（数据里合金种类多、分布散），取边缘最大值会过聚类；
        # 这里按材料族数量固定取 6（钢 / Ni-Cu / Ti / Al / HEA 等）。ipynb 里会画出曲线。
        k = 6
    labels, km, scaler, X_scaled, ve = cluster_composition(df, k, ve)

    rf = train_eval(df, labels, scaler, ve,
                    lambda: RandomForestRegressor(n_estimators=300, random_state=42))
    gb = train_eval(df, labels, scaler, ve,
                    lambda: GradientBoostingRegressor(random_state=42))

    return {
        "df": df, "ve": ve, "k": k, "k_range_df": k_range_df,
        "labels": labels, "X_scaled": X_scaled,
        "profile": cluster_profile(df, labels, ve),
        "rf": rf, "gb": gb,
    }


if __name__ == "__main__":
    res = run_pipeline()
    print(f"样本 {res['rf']['n_samples']} 行，特征 {res['rf']['n_features']} 列，聚类 k={res['k']}")
    print("选 k 依据（惯量 / 轮廓系数）：")
    print(res["k_range_df"].round(4).to_string(index=False))
    print(f"RF  测试 R2={res['rf']['test_r2']:.3f}  RMSE={res['rf']['test_rmse']:.3f}  CV R2={res['rf']['cv_r2_mean']:.3f}±{res['rf']['cv_r2_std']:.3f}")
    print(f"GB  测试 R2={res['gb']['test_r2']:.3f}  RMSE={res['gb']['test_rmse']:.3f}  CV R2={res['gb']['cv_r2_mean']:.3f}±{res['gb']['cv_r2_std']:.3f}")
    print("\n各簇样本数与平均成分(wt%)：")
    print(res["profile"].round(1))
