"""
DIC-2: Multiple Linear Regression (CRISP-DM)
Problem 1: Kaggle 50 Startups  -> predict Profit
Problem 2: Boston Housing      -> predict medv

Usage:
    python mlr_analysis.py                      # run both problems
    python mlr_analysis.py --problem startups   # run one problem
    python mlr_analysis.py --url <csv_url> --target <column>   # any dataset
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = {
    "startups": {
        "name": "50 Startups",
        "url": "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv",
        "local": os.path.join(HERE, "data", "50_Startups.csv"),
        "target": "Profit",
    },
    "boston": {
        "name": "Boston Housing",
        "url": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        "local": os.path.join(HERE, "data", "BostonHousing.csv"),
        "target": "medv",
    },
}


# ---------- 1. Business / 2. Data Understanding ----------
def load(cfg):
    path = cfg.get("local")
    df = pd.read_csv(path if path and os.path.exists(path) else cfg["url"])
    print(f"\n=== {cfg['name']} ===  shape={df.shape}")
    print(df.describe().T.round(3))
    print("missing values:", int(df.isna().sum().sum()))
    return df


# ---------- 3. Data Preparation ----------
def prepare(df, target):
    df = df.dropna()
    y = df[target].astype(float)
    X = df.drop(columns=[target])
    # one-hot encode categorical columns, drop first level to avoid dummy-variable trap
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    return X, y


def ols_pvalues(X, y):
    """OLS with intercept; returns (p-values of features, adjusted R^2)."""
    A = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(A, y.values, rcond=None)
    resid = y.values - A @ beta
    n, k = A.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(A.T @ A)))
    p = 2 * stats.t.sf(np.abs(beta / se), n - k)
    r2 = 1 - resid @ resid / ((y.values - y.values.mean()) ** 2).sum()
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return pd.Series(p[1:], index=X.columns), adj


def backward_elimination(X, y, sl=0.05):
    """Remove the feature with the highest p-value until all p < sl."""
    cols = list(X.columns)
    steps = []
    while cols:
        pvals, adj_r2 = ols_pvalues(X[cols], y)
        worst = pvals.idxmax()
        if pvals[worst] <= sl:
            break
        steps.append({"removed": worst, "p_value": round(float(pvals[worst]), 4),
                      "adj_r2": round(float(adj_r2), 4)})
        cols.remove(worst)
    return cols, steps


# ---------- 4. Modeling / 5. Evaluation ----------
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "R2": round(float(r2_score(y_test, pred)), 4),
        "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
    }, pred


def run(key, cfg, outdir):
    df = load(cfg)
    target = cfg["target"]
    X, y = prepare(df, target)

    # correlation heatmap
    corr = pd.concat([X, y], axis=1).corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    fig.colorbar(im)
    ax.set_title(f"{cfg['name']} - correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{key}_corr.png"), dpi=110)
    plt.close(fig)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    full = LinearRegression().fit(X_tr, y_tr)
    m_full, _ = evaluate(full, X_te, y_te)

    selected, steps = backward_elimination(X_tr, y_tr)
    sel = LinearRegression().fit(X_tr[selected], y_tr)
    m_sel, pred = evaluate(sel, X_te[selected], y_te)

    print("All features   :", m_full)
    print("Backward elim. :", selected, m_sel)

    # actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_te, pred, alpha=0.7)
    lo, hi = min(y_te.min(), pred.min()), max(y_te.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"{cfg['name']} - actual vs predicted (R2={m_sel['R2']})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{key}_pred.png"), dpi=110)
    plt.close(fig)

    return {
        "name": cfg["name"], "target": target, "n_rows": int(len(df)),
        "features": list(X.columns),
        "all_features": {"metrics": m_full,
                         "intercept": round(float(full.intercept_), 4),
                         "coef": {c: round(float(v), 4) for c, v in zip(X.columns, full.coef_)}},
        "backward_elimination": {"steps": steps, "selected": selected, "metrics": m_sel,
                                 "intercept": round(float(sel.intercept_), 4),
                                 "coef": {c: round(float(v), 4) for c, v in zip(selected, sel.coef_)}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=list(PROBLEMS), help="run only one problem")
    ap.add_argument("--url", help="any CSV URL")
    ap.add_argument("--target", help="target column (with --url; default = last column)")
    args = ap.parse_args()

    outdir = os.path.join(HERE, "results")
    os.makedirs(outdir, exist_ok=True)

    if args.url:
        df_head = pd.read_csv(args.url, nrows=1)
        jobs = {"custom": {"name": "Custom", "url": args.url,
                           "target": args.target or df_head.columns[-1]}}
    else:
        jobs = {args.problem: PROBLEMS[args.problem]} if args.problem else PROBLEMS

    summary = {k: run(k, cfg, outdir) for k, cfg in jobs.items()}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved figures and summary.json to {outdir}")


if __name__ == "__main__":
    main()
