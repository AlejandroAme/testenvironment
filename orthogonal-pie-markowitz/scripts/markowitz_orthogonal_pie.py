import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

TICKERS = {
    "IWQE": "IWQE.AS",
    "SPFF": "SPFF.DE",
    "4GLD": "4GLD.DE",
    "CMOE": "CMOE.DE",
    "BNKE": "BNKE.PA",
    "CBUX": "EMXC.L",
    "VGEK": "VAPX.L",
    "XDEX": "INFR.L",
    "H4Z6": "HMCH.L",
    "BTIC": "BTIC.L",
    "DTLE": "DTLE.L",
    "URNU": "URNU.L",
    "FUC": "FUC.F",
    "34U": "ULTA",
}

CURRENT_WEIGHTS = pd.Series({
    "IWQE": 0.18,
    "SPFF": 0.15,
    "4GLD": 0.12,
    "CMOE": 0.10,
    "VGEK": 0.06,
    "BNKE": 0.06,
    "CBUX": 0.06,
    "XDEX": 0.06,
    "H4Z6": 0.04,
    "BTIC": 0.04,
    "DTLE": 0.04,
    "URNU": 0.03,
    "FUC": 0.03,
    "34U": 0.03,
})

RISK_FREE_RATE = 0.02
START = "2021-01-01"
N_RANDOM_PORTFOLIOS = 7000
OUTPUT_DIR = Path("docs/orthogonal-pie-markowitz")

SPECIAL_MAX = {
    "BTIC": 0.05,
    "URNU": 0.05,
    "FUC": 0.05,
    "34U": 0.05,
}
DEFAULT_MAX = 0.25


def portfolio_stats(weights, mean_returns, cov_matrix):
    weights = np.asarray(weights)
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights.T @ cov_matrix @ weights))
    sharpe = float((ret - RISK_FREE_RATE) / vol) if vol else 0.0
    return ret, vol, sharpe


def negative_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_stats(weights, mean_returns, cov_matrix)[2]


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_stats(weights, mean_returns, cov_matrix)[1]


def optimize(mean_returns, cov_matrix):
    names = list(mean_returns.index)
    bounds = [(0.0, SPECIAL_MAX.get(name, DEFAULT_MAX)) for name in names]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.repeat(1 / len(names), len(names))

    max_sharpe = minimize(
        negative_sharpe,
        x0,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    min_var = minimize(
        portfolio_volatility,
        x0,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    if not max_sharpe.success:
        raise RuntimeError(f"Max Sharpe optimization failed: {max_sharpe.message}")
    if not min_var.success:
        raise RuntimeError(f"Minimum variance optimization failed: {min_var.message}")

    return max_sharpe.x, min_var.x


def download_prices():
    yahoo_tickers = list(TICKERS.values())
    inverse = {v: k for k, v in TICKERS.items()}
    raw = yf.download(yahoo_tickers, start=START, auto_adjust=True, progress=False, threads=True)["Close"]
    raw = raw.rename(columns=inverse)
    raw = raw.dropna(axis=1, how="all")
    # Keep only rows where all available assets have prices.
    raw = raw.dropna()
    if raw.empty:
        raise RuntimeError("No overlapping Yahoo Finance data after dropping missing values.")
    return raw


def build_frontier(mean_returns, cov_matrix, available):
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(N_RANDOM_PORTFOLIOS):
        # Dirichlet random weights, then clip key satellite max constraints and renormalize.
        w = rng.dirichlet(np.ones(len(available)))
        for i, name in enumerate(available):
            max_w = SPECIAL_MAX.get(name)
            if max_w is not None and w[i] > max_w:
                w[i] = max_w
        w = w / w.sum()
        ret, vol, sharpe = portfolio_stats(w, mean_returns, cov_matrix)
        rows.append({"return": ret, "volatility": vol, "sharpe": sharpe})
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices = download_prices()
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    mean_returns = monthly_returns.mean() * 12
    cov_matrix = monthly_returns.cov() * 12
    corr_matrix = monthly_returns.corr()

    available = list(mean_returns.index)
    current = CURRENT_WEIGHTS.loc[available]
    current = current / current.sum()

    max_sharpe_w, min_var_w = optimize(mean_returns, cov_matrix)

    current_stats = portfolio_stats(current.values, mean_returns, cov_matrix)
    max_sharpe_stats = portfolio_stats(max_sharpe_w, mean_returns, cov_matrix)
    min_var_stats = portfolio_stats(min_var_w, mean_returns, cov_matrix)

    frontier = build_frontier(mean_returns, cov_matrix, available)

    weights = pd.DataFrame({
        "current": current.values,
        "max_sharpe": max_sharpe_w,
        "minimum_variance": min_var_w,
    }, index=available)

    summary = pd.DataFrame({
        "return": [current_stats[0], max_sharpe_stats[0], min_var_stats[0]],
        "volatility": [current_stats[1], max_sharpe_stats[1], min_var_stats[1]],
        "sharpe": [current_stats[2], max_sharpe_stats[2], min_var_stats[2]],
    }, index=["current", "max_sharpe", "minimum_variance"])

    results = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "risk_free_rate": RISK_FREE_RATE,
        "start": START,
        "data_start": monthly_returns.index.min().strftime("%Y-%m-%d"),
        "data_end": monthly_returns.index.max().strftime("%Y-%m-%d"),
        "tickers": TICKERS,
        "available_assets": available,
        "summary": summary.round(6).to_dict(orient="index"),
        "weights": weights.round(6).to_dict(orient="index"),
        "annual_returns": mean_returns.round(6).to_dict(),
        "correlation": corr_matrix.round(4).to_dict(),
        "frontier": frontier,
        "notes": [
            "Returns are calculated from Yahoo Finance adjusted close prices using monthly returns.",
            "Optimization constraints: no shorts; 25% max per asset; BTIC, URNU, FUC and 34U capped at 5% each.",
            "Mixed listing currencies can affect an EUR investor's realized return. Prefer EUR-listed tickers where available for a cleaner EUR analysis.",
        ],
    }

    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with pd.ExcelWriter(OUTPUT_DIR / "markowitz_results.xlsx") as writer:
        prices.to_excel(writer, sheet_name="Daily prices")
        monthly_returns.to_excel(writer, sheet_name="Monthly returns")
        summary.to_excel(writer, sheet_name="Summary")
        weights.to_excel(writer, sheet_name="Weights")
        mean_returns.to_excel(writer, sheet_name="Annual returns")
        cov_matrix.to_excel(writer, sheet_name="Covariance")
        corr_matrix.to_excel(writer, sheet_name="Correlation")

    print(summary)
    print(weights)
    print(f"Wrote {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
