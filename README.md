# Investment Club
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from io import StringIO
import requests

try:
    import yfinance as yf
except Exception:
    yf = None


MAG7_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

# ON/OFF switches
RUN_MAG7_COMPARE = True

SECTOR_UNIVERSES: Dict[str, List[str]] = {}
if RUN_MAG7_COMPARE:
    SECTOR_UNIVERSES["Magnificent_7"] = MAG7_TICKERS
FOCUS_TICKER = "AAPL"
MARKET_TICKER = "^GSPC"
INDUSTRY_ETFS = ["XLK", "XLC", "QQQ"]
STOCK_FULL_NAMES = {
    "AAPL": "Apple (AAPL)",
    "MSFT": "Microsoft (MSFT)",
    "NVDA": "NVIDIA (NVDA)",
    "AMZN": "Amazon (AMZN)",
    "GOOGL": "Alphabet (GOOGL)",
    "META": "Meta (META)",
    "TSLA": "Tesla (TSLA)",
}
ETF_FULL_NAMES = {
    "XLK": "Technology Select Sector SPDR (XLK)",
    "XLC": "Communication Services Select Sector SPDR (XLC)",
    "QQQ": "Invesco QQQ Trust (QQQ)",
}
VALUATION_TICKERS = MAG7_TICKERS
SPECIAL_COMPARE_TICKERS = MAG7_TICKERS
SPECIAL_COMPARE_LABEL = "Magnificent 7 Mean-Variance Universe"
START = "2020-01-01"
END = "2026-02-01"  # yfinance end is exclusive
RISK_FREE_RATE = 0.04  # annualized, 4%
N_PORTFOLIOS = 50000
SEED = 42
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _to_stooq_symbol(ticker: str) -> str:
    return f"{ticker.lower().replace('.', '-')}.us"


def _yahoo_chart_close(ticker: str, start: str, end: str, timeout: int = 10) -> pd.Series:
    p1 = int(pd.Timestamp(start).timestamp())
    # Yahoo period2 is exclusive; add one day to include `end`.
    p2 = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    r = requests.get(url, timeout=timeout, headers=HTTP_HEADERS)
    r.raise_for_status()
    data = r.json()
    result = data.get("chart", {}).get("result")
    if not result:
        err = data.get("chart", {}).get("error")
        raise RuntimeError(f"No Yahoo chart result for {ticker}: {err}")

    result0 = result[0]
    ts = result0.get("timestamp") or []
    if not ts:
        raise RuntimeError(f"No Yahoo timestamps for {ticker}")

    inds = result0.get("indicators", {})
    adj = (inds.get("adjclose") or [{}])[0].get("adjclose")
    close = (inds.get("quote") or [{}])[0].get("close")
    prices = adj if adj is not None else close
    if prices is None:
        raise RuntimeError(f"No Yahoo close series for {ticker}")

    s = pd.Series(prices, index=pd.to_datetime(ts, unit="s"), dtype="float64").dropna().sort_index()
    if s.empty:
        raise RuntimeError(f"Empty Yahoo close series for {ticker}")
    return s


def _stooq_daily_close(symbol: str, start: str, end: str, timeout: int = 10) -> pd.Series:
    d1 = pd.Timestamp(start).strftime("%Y%m%d")
    d2 = pd.Timestamp(end).strftime("%Y%m%d")
    urls = [
        f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d",
        f"https://stooq.com/q/d/l/?s={symbol}&i=d",
        f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=w",
        f"https://stooq.com/q/d/l/?s={symbol}&i=w",
    ]

    last_err = None
    for url in urls:
        for _ in range(2):
            try:
                r = requests.get(url, timeout=timeout, headers=HTTP_HEADERS)
                r.raise_for_status()
                if "<html" in r.text.lower():
                    raise RuntimeError("HTML response (likely blocked/non-data response)")

                # Handle both comma and semicolon CSV variants.
                df = pd.read_csv(StringIO(r.text), sep=r"[;,]", engine="python")
                if df.empty:
                    raise RuntimeError("Empty CSV")

                norm = {c: str(c).strip().lower() for c in df.columns}
                df = df.rename(columns=norm)
                date_col = "date" if "date" in df.columns else (df.columns[0] if len(df.columns) > 0 else None)
                close_col = None
                for c in ["close", "zamkniecie", "zamknięcie"]:
                    if c in df.columns:
                        close_col = c
                        break
                if date_col is None or close_col is None:
                    raise RuntimeError("Missing date/close columns")

                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
                s = df.dropna(subset=[date_col, close_col]).set_index(date_col)[close_col].sort_index()
                if not s.empty:
                    return s
                raise RuntimeError("No usable rows after cleaning")
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"No usable Stooq data for {symbol}: {last_err}")


def _stooq_market_proxy(start: str, end: str) -> pd.Series:
    # Try several liquid US proxies; some symbols can be unavailable intermittently.
    for sym in ["spy.us", "spx.us", "ivv.us", "voo.us", "dia.us", "qqq.us"]:
        try:
            return _stooq_daily_close(sym, start, end)
        except Exception:
            continue
    raise RuntimeError("No usable Stooq market proxy found.")


def _yahoo_quote_summary_value(data: dict, section: str, field: str):
    v = data.get(section, {}).get(field)
    if isinstance(v, dict):
        return v.get("raw")
    return v


def fetch_valuation_metrics(ticker: str, timeout: int = 10) -> dict:
    yfin_err = None
    if yf is not None:
        try:
            info = yf.Ticker(ticker).info
            out = {
                "ticker": ticker,
                "forward_pe": info.get("forwardPE"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "fcf_yield": None,
            }
            market_cap = info.get("marketCap")
            fcf = info.get("freeCashflow")
            if market_cap not in [None, 0] and fcf is not None:
                out["fcf_yield"] = fcf / market_cap
            if any(v is not None for k, v in out.items() if k != "ticker"):
                return out
            yfin_err = "yfinance returned empty valuation fields"
        except Exception as e:
            yfin_err = str(e)

    http_err = None
    try:
        url = (
            f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
            "?modules=summaryDetail,defaultKeyStatistics,financialData"
        )
        r = requests.get(url, timeout=timeout, headers=HTTP_HEADERS)
        r.raise_for_status()
        js = r.json()
        result = js.get("quoteSummary", {}).get("result")
        if not result:
            err = js.get("quoteSummary", {}).get("error")
            raise RuntimeError(f"No quote summary for {ticker}: {err}")

        q = result[0]
        return {
            "ticker": ticker,
            "forward_pe": _yahoo_quote_summary_value(q, "summaryDetail", "forwardPE"),
            "price_to_sales": _yahoo_quote_summary_value(q, "summaryDetail", "priceToSalesTrailing12Months"),
            "ev_to_ebitda": _yahoo_quote_summary_value(q, "defaultKeyStatistics", "enterpriseToEbitda"),
            "fcf_yield": None,
        }
    except Exception as e:
        http_err = str(e)

    raise RuntimeError(f"yfinance: {yfin_err}; http: {http_err}")


def build_valuation_comps_table(tickers):
    rows = []
    for t in tickers:
        try:
            rows.append(fetch_valuation_metrics(t))
        except Exception as e:
            print(f"[WARN] Valuation metrics unavailable for {t}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_valuation_comps(df: pd.DataFrame):
    metrics = [
        ("forward_pe", "Forward P/E"),
        ("price_to_sales", "Price / Sales"),
        ("fcf_yield", "FCF Yield (EV/FCF)"),
        ("ev_to_ebitda", "EV / EBITDA"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    axes = axes.ravel()
    if df.empty:
        for i, (_, title) in enumerate(metrics):
            ax = axes[i]
            ax.set_title(title)
            ax.text(0.5, 0.5, "No valuation data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    else:
        tickers = df["ticker"].tolist()
        x = np.arange(len(tickers))

        for i, (col, title) in enumerate(metrics):
            ax = axes[i]
            vals = pd.to_numeric(df[col], errors="coerce").values
            colors = ["#DA291C" if t == FOCUS_TICKER else "#4C78A8" for t in tickers]
            ax.bar(x, np.nan_to_num(vals, nan=0.0), color=colors, alpha=0.9)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(tickers)
            ax.grid(axis="y", alpha=0.25)
            for j, v in enumerate(vals):
                if np.isfinite(v):
                    label = f"{v:.1%}" if col == "fcf_yield" else f"{v:.1f}"
                    ax.text(j, v, label, ha="center", va="bottom", fontsize=8)

    fig.suptitle("Valuation Comparables: Magnificent 7", fontsize=13)
    plt.tight_layout()
    fig.savefig("valuation_comparables.png", dpi=150, bbox_inches="tight")
    plt.show()


def download_returns(tickers, market_ticker, start, end, min_assets=3):
    def _fetch_single_fallback(ticker):
        errors = []
        try:
            return _yahoo_chart_close(ticker, start, end), "yahoo"
        except Exception as e:
            errors.append(f"yahoo: {e}")

        candidates = [_to_stooq_symbol(ticker), ticker.lower(), ticker.lower().replace(".", "-"), f"{ticker.lower()}.us"]
        for cand in candidates:
            try:
                return _stooq_daily_close(cand, start, end), f"stooq:{cand}"
            except Exception as e:
                errors.append(f"stooq:{cand}: {e}")

        raise RuntimeError(" | ".join(errors))

    if yf is None:
        print("[WARN] yfinance not available; using Yahoo Chart API fallback (Stooq backup).")
        close = {}
        failed_reasons = {}
        for t in tickers:
            try:
                s, _ = _fetch_single_fallback(t)
                close[t] = s
            except Exception as e:
                failed_reasons[t] = str(e)
        close = pd.DataFrame(close).dropna(how="all")
        ret = close.pct_change().dropna(how="all")

        available = [t for t in tickers if t in ret.columns]
        missing = sorted(set(tickers) - set(available))
        if missing:
            print(f"[WARN] Missing tickers excluded: {missing}")
            for t in missing:
                if t in failed_reasons:
                    print(f"[WARN] {t} fetch failure: {failed_reasons[t]}")
        if len(available) < min_assets:
            raise RuntimeError(f"Not enough available assets after download. Need >= {min_assets}, got {len(available)}")

        asset_ret = ret[available].dropna()
        try:
            try:
                mkt_close = _yahoo_chart_close(market_ticker, start, end)
            except Exception:
                mkt_close = _stooq_market_proxy(start, end)
            mkt_ret = mkt_close.pct_change().dropna().reindex(asset_ret.index).dropna()
            asset_ret = asset_ret.reindex(mkt_ret.index).dropna()
            if len(asset_ret) == 0:
                raise RuntimeError("No overlapping dates after aligning market and asset returns.")
        except Exception:
            # Last resort: use equal-weight asset basket as market proxy.
            print("[WARN] Market proxy download failed; using equal-weight asset basket as market proxy.")
            mkt_ret = asset_ret.mean(axis=1)
        return asset_ret, mkt_ret

    symbols = tickers + [market_ticker]
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("No data downloaded from Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        else:
            close = raw.xs("Close", axis=1, level=0, drop_level=True).copy()
    else:
        close = raw.copy()

    close = close.dropna(how="all")
    close = close[[c for c in symbols if c in close.columns]]

    # Retry any missing asset tickers individually using the non-yfinance fallback path.
    retry_failed = {}
    for t in tickers:
        if t not in close.columns:
            try:
                s, src = _fetch_single_fallback(t)
                close[t] = s
                print(f"[INFO] Recovered missing ticker {t} via {src}")
            except Exception as e:
                retry_failed[t] = str(e)

    ret = close.pct_change().dropna(how="all")

    available = [t for t in tickers if t in ret.columns]
    missing = sorted(set(tickers) - set(available))
    if missing:
        print(f"[WARN] Missing tickers excluded: {missing}")
        for t in missing:
            if t in retry_failed:
                print(f"[WARN] {t} fetch failure: {retry_failed[t]}")
    if len(available) < min_assets:
        raise RuntimeError(f"Not enough available assets after download. Need >= {min_assets}, got {len(available)}")
    if market_ticker not in ret.columns:
        raise RuntimeError(f"Missing market ticker in downloaded data: {market_ticker}")

    asset_ret = ret[available].dropna()
    mkt_ret = ret[market_ticker].reindex(asset_ret.index).dropna()
    asset_ret = asset_ret.reindex(mkt_ret.index).dropna()
    return asset_ret, mkt_ret


def simulate_portfolios(mu, cov, rf, n_portfolios=50000, seed=42):
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    w = rng.dirichlet(np.ones(n_assets), size=n_portfolios)
    port_ret = w @ mu.values
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov.values, w))
    sharpe = np.where(port_vol > 0, (port_ret - rf) / port_vol, np.nan)
    return w, port_ret, port_vol, sharpe


def efficient_frontier_from_cloud(port_vol, port_ret):
    order = np.argsort(port_vol)
    vol_sorted = port_vol[order]
    ret_sorted = port_ret[order]
    frontier_ret = np.maximum.accumulate(ret_sorted)
    improve = np.r_[True, frontier_ret[1:] > frontier_ret[:-1] + 1e-10]
    return vol_sorted[improve], frontier_ret[improve]


def analyze_sector(sector_name, tickers):
    asset_ret, mkt_ret = download_returns(tickers, MARKET_TICKER, START, END)
    tickers_used = asset_ret.columns.tolist()
    mu = asset_ret.mean() * 252.0
    cov = asset_ret.cov() * 252.0

    w, port_ret, port_vol, sharpe = simulate_portfolios(
        mu, cov, RISK_FREE_RATE, n_portfolios=N_PORTFOLIOS, seed=SEED
    )
    i_tan = int(np.nanargmax(sharpe))
    w_tan = w[i_tan]
    ret_tan = float(port_ret[i_tan])
    vol_tan = float(port_vol[i_tan])
    sharpe_tan = float(sharpe[i_tan])

    f_vol, f_ret = efficient_frontier_from_cloud(port_vol, port_ret)

    asset_vol = np.sqrt(np.diag(cov.values))
    asset_ann = mu.values

    market_ann = float(mkt_ret.mean() * 252.0)
    market_var = float(mkt_ret.var() * 252.0)
    betas = np.array([float(asset_ret[t].cov(mkt_ret) * 252.0 / market_var) if market_var > 0 else np.nan for t in tickers_used])
    capm_expected = RISK_FREE_RATE + betas * (market_ann - RISK_FREE_RATE)

    b_min = float(np.nanmin(betas) - 0.2)
    b_max = float(np.nanmax(betas) + 0.2)
    beta_line = np.linspace(b_min, b_max, 200)
    sml_y = RISK_FREE_RATE + beta_line * (market_ann - RISK_FREE_RATE)
    x_max = max(float(np.nanmax(port_vol)), float(np.nanmax(asset_vol))) * 1.15
    cml_x = np.linspace(0.0, x_max, 200)
    cml_y = RISK_FREE_RATE + sharpe_tan * cml_x

    summary = pd.DataFrame(
        {
            "ticker": tickers_used,
            "ann_return": asset_ann,
            "ann_vol": asset_vol,
            "beta_vs_sp500": betas,
            "capm_return": capm_expected,
            "alpha_actual_minus_capm": asset_ann - capm_expected,
        }
    ).sort_values("ticker")

    w_df = pd.DataFrame({"ticker": tickers_used, "weight_tangency": w_tan}).sort_values("weight_tangency", ascending=False)
    focus_row = summary[summary["ticker"] == FOCUS_TICKER]

    # Industry ETF reference points (for plotting context only, not optimization universe).
    etf_ret = pd.DataFrame()
    try:
        etf_ret_raw, _ = download_returns(INDUSTRY_ETFS, MARKET_TICKER, START, END, min_assets=1)
        common_idx = asset_ret.index.intersection(etf_ret_raw.index)
        if len(common_idx) > 10:
            etf_ret = etf_ret_raw.reindex(common_idx).dropna(how="all")
    except Exception:
        etf_ret = pd.DataFrame()

    etf_tickers = etf_ret.columns.tolist()
    if etf_tickers:
        etf_ann = (etf_ret.mean() * 252.0).reindex(etf_tickers).values
        etf_vol = (etf_ret.std(ddof=1) * np.sqrt(252.0)).reindex(etf_tickers).values
        etf_betas = np.array(
            [
                float(etf_ret[t].cov(mkt_ret.reindex(etf_ret.index)) * 252.0 / market_var)
                if market_var > 0
                else np.nan
                for t in etf_tickers
            ]
        )
        etf_capm = RISK_FREE_RATE + etf_betas * (market_ann - RISK_FREE_RATE)
    else:
        etf_ann = np.array([])
        etf_vol = np.array([])
        etf_betas = np.array([])
        etf_capm = np.array([])

    return {
        "sector": sector_name,
        "tickers": tickers_used,
        "asset_ret": asset_ret,
        "summary": summary,
        "weights": w_df,
        "port_vol": port_vol,
        "port_ret": port_ret,
        "sharpe": sharpe,
        "frontier_vol": f_vol,
        "frontier_ret": f_ret,
        "tangency_vol": vol_tan,
        "tangency_ret": ret_tan,
        "tangency_sharpe": sharpe_tan,
        "asset_vol": asset_vol,
        "asset_ann": asset_ann,
        "betas": betas,
        "capm_expected": capm_expected,
        "beta_line": beta_line,
        "sml_y": sml_y,
        "cml_x": cml_x,
        "cml_y": cml_y,
        "etf_tickers": etf_tickers,
        "etf_ann": etf_ann,
        "etf_vol": etf_vol,
        "etf_betas": etf_betas,
        "etf_capm": etf_capm,
        "focus_row": focus_row,
    }


def analyze_frontier_universe(title, tickers):
    asset_ret, _ = download_returns(tickers, MARKET_TICKER, START, END, min_assets=4)
    tickers_used = asset_ret.columns.tolist()
    mu = asset_ret.mean() * 252.0
    cov = asset_ret.cov() * 252.0

    w, port_ret, port_vol, sharpe = simulate_portfolios(
        mu, cov, RISK_FREE_RATE, n_portfolios=N_PORTFOLIOS, seed=SEED
    )
    i_tan = int(np.nanargmax(sharpe))
    ret_tan = float(port_ret[i_tan])
    vol_tan = float(port_vol[i_tan])
    sharpe_tan = float(sharpe[i_tan])
    f_vol, f_ret = efficient_frontier_from_cloud(port_vol, port_ret)

    asset_vol = np.sqrt(np.diag(cov.values))
    asset_ann = mu.values
    x_max = max(float(np.nanmax(port_vol)), float(np.nanmax(asset_vol))) * 1.15
    cml_x = np.linspace(0.0, x_max, 200)
    cml_y = RISK_FREE_RATE + sharpe_tan * cml_x

    return {
        "title": title,
        "tickers": tickers_used,
        "port_vol": port_vol,
        "port_ret": port_ret,
        "sharpe": sharpe,
        "frontier_vol": f_vol,
        "frontier_ret": f_ret,
        "tangency_vol": vol_tan,
        "tangency_ret": ret_tan,
        "asset_vol": asset_vol,
        "asset_ann": asset_ann,
        "cml_x": cml_x,
        "cml_y": cml_y,
    }


def plot_special_frontier_cml(res):
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.2))
    sc = ax.scatter(res["port_vol"], res["port_ret"], c=res["sharpe"], cmap="viridis", s=6, alpha=0.22)
    ax.plot(res["frontier_vol"], res["frontier_ret"], color="black", linewidth=2.2, label="Efficient frontier")
    ax.plot(res["cml_x"], res["cml_y"], color="crimson", linestyle="--", linewidth=2.1, label="CML")
    ax.scatter([res["tangency_vol"]], [res["tangency_ret"]], color="crimson", s=100, marker="*", label="Tangency")

    for j, t in enumerate(res["tickers"]):
        if t == FOCUS_TICKER:
            ax.scatter(res["asset_vol"][j], res["asset_ann"][j], color="#DA291C", s=110, marker="D", edgecolor="black", zorder=6)
        else:
            ax.scatter(res["asset_vol"][j], res["asset_ann"][j], color="white", s=58, edgecolor="black", zorder=6)
        ax.annotate(t, (res["asset_vol"][j], res["asset_ann"][j]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_title(f"{res['title']}: Efficient Frontier + CML")
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected Return (annualized)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.colorbar(sc, ax=ax, label="Sharpe")
    plt.tight_layout()
    plt.show()


def plot_multi_sector(results):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(16, max(5, 4.5 * n)))
    if n == 1:
        axes = np.array([axes])

    for i, res in enumerate(results):
        ax = axes[i, 0]
        ax2 = axes[i, 1]

        sc = ax.scatter(res["port_vol"], res["port_ret"], c=res["sharpe"], cmap="viridis", s=6, alpha=0.25)
        ax.plot(res["frontier_vol"], res["frontier_ret"], color="black", linewidth=2, label="Efficient frontier")
        ax.plot(res["cml_x"], res["cml_y"], color="crimson", linestyle="--", linewidth=2, label="CML")
        ax.scatter([res["tangency_vol"]], [res["tangency_ret"]], color="crimson", s=90, marker="*", label="Tangency")

        for j, t in enumerate(res["tickers"]):
            if t == FOCUS_TICKER:
                ax.scatter(
                    res["asset_vol"][j],
                    res["asset_ann"][j],
                    color="red",
                    s=90,
                    marker="D",
                    edgecolor="black",
                    zorder=5,
                    label=STOCK_FULL_NAMES.get(FOCUS_TICKER, FOCUS_TICKER),
                )
            else:
                ax.scatter(
                    res["asset_vol"][j],
                    res["asset_ann"][j],
                    color="white",
                    s=50,
                    edgecolor="black",
                    zorder=5,
                    label="Universe stocks" if j == 0 else None,
                )
            ax.annotate(t, (res["asset_vol"][j], res["asset_ann"][j]), xytext=(4, 4), textcoords="offset points", fontsize=8)

        for j, t in enumerate(res["etf_tickers"]):
            ax.scatter(
                res["etf_vol"][j],
                res["etf_ann"][j],
                color="gold",
                s=80,
                marker="^",
                edgecolor="black",
                zorder=6,
                label="Industry ETFs (XLY/PEJ/IYC)" if j == 0 else None,
            )
            ax.annotate(t, (res["etf_vol"][j], res["etf_ann"][j]), xytext=(4, -9), textcoords="offset points", fontsize=8)

        ax.set_title(f"{res['sector']}: Efficient Frontier + CML")
        ax.set_xlabel("Volatility (annualized)")
        ax.set_ylabel("Expected Return (annualized)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, ncol=1, framealpha=0.9)
        fig.colorbar(sc, ax=ax, label="Sharpe")

        ax2.plot(res["beta_line"], res["sml_y"], color="navy", linewidth=2, label="SML (CAPM)")
        for j, t in enumerate(res["tickers"]):
            ax2.plot([res["betas"][j], res["betas"][j]], [res["capm_expected"][j], res["asset_ann"][j]], color="gray", alpha=0.7, linewidth=1)
            if t == FOCUS_TICKER:
                ax2.scatter(res["betas"][j], res["asset_ann"][j], color="red", s=90, marker="D", edgecolor="black", label=f"{FOCUS_TICKER} actual")
                ax2.scatter(res["betas"][j], res["capm_expected"][j], color="red", s=70, marker="x", label=f"{FOCUS_TICKER} CAPM")
            else:
                ax2.scatter(res["betas"][j], res["asset_ann"][j], color="tab:blue", s=50, label="Universe stocks actual" if j == 0 else None)
                ax2.scatter(res["betas"][j], res["capm_expected"][j], color="tab:orange", s=50, marker="x", label="Universe stocks CAPM" if j == 0 else None)
            ax2.annotate(t, (res["betas"][j], res["asset_ann"][j]), xytext=(4, 4), textcoords="offset points", fontsize=8)

        for j, t in enumerate(res["etf_tickers"]):
            ax2.plot(
                [res["etf_betas"][j], res["etf_betas"][j]],
                [res["etf_capm"][j], res["etf_ann"][j]],
                color="goldenrod",
                alpha=0.7,
                linewidth=1,
            )
            ax2.scatter(res["etf_betas"][j], res["etf_ann"][j], color="gold", s=80, marker="^", edgecolor="black", label="Industry ETF actual" if j == 0 else None)
            ax2.scatter(res["etf_betas"][j], res["etf_capm"][j], color="goldenrod", s=65, marker="x", label="Industry ETF CAPM" if j == 0 else None)
            ax2.annotate(t, (res["etf_betas"][j], res["etf_ann"][j]), xytext=(4, -9), textcoords="offset points", fontsize=8)

        ax2.set_title(f"{res['sector']}: SML (Actual vs CAPM)")
        ax2.set_xlabel("Beta vs S&P 500")
        ax2.set_ylabel("Expected Return (annualized)")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper left", fontsize=8, ncol=1, framealpha=0.9)

    plt.tight_layout()
    plt.show()


def main():
    if not SECTOR_UNIVERSES:
        raise RuntimeError("No universe selected. Turn on RUN_MAG7_COMPARE.")

    all_results = []
    for sector, tickers in SECTOR_UNIVERSES.items():
        print(f"\n=== Running {sector} ===")
        try:
            res = analyze_sector(sector, tickers)
            all_results.append(res)
        except Exception as e:
            print(f"[ERROR] {sector} skipped: {e}")

    if not all_results:
        raise RuntimeError("No sector analysis completed.")

    plot_multi_sector(all_results)
    try:
        special_res = analyze_frontier_universe(SPECIAL_COMPARE_LABEL, SPECIAL_COMPARE_TICKERS)
        plot_special_frontier_cml(special_res)
    except Exception as e:
        print(f"[WARN] Special mean-variance comparison skipped: {e}")
    valuation_df = build_valuation_comps_table(VALUATION_TICKERS)
    plot_valuation_comps(valuation_df)

    for res in all_results:
        print(f"\n=== {res['sector']} | Asset Summary ===")
        print(res["summary"].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
        print(f"\n=== {res['sector']} | Tangency Portfolio (Max Sharpe) ===")
        print(
            f"risk_free_rate={RISK_FREE_RATE:.2%} | return={res['tangency_ret']:.2%} | "
            f"vol={res['tangency_vol']:.2%} | sharpe={res['tangency_sharpe']:.4f}"
        )
        print(res["weights"].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

        if not res["focus_row"].empty:
            fr = res["focus_row"].iloc[0]
            print(
                f"[{FOCUS_TICKER} focus in {res['sector']}] "
                f"beta={fr['beta_vs_sp500']:.4f}, actual={fr['ann_return']:.2%}, CAPM={fr['capm_return']:.2%}"
            )


if __name__ == "__main__":
    main()

