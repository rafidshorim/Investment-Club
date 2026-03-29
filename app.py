import json

import pandas as pd
import streamlit as st

from Efficient_frontier import DEFAULT_ANALYSIS_CONFIG, run_analysis


def parse_tickers(raw: str):
    items = [x.strip().upper() for x in raw.replace("\n", ",").split(",")]
    out = []
    seen = set()
    for t in items:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def build_runtime_config(
    universe_tickers,
    focus_ticker,
    start,
    end,
    risk_free_rate,
    n_portfolios,
    market_ticker,
    industry_etfs,
):
    return {
        "sector_universes": {"Selected_Universe": universe_tickers},
        "universe_tickers": universe_tickers,
        "focus_ticker": focus_ticker,
        "market_ticker": market_ticker,
        "industry_etfs": industry_etfs,
        "valuation_tickers": universe_tickers,
        "special_compare_tickers": universe_tickers,
        "special_compare_label": "Selected Universe Mean-Variance Universe",
        "start": str(start),
        "end": str(end),
        "risk_free_rate": float(risk_free_rate),
        "n_portfolios": int(n_portfolios),
        "seed": int(DEFAULT_ANALYSIS_CONFIG["seed"]),
    }


def compute_analysis(config):
    return run_analysis(
        config=config,
        render_plots=True,
        show_plots=False,
        save_valuation_png=False,
        suppress_warnings=True,
        include_valuation=False,
    )


def main():
    st.set_page_config(page_title="Efficient Frontier Dashboard", layout="wide")
    st.title("Efficient Frontier Dashboard")
    st.caption("Interactive mean-variance, CML/SML, CAPM and valuation comparables dashboard.")

    default_tickers = DEFAULT_ANALYSIS_CONFIG["sector_universes"]["Magnificent_7"]

    with st.sidebar:
        st.header("Configuration")
        ticker_raw = st.text_area("Universe Tickers (comma-separated)", value=", ".join(default_tickers), height=110)
        tickers = parse_tickers(ticker_raw)
        focus_default = DEFAULT_ANALYSIS_CONFIG["focus_ticker"]
        focus_ticker = st.text_input("Anchor / Focus Ticker", value=focus_default).strip().upper()

        start_default = pd.Timestamp(DEFAULT_ANALYSIS_CONFIG["start"]).date()
        end_default = pd.Timestamp(DEFAULT_ANALYSIS_CONFIG["end"]).date()
        start_date = st.date_input("Start Date", value=start_default)
        end_date = st.date_input("End Date", value=end_default, min_value=start_date)

        risk_free_rate = st.number_input(
            "Risk-Free Rate (annual)",
            min_value=0.0,
            max_value=0.30,
            value=float(DEFAULT_ANALYSIS_CONFIG["risk_free_rate"]),
            step=0.005,
            format="%.3f",
        )
        n_portfolios = st.slider("Monte Carlo Portfolios", 5000, 120000, int(DEFAULT_ANALYSIS_CONFIG["n_portfolios"]), step=5000)
        market_ticker = st.text_input("Market Benchmark Ticker", value=DEFAULT_ANALYSIS_CONFIG["market_ticker"]).strip().upper()

        industry_raw = st.text_input(
            "Industry ETFs (comma-separated)", value=", ".join(DEFAULT_ANALYSIS_CONFIG["industry_etfs"])
        )
        industry_etfs = parse_tickers(industry_raw)

        run_clicked = st.button("Run Analysis", type="primary")

    if not tickers:
        st.error("Please provide at least one valid ticker in the universe.")
        return

    if not focus_ticker:
        focus_ticker = tickers[0]

    if focus_ticker not in tickers:
        st.warning(f"Focus ticker {focus_ticker} is not in the universe. It will be appended.")
        tickers = tickers + [focus_ticker]

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        return

    if not run_clicked:
        st.info("Set parameters in the sidebar, then click **Run Analysis**.")
        return

    config = build_runtime_config(
        universe_tickers=tickers,
        focus_ticker=focus_ticker,
        start=start_date,
        end=end_date,
        risk_free_rate=risk_free_rate,
        n_portfolios=n_portfolios,
        market_ticker=market_ticker,
        industry_etfs=industry_etfs,
    )

    with st.spinner("Running portfolio analytics..."):
        try:
            result = compute_analysis(config)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    all_results = result["all_results"]
    figs = result["figures"]

    top = all_results[0]
    st.subheader("Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Universe Size", len(top["tickers"]))
    col2.metric("Tangency Return", f"{top['tangency_ret']:.2%}")
    col3.metric("Tangency Volatility", f"{top['tangency_vol']:.2%}")
    col4.metric("Tangency Sharpe", f"{top['tangency_sharpe']:.3f}")

    if not top["focus_row"].empty:
        fr = top["focus_row"].iloc[0]
        st.info(
            f"{focus_ticker} | beta={fr['beta_vs_sp500']:.3f}, actual return={fr['ann_return']:.2%}, "
            f"CAPM return={fr['capm_return']:.2%}, alpha={fr['alpha_actual_minus_capm']:.2%}"
        )

    st.subheader("Efficient Frontier + CML and SML")
    if "multi_sector" in figs:
        st.pyplot(figs["multi_sector"], clear_figure=False)

    st.subheader("Selected Universe Frontier")
    if "special_frontier" in figs:
        st.pyplot(figs["special_frontier"], clear_figure=False)
    elif result.get("special_error"):
        st.warning(f"Special frontier unavailable: {result['special_error']}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Asset Summary Table**")
        st.dataframe(top["summary"], width="stretch")
    with c2:
        st.markdown("**Tangency Weights**")
        st.dataframe(top["weights"], width="stretch")

    csv_bytes = top["summary"].to_csv(index=False).encode("utf-8")
    st.download_button("Download Asset Summary CSV", data=csv_bytes, file_name="asset_summary.csv", mime="text/csv")

    with st.expander("Run Config (JSON)"):
        st.code(json.dumps(config, indent=2), language="json")


if __name__ == "__main__":
    main()
