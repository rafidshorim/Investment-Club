import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import Efficient_frontier as ef


def fake_download_returns(tickers, market_ticker, start, end, min_assets=3):
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    base = np.linspace(-0.01, 0.01, len(idx))
    data = {}
    for i, t in enumerate(tickers):
        data[t] = 0.0005 + (i * 0.0001) + base * 0.02
    asset_ret = pd.DataFrame(data, index=idx)
    mkt_ret = pd.Series(0.0004 + base * 0.015, index=idx, name=market_ticker)
    return asset_ret, mkt_ret


def fake_valuation_table(tickers):
    return pd.DataFrame(
        {
            "ticker": tickers,
            "forward_pe": [20.0 + i for i in range(len(tickers))],
            "price_to_sales": [8.0 + i * 0.1 for i in range(len(tickers))],
            "ev_to_ebitda": [15.0 + i * 0.3 for i in range(len(tickers))],
            "fcf_yield": [0.03 + i * 0.001 for i in range(len(tickers))],
        }
    )


class AnalysisSmokeTests(unittest.TestCase):
    @patch("Efficient_frontier.build_valuation_comps_table", side_effect=fake_valuation_table)
    @patch("Efficient_frontier.download_returns", side_effect=fake_download_returns)
    def test_run_analysis_contract(self, _mock_download, _mock_val):
        cfg = {
            "universe_tickers": ["AAPL", "MSFT", "NVDA"],
            "focus_ticker": "NVDA",
            "sector_universes": {"TestUniverse": ["AAPL", "MSFT", "NVDA"]},
            "n_portfolios": 2000,
            "industry_etfs": ["QQQ"],
            "valuation_tickers": ["AAPL", "MSFT", "NVDA"],
            "special_compare_tickers": ["AAPL", "MSFT", "NVDA"],
        }
        out = ef.run_analysis(config=cfg, render_plots=False, show_plots=False, save_valuation_png=False)
        self.assertIn("all_results", out)
        self.assertTrue(len(out["all_results"]) >= 1)
        first = out["all_results"][0]
        self.assertIn("summary", first)
        self.assertIn("weights", first)
        self.assertTrue(np.isfinite(first["tangency_sharpe"]))
        self.assertFalse(first["summary"].empty)
        self.assertFalse(first["weights"].empty)


class DashboardSmokeTests(unittest.TestCase):
    def test_dashboard_compute_analysis(self):
        try:
            import app
        except ModuleNotFoundError as e:
            if e.name == "streamlit":
                self.skipTest("streamlit is not installed in this environment")
            raise

        with patch("app.run_analysis", return_value={"ok": True}) as mock_run:
            out = app.compute_analysis({"x": 1})
            self.assertEqual(out, {"ok": True})
            mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
