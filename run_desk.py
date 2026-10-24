"""One full TradingAgents run on desk data: 4 analysts -> bull/bear debate -> trader -> risk team -> portfolio manager.

usage: python run_desk.py NVDA 2026-10-23
"""
import sys
import time
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import write_report_tree


class Usage(BaseCallbackHandler):
    """Counts model calls and tokens per model, for the cost line at the end."""

    def __init__(self):
        self.by_model = {}

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                meta = getattr(msg, "usage_metadata", None) or {}
                model = (getattr(msg, "response_metadata", None) or {}).get("model_name", "?")
                row = self.by_model.setdefault(model, [0, 0, 0])
                row[0] += 1
                row[1] += meta.get("input_tokens", 0)
                row[2] += meta.get("output_tokens", 0)


ticker, date = sys.argv[1], sys.argv[2]
config = DEFAULT_CONFIG.copy()
usage = Usage()
print(f"== {ticker} {date} | quick={config['quick_think_llm']} deep={config['deep_think_llm']} | "
      f"prices={config['data_vendors']['core_stock_apis']}", flush=True)
ta = TradingAgentsGraph(selected_analysts=("market", "social", "news", "fundamentals"),
                        debug=True, config=config, callbacks=[usage])
t0 = time.time()
final_state, rating = ta.propagate(ticker, date)
out = Path("reports") / f"{ticker}_{date}"
path = write_report_tree(final_state, ticker, out, settings=ta.run_settings())
print(f"\n== RATING: {rating} | {time.time() - t0:.0f}s | report: {path}", flush=True)
for model, (calls, tin, tout) in usage.by_model.items():
    print(f"== usage {model}: {calls} calls, {tin:,} in / {tout:,} out tokens", flush=True)
