"""Turn a finished run's state log into the JSON the Cross-Asset Desk research panel reads.

usage: python export_desk.py NVDA 2026-10-23 OUT.json [usage-line ...]
"""
import json
import re
import sys
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG

ticker, date, out = sys.argv[1], sys.argv[2], sys.argv[3]
log = Path(DEFAULT_CONFIG["results_dir"]) / ticker / "TradingAgentsStrategy_logs" / f"full_states_log_{date}.json"
s = json.loads(log.read_text(encoding="utf-8"))

TURN = re.compile(r"(?:^|\n)(Bull|Bear|Aggressive|Conservative|Neutral) Analyst: ")


def turns(history: str) -> list[dict]:
    """'\\nBull Analyst: ...\\nBear Analyst: ...' -> [{speaker, text}] in order."""
    parts = TURN.split(history or "")
    return [{"speaker": parts[i], "text": parts[i + 1].strip()} for i in range(1, len(parts) - 1, 2)]


settings = s.get("run_settings") or {}
data = {
    "ticker": s["company_of_interest"], "trade_date": s["trade_date"], "rating": s["final_rating"],
    "settings": {k: settings.get(k) for k in ("llm_provider", "quick_think_llm", "deep_think_llm", "data_vendors",
                                               "max_debate_rounds", "max_risk_discuss_rounds")},
    "usage": sys.argv[4:],
    "analysts": [
        {"role": "Technicals", "agent": "Market Analyst", "text": s["market_report"]},
        {"role": "Sentiment", "agent": "Social Media Analyst", "text": s["sentiment_report"]},
        {"role": "News", "agent": "News Analyst", "text": s["news_report"]},
        {"role": "Fundamentals", "agent": "Fundamentals Analyst", "text": s["fundamentals_report"]},
    ],
    "debate": turns(s["investment_debate_state"]["history"]),
    "research_manager": s["investment_plan"],
    "trader": s["trader_investment_plan"],
    "risk": turns(s["risk_debate_state"]["history"]),
    "portfolio_manager": s["final_trade_decision"],
}
Path(out).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
print(f"{data['ticker']} {data['trade_date']}: rating {data['rating']}; {len(data['debate'])} debate turns, "
      f"{len(data['risk'])} risk turns; wrote {out}")
