from unittest import mock

import pandas as pd
import pytest

from tradingagents.memory import settlement


def _cdc(start="2026-07-01", n=15):
    days = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"Date": days, "Close": [100.0 + i for i in range(n)]}), "UNI_USD"


@pytest.mark.unit
def test_a_coin_yahoo_has_nothing_for_is_priced_from_crypto_com():
    with mock.patch.object(settlement, "get_closes", return_value=pd.Series(dtype=float)), \
         mock.patch.object(settlement.cryptocom, "load_daily", return_value=_cdc()) as cdc:
        closes = settlement._closes("UNI-USD", "2026-07-03", "2026-07-10")
    cdc.assert_called_once()
    assert list(closes.index.strftime("%Y-%m-%d")) == [f"2026-07-0{d}" for d in range(3, 10)]


@pytest.mark.unit
def test_yahoo_still_wins_when_it_has_the_coin():
    yahoo = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-07-03", "2026-07-04"]))
    with mock.patch.object(settlement, "get_closes", return_value=yahoo), \
         mock.patch.object(settlement.cryptocom, "load_daily") as cdc:
        assert settlement._closes("BTC-USD", "2026-07-03", "2026-07-10").equals(yahoo)
    cdc.assert_not_called()


@pytest.mark.unit
def test_a_stock_never_falls_back():
    with mock.patch.object(settlement, "get_closes", return_value=pd.Series(dtype=float)), \
         mock.patch.object(settlement.cryptocom, "load_daily") as cdc:
        assert settlement._closes("NVDA", "2026-07-03", "2026-07-10").empty
    cdc.assert_not_called()
