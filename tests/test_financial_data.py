"""数据工具层测试：_clean_value / _period_key 纯函数 + AkshareProvider 行序与清洗。

全部离线：用 **假的 akshare 模块**（monkeypatch sys.modules）喂固定 DataFrame，
不联网、不 import 真实 akshare。
"""

import json
import math
import sys
import types

import pytest

from src.tools.financial_data import AkshareProvider, _clean_value, _period_key


class TestCleanValue:
    def test_nan_becomes_none(self):
        assert _clean_value(float("nan")) is None
        assert _clean_value(float("inf")) is None
        assert _clean_value(float("-inf")) is None

    def test_none_and_blank(self):
        assert _clean_value(None) is None
        assert _clean_value("") is None
        assert _clean_value("   ") is None

    def test_numbers_stay_numbers(self):
        assert _clean_value(11.41) == 11.41
        assert _clean_value("63.6525") == 63.6525
        assert _clean_value(0) == 0.0

    def test_non_numeric_falls_back_to_str(self):
        assert _clean_value("--") == "--"


class TestPeriodKey:
    def test_date_and_datetime(self):
        import datetime as dt

        assert _period_key(dt.date(2026, 6, 30)) == "20260630"
        assert _period_key(dt.datetime(2025, 12, 31, 3, 0)) == "20251231"

    def test_string_forms(self):
        assert _period_key("2026-06-30") == "20260630"
        assert _period_key("2026年06月30日") == "20260630"
        assert _period_key("") == ""
        assert _period_key(None) == ""


def _fake_akshare(run):
    """把假 akshare 模块注入 sys.modules，并在用例结束后还原。"""

    def _call(symbol):
        return run

    module = types.ModuleType("akshare")
    module.stock_financial_analysis_indicator = _call
    return module


def _df(rows, columns=None):
    """用 pandas 造 DataFrame（pandas 是 akshare 的传递依赖，环境里必有）。"""
    import pandas as pd

    if columns is not None:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows)


@pytest.fixture
def patch_akshare(monkeypatch):
    def _install(run):
        monkeypatch.setitem(sys.modules, "akshare", _fake_akshare(run))

    return _install


def test_provider_picks_latest_period_not_first_row(patch_akshare):
    """akshare 该接口按报告期升序返回，首行是最老一期——必须取最新一期。"""
    df = _df(
        [
            {"日期": "2014-12-31", "净资产收益率(%)": 21.19, "销售净利率(%)": 6.41},
            {"日期": "2025-12-31", "净资产收益率(%)": 18.0, "销售净利率(%)": 13.0},
            {"日期": "2026-06-30", "净资产收益率(%)": 11.41, "销售净利率(%)": 16.98},
        ]
    )
    patch_akshare(df)
    out = AkshareProvider().get_financials("300750")

    assert out["report_period"] == "2026-06-30"
    assert out["as_of"] == "2026-06"
    assert out["roe"] == {"value": 11.41, "as_of": "2026-06"}
    assert out["net_margin"]["value"] == 16.98


def test_provider_handles_shuffled_order(patch_akshare):
    """行序被打乱（或接口改成降序）也要选到最新一期。"""
    df = _df(
        [
            {"日期": "2026-06-30", "净资产收益率(%)": 11.41},
            {"日期": "2014-12-31", "净资产收益率(%)": 21.19},
            {"日期": "2025-12-31", "净资产收益率(%)": 18.0},
        ]
    )
    patch_akshare(df)
    assert AkshareProvider().get_financials("300750")["report_period"] == "2026-06-30"


def test_provider_drops_nan_and_is_json_serializable(patch_akshare):
    """NaN 必须丢掉：留着会被 json.dumps(default=str) 写成非法 JSON 的 "nan"。"""
    df = _df(
        [
            {
                "日期": "2026-06-30",
                "净资产收益率(%)": 11.41,
                "销售毛利率(%)": float("nan"),
                "销售净利率(%)": 16.98,
                "资产负债率(%)": float("inf"),
            }
        ]
    )
    patch_akshare(df)
    out = AkshareProvider().get_financials("300750")

    assert "gross_margin" not in out  # NaN 直接不出现
    assert "debt_ratio" not in out
    assert out["missing_metrics"] == sorted(
        set(AkshareProvider.METRIC_MAP.values()) - set(out)
    )
    assert "gross_margin" in out["missing_metrics"]
    assert "debt_ratio" in out["missing_metrics"]

    blob = json.dumps(out, ensure_ascii=False)  # 不抛异常即通过
    assert "nan" not in blob.lower().replace("financial", "")
    assert math.isnan(float("nan"))  # 保持 math 引用，明确 NaN 语义


def test_provider_raises_when_all_key_metrics_missing(patch_akshare):
    df = _df([{"日期": "2026-06-30", "无关列": 1}])
    patch_akshare(df)
    with pytest.raises(RuntimeError, match="缺失全部关键指标"):
        AkshareProvider().get_financials("300750")


def test_provider_raises_on_empty_or_bad_structure(patch_akshare):
    patch_akshare(_df([], columns=["日期", "净资产收益率(%)"]))
    with pytest.raises(RuntimeError, match="未返回数据"):
        AkshareProvider().get_financials("300750")

    patch_akshare(_df([{"foo": 1}]))
    with pytest.raises(RuntimeError, match="结构与预期不符"):
        AkshareProvider().get_financials("300750")

    patch_akshare(_df([{"日期": "不是日期", "净资产收益率(%)": 1.0}]))
    with pytest.raises(RuntimeError, match="报告期列无法解析"):
        AkshareProvider().get_financials("300750")
