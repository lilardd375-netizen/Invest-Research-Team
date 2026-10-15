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


# ---------------------------------------------------------------------------
# 第九步任务 1：口径自洽诊断的落地——杜邦上下文、口径标注、多期趋势
# ---------------------------------------------------------------------------


class TestPeriodScale:
    def test_scale_follows_report_month(self):
        from src.tools.financial_data import period_scale

        assert "半年报" in period_scale("2026-06-30")
        assert "一季报" in period_scale("2026-03-31")
        assert "年报" in period_scale("2025-12-31")
        assert "非年化" in period_scale("2026-09-30")


class TestDupontContext:
    def test_real_numbers_are_self_consistent(self):
        """300750 真实数据：ROE ≈ 总资产利润率 × 权益乘数，偏差 < 1%。"""
        from src.tools.financial_data import dupont_context

        ctx = dupont_context(
            {"roe": 11.41, "roa": 4.1295, "net_margin": 16.9837, "debt_ratio": 63.6525}
        )
        assert ctx["equity_multiplier"] == pytest.approx(2.7512, abs=1e-3)
        assert ctx["roe_implied_roa_x_em"] == pytest.approx(11.36, abs=0.05)
        assert ctx["identity_gap_pct"] < 1

    def test_missing_inputs_are_omitted_not_guessed(self):
        from src.tools.financial_data import dupont_context

        assert dupont_context({"roe": 11.41}) == {"roe": 11.41}


class TestCaliberNotes:
    def test_notes_cover_scale_and_caliber_mixing(self):
        from src.tools.financial_data import caliber_notes

        blob = "".join(
            caliber_notes({"roa": 4.1295, "roe": 11.41, "roe_weighted": 12.08}, "2026-06-30")
        )
        assert "非年化" in blob
        assert "利润总额" in blob  # 两种 ROA 口径
        assert "加权" in blob  # 摊薄 vs 加权
        assert "归母" in blob  # 归母 vs 全部权益
        assert "权益乘数" in blob  # 恒等式写入 prompt


def test_provider_outputs_dupont_trend_and_caliber(patch_akshare):
    """真实接口同源字段：杜邦上下文 + 多期趋势 + 口径标注一次给全。"""
    df = _df(
        [
            {
                "日期": "2025-12-31",
                "净资产收益率(%)": 21.42,
                "总资产利润率(%)": 7.8769,
                "销售净利率(%)": 18.1227,
                "资产负债率(%)": 61.9393,
                "主营业务收入增长率(%)": 17.0406,
            },
            {
                "日期": "2026-03-31",
                "净资产收益率(%)": 5.8,
                "总资产利润率(%)": 2.1731,
                "销售净利率(%)": 17.6079,
                "资产负债率(%)": 62.3223,
                "主营业务收入增长率(%)": 52.4487,
            },
            {
                "日期": "2026-06-30",
                "净资产收益率(%)": 11.41,
                "总资产利润率(%)": 4.1295,
                "销售净利率(%)": 16.9837,
                "资产负债率(%)": 63.6525,
                "主营业务收入增长率(%)": 54.8004,
                "总资产净利润率(%)": 4.4501,
                "总资产周转率(次)": 0.262,
                "股东权益比率(%)": 36.3475,
                "加权净资产收益率(%)": 12.08,
            },
        ]
    )
    patch_akshare(df)
    out = AkshareProvider().get_financials("300750")

    assert out["dupont"]["equity_multiplier"] == pytest.approx(2.7512, abs=1e-3)
    assert out["dupont"]["total_asset_turnover"] == 0.262
    assert out["dupont"]["identity_gap_pct"] < 1
    assert out["period_scale"].startswith("半年报")
    assert [item["period"] for item in out["trend"]] == [
        "2025-12-31",
        "2026-03-31",
        "2026-06-30",
    ]
    assert out["trend"][-1]["roe"] == 11.41
    assert any("归母" in note for note in out["caliber_notes"])
    json.dumps(out, ensure_ascii=False)  # 可序列化


def test_provider_trend_caps_at_five_periods(patch_akshare):
    rows = [{"日期": f"20{20 + i}-12-31", "净资产收益率(%)": 10.0 + i} for i in range(8)]
    patch_akshare(_df(rows))
    out = AkshareProvider().get_financials("300750")

    assert len(out["trend"]) == 5
    assert [item["roe"] for item in out["trend"]] == [13.0, 14.0, 15.0, 16.0, 17.0]


def test_provider_without_trend_columns_still_works(patch_akshare):
    """缺趋势 / 杜邦列时不崩，只是不给这些上下文。"""
    patch_akshare(_df([{"日期": "2026-06-30", "净资产收益率(%)": 11.41}]))
    out = AkshareProvider().get_financials("300750")

    assert out["dupont"] == {"roe": 11.41}
    assert out["trend"] == [{"period": "2026-06-30", "roe": 11.41}]
    assert isinstance(out["caliber_notes"], list) and out["caliber_notes"]
