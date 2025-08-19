from barchart_swing_bot.confidence import (
    backtest_moving_average,
    compute_confidence,
    determine_best_timeframe,
    is_actionable,
)


def test_confidence_actionable():
    features = {
        "wtd_alpha": 1,
        "momentum_daily": 1,
        "momentum_weekly": 1,
        "inst_score": 1,
        "rsi": 1,
        "stochastic": 0,
        "macd": 1,
        "ma_signal": 0,
    }
    assert compute_confidence(features) == 80
    assert is_actionable(features)


def test_confidence_not_actionable():
    features = {
        "wtd_alpha": 1,
        "momentum_daily": 0,
        "momentum_weekly": 1,
        "inst_score": 1,
        "rsi": 0,
        "stochastic": 0,
        "macd": 1,
        "ma_signal": 1,
    }
    assert compute_confidence(features) == 65
    assert not is_actionable(features)


def test_determine_best_timeframe():
    daily = {
        "wtd_alpha": 1,
        "momentum_daily": 1,
        "momentum_weekly": 0,
        "inst_score": 1,
        "rsi": 1,
        "stochastic": 1,
        "macd": 1,
        "ma_signal": 1,
    }
    weekly = {
        "wtd_alpha": 1,
        "momentum_daily": 0,
        "momentum_weekly": 1,
        "inst_score": 1,
        "rsi": 1,
        "stochastic": 0,
        "macd": 1,
        "ma_signal": 0,
    }
    tf, conf, passes = determine_best_timeframe(daily, weekly)
    assert tf == "daily"
    assert conf == compute_confidence(daily)
    assert passes


def test_backtest_moving_average():
    prices = [1, 2, 3, 4, 5, 6]
    result = backtest_moving_average(prices, short=2, long=3)
    assert result == 0.5
