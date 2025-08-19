from barchart_swing_bot.paper import PaperBroker


def test_trailing_stop_updates():
    broker = PaperBroker(100000)
    pos = broker.buy("ABC", qty=10, price=10.0, atr=1.0, epsilon=0.01, atr_lambda=1.0)
    assert pos.trail == 10.01
    broker.mark("ABC", 12.0)
    assert pos.trail == 11.0
    broker.mark("ABC", 11.5)
    assert pos.trail == 11.0
    assert not broker.should_exit("ABC", 11.2)
    assert broker.should_exit("ABC", 10.5)
