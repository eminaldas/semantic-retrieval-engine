from backend.knowledge import gate


def test_passes_within_limits():
    summary = {"hit@3": 1.0, "mrr": 0.94, "false_abstain": 0.0}
    assert gate.check_limits(summary, {"min": {"hit@3": 0.97, "mrr": 0.92}, "max": {"false_abstain": 0.03}}) == []


def test_min_and_max_violations():
    summary = {"hit@3": 0.95, "false_abstain": 0.05}
    failures = gate.check_limits(summary, {"min": {"hit@3": 0.97}, "max": {"false_abstain": 0.03}})
    assert failures == [("hit@3", 0.95, 0.97, "min"), ("false_abstain", 0.05, 0.03, "max")]


def test_bound_itself_passes():
    assert gate.check_limits({"multihop_recall": 0.71}, {"min": {"multihop_recall": 0.71}}) == []


def test_missing_metric_fails():
    # Altin setten cok adimli sorular silinirse metrik kaybolur: sessizce gecmemeli.
    assert gate.check_limits({}, {"min": {"multihop_recall": 0.7}}) == [("multihop_recall", None, 0.7, "min")]


def test_warnings_only_above_bound():
    assert gate.check_warnings({"latency_p50": 1.4}, {"latency_p50": 1.0}) == [("latency_p50", 1.4, 1.0)]
    assert gate.check_warnings({"latency_p50": 0.3}, {"latency_p50": 1.0}) == []
    assert gate.check_warnings({}, None) == []


def test_missing_config_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["gate", "--config", str(tmp_path / "yok.json")])
    assert gate.main() == 0
    assert "atlandi" in capsys.readouterr().out
