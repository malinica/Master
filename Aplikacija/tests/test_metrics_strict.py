from eval import metrics
from eval.scenarios import SCENARIOS, SCENARIOS_BY_NAME

TOOL_ONLY_REPLY = [["Town A: Schools nearby: 3, Humidity: 70%, price $180,000. Town B ..."], ["Town A: Schools nearby: 3, Humidity: 70%"]]


def test_baseline_like_reply_scores_zero():
    sc = SCENARIOS_BY_NAME["state_and_school_preference"]
    replies = [["x"], ["Here are towns. Schools nearby: 4 for each."]]
    assert metrics.memory_accuracy_strict(sc, replies) == 0


def test_humidity_field_does_not_count_for_budget_scenario():
    sc = SCENARIOS_BY_NAME["budget_and_climate_preference"]
    assert metrics.memory_accuracy_strict(sc, TOOL_ONLY_REPLY) == 0.0


def test_real_recall_scores_one_strict():
    sc = SCENARIOS_BY_NAME["state_and_school_preference"]
    replies = [["x"], ["Based on your interest in Texas and Arizona, try ..."]]
    assert metrics.memory_accuracy_strict(sc, replies) == 1.0


def test_every_scenario_declares_valid_discriminative_groups():
    for sc in SCENARIOS:
        assert sc.discriminative_groups, sc.name
        assert all(0 <= i < len(sc.expected_facts) for i in sc.discriminative_groups), sc.name
