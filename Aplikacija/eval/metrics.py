from eval.scenarios import Scenario


def _text_for_sessions(session_replies: list[list[str]], from_index: int) -> str:
    return " ".join(
        reply
        for session_idx, replies in enumerate(session_replies)
        if session_idx >= from_index
        for reply in replies
    ).lower()


def _discriminative_indices(scenario: Scenario) -> list[int]:
    if scenario.discriminative_groups is None:
        return list(range(len(scenario.expected_facts)))
    return scenario.discriminative_groups


def memory_accuracy_strict(scenario: Scenario, session_replies: list[list[str]]) -> float:
    # groups whose marker the search tool prints on its own ("school" in "Schools nearby") are excluded
    if not scenario.expected_facts:
        return 1.0
    idx = _discriminative_indices(scenario)
    if not idx:
        return 1.0
    later_text = _text_for_sessions(session_replies, scenario.check_from_session)
    hits = sum(
        1 for i in idx if any(alt.lower() in later_text for alt in scenario.expected_facts[i])
    )
    return hits / len(idx)


def behavioral_consistency(scenario: Scenario, session_replies: list[list[str]]) -> float:
    groups = [scenario.expected_facts[i] for i in _discriminative_indices(scenario)]
    later_sessions = session_replies[scenario.check_from_session:]
    if not groups or not later_sessions:
        return 1.0

    later_texts = [" ".join(replies).lower() for replies in later_sessions]

    consistent = 0
    for group in groups:
        hits = [any(alt.lower() in text for alt in group) for text in later_texts]
        if any(hits):
            first_hit = hits.index(True)
            if all(hits[first_hit:]):
                consistent += 1
    return consistent / len(groups)


def false_memory_rate(scenario: Scenario, session_replies: list[list[str]]) -> float | None:
    if not scenario.distractor_facts:
        return None
    later_text = _text_for_sessions(session_replies, scenario.check_from_session)
    leaks = sum(1 for fact in scenario.distractor_facts if fact.lower() in later_text)
    return leaks / len(scenario.distractor_facts)


def memory_update_accuracy(old_values: list[str], new_values: list[str], replies_after_update: list[str]) -> float:
    text = " ".join(replies_after_update).lower()
    adopted = any(v.lower() in text for v in new_values)
    dropped = not any(v.lower() in text for v in old_values)
    return (float(adopted) + float(dropped)) / 2


def procedure_adherence(examples: list[tuple[list[str], str]]) -> float:
    if not examples:
        return 1.0
    hits = sum(1 for markers, reply in examples if any(m.lower() in reply.lower() for m in markers))
    return hits / len(examples)


def episodic_retrieval_accuracy(scenario: Scenario, turn_logs: list[dict]) -> float | None:
    # markers are looked up in both query and summary: the distinctive detail often lands only in the summary
    if not scenario.expected_episode_markers:
        return None
    relevant = [
        tl for tl in turn_logs
        if tl.get("session_index", 0) >= scenario.check_from_session and tl.get("episodes_found")
    ]
    if not relevant:
        return None
    markers = [m.lower() for m in scenario.expected_episode_markers]

    def top_episode_text(tl: dict) -> str:
        top = tl["episodes_found"][0]
        return f"{top.get('query') or ''} {top.get('summary') or ''}".lower()

    hits = sum(1 for tl in relevant if any(m in top_episode_text(tl) for m in markers))
    return hits / len(relevant)


def latency_overhead_seconds(variant_latencies: list[float], baseline_latencies: list[float]) -> float | None:
    if not variant_latencies or not baseline_latencies:
        return None
    return (sum(variant_latencies) / len(variant_latencies)) - (sum(baseline_latencies) / len(baseline_latencies))
