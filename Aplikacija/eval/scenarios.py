from dataclasses import dataclass, field


@dataclass
class Scenario:
    name: str
    sessions: list[list[str]]
    expected_facts: list[list[str]] = field(default_factory=list)
    check_from_session: int = 1
    distractor_facts: list[str] = field(default_factory=list)

    expected_episode_markers: list[str] = field(default_factory=list)
    discriminative_groups: list[int] | None = None


SCENARIOS: list[Scenario] = [
    Scenario(
        name="budget_and_climate_preference",
        sessions=[
            [
                "I really don't like humid weather and my budget is usually around "
                "$200,000. Find me some cheap sunny towns."
            ],
            [
                "Find me some good places to live.",
            ],
        ],
        expected_facts=[
            ["200,000", "200000", "$200k", "200k"],
            ["humid"],
        ],
        check_from_session=1,
        discriminative_groups=[0],
    ),
    Scenario(
        name="state_and_school_preference",
        sessions=[
            [
                "I'm only interested in Texas or Arizona, and I need a place with "
                "good schools nearby since I have kids."
            ],
            [
                "What towns would you recommend for me?",
            ],
        ],
        expected_facts=[
            ["texas"],
            ["arizona"],
            ["school"],
        ],
        check_from_session=1,
        discriminative_groups=[0, 1],
    ),
    Scenario(
        name="dislike_persists_across_three_sessions",
        sessions=[
            ["I hate snow and cold winters, keep that in mind for future searches."],
            ["Any good towns for me this week?"],
            ["What about now, anything new you'd suggest?"],
        ],
        expected_facts=[
            ["snow"],
            ["cold", "winter"],
        ],
        check_from_session=1,
        # Group 0 ("snow") is not discriminative: the tool prints a "Snowy days" field.
        discriminative_groups=[1],
    ),
    Scenario(
        name="episodic_recall_of_past_search",
        sessions=[
            [
                "Find me cheap sunny towns in Arizona. I'm looking today specifically "
                "because my daughter is starting college near Phoenix this fall."
            ],
            [
                "What did I mention was the reason I was looking in Arizona, last time we talked?",
            ],
        ],
        expected_facts=[["phoenix"]],
        check_from_session=1,
        discriminative_groups=[0],
        expected_episode_markers=["phoenix"],
    ),
    Scenario(
        name="episodic_disambiguation_of_two_searches",
        sessions=[
            [
                "Find me cheap sunny towns in Arizona. I'm looking today specifically "
                "because my daughter is starting college near Phoenix this fall."
            ],
            [
                "Now find me cheap towns in Oregon. I'm planning to open a bakery there."
            ],
            [
                "Remind me why I was looking in Oregon last time",
            ],
        ],
        expected_facts=[["bakery"]],
        check_from_session=2,
        # "daughter" belongs to the other episode; unlike "phoenix" (Phoenix_Oregon exists)
        # no town name contains it, so only a mix-up can put it in the reply.
        distractor_facts=["daughter"],
        discriminative_groups=[0],
        expected_episode_markers=["bakery"],
    ),
    Scenario(
        name="distractor_relevant_vs_irrelevant",
        sessions=[
            [
                "My budget is around $250,000 and I only want to see towns in Oregon. "
                "By the way, my cousin just moved to Vermont, not that it matters for me."
            ],
            [
                "What towns would you recommend for me?",
            ],
        ],
        expected_facts=[
            ["250,000", "250000", "$250k", "250k"],
            ["oregon"],
        ],
        distractor_facts=["vermont"],
        check_from_session=1,
        discriminative_groups=[0, 1],
    ),
]

SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


def resolve_scenarios(names: str) -> list[Scenario]:
    if names == "all":
        return list(SCENARIOS)
    selected = []
    for name in names.split(","):
        name = name.strip()
        if name not in SCENARIOS_BY_NAME:
            raise ValueError(f"Unknown scenario '{name}'. Known: {list(SCENARIOS_BY_NAME)}")
        selected.append(SCENARIOS_BY_NAME[name])
    return selected
