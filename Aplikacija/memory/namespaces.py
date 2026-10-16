
def semantic_ns(user_id: int) -> tuple[str, ...]:
    return ("memories", str(user_id), "semantic_profile")


def episodic_ns(user_id: int) -> tuple[str, ...]:
    return ("memories", str(user_id), "episodic_search")
