from enum import Enum


class ExtractionMode(str, Enum):

    HOT_PATH = "hot_path"
    BACKGROUND = "background"
    BOTH = "both"
    """Hot-path tools AND background extraction on the same turn. Semantic is safe to
    combine (the profile is merged by known id); episodic is an append-only log, so the
    background episodic pass is skipped (the hot-path tool already logs the episode).
    """


class MemoryType(str, Enum):

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


ALL_MEMORY_TYPES: frozenset[MemoryType] = frozenset(MemoryType)


class SemanticBackgroundStrategy(str, Enum):

    FETCH_MERGE = "fetch_merge"
    STORE_MANAGER = "store_manager"


class SearchStrategy(str, Enum):

    SIMILARITY = "similarity"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
