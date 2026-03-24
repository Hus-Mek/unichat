"""
Five-level hierarchical access control.

Levels (lowest to highest):
    PUBLIC -> STUDENT -> FACULTY -> ADMIN_STAFF -> EXECUTIVE_BOARD

Higher levels can access all documents at or below their level.
"""

from .config import AccessLevel, ACCESS_HIERARCHY


def parse_access_level(level_str: str) -> AccessLevel:
    """Convert a string like 'admin_staff' to an AccessLevel enum member."""
    try:
        return AccessLevel[level_str.upper()]
    except KeyError:
        return AccessLevel.PUBLIC


def get_accessible_levels(user_level: AccessLevel) -> list[AccessLevel]:
    """Return all access levels visible to the given user level."""
    return ACCESS_HIERARCHY.get(user_level, [AccessLevel.PUBLIC])


def get_accessible_level_names(user_level: AccessLevel) -> list[str]:
    """Return accessible level names as lowercase strings (for ChromaDB filters)."""
    return [level.name.lower() for level in get_accessible_levels(user_level)]


def build_chroma_filter(
    user_level: AccessLevel,
    user_id: str | None = None,
) -> dict:
    """
    Build a ChromaDB ``where`` filter for access-controlled retrieval.

    If *user_id* is provided, the filter also includes documents owned
    by that user regardless of access level.
    """
    accessible = get_accessible_level_names(user_level)
    base_filter: dict = {"access_level": {"$in": accessible}}

    if user_id:
        return {
            "$or": [
                {"access_level": {"$in": accessible}},
                {"owner_id": user_id},
            ]
        }

    return base_filter


def can_access_collection(
    user_level: AccessLevel,
    collection_min_level: AccessLevel,
) -> bool:
    """Return True if the user's level meets or exceeds the collection minimum."""
    return user_level >= collection_min_level
