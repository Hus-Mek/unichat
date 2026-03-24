"""Tests for access control logic."""

from backend.core.access_control import (
    AccessLevel,
    build_chroma_filter,
    can_access_collection,
    get_accessible_level_names,
    get_accessible_levels,
    parse_access_level,
)
from backend.core.config import ACCESS_HIERARCHY


def test_parse_access_level_valid():
    assert parse_access_level("public") == AccessLevel.PUBLIC
    assert parse_access_level("student") == AccessLevel.STUDENT
    assert parse_access_level("faculty") == AccessLevel.FACULTY
    assert parse_access_level("admin_staff") == AccessLevel.ADMIN_STAFF
    assert parse_access_level("executive_board") == AccessLevel.EXECUTIVE_BOARD


def test_parse_access_level_invalid():
    assert parse_access_level("unknown") == AccessLevel.PUBLIC
    assert parse_access_level("") == AccessLevel.PUBLIC


def test_hierarchy_completeness():
    """Every AccessLevel should be a key in ACCESS_HIERARCHY."""
    for level in AccessLevel:
        assert level in ACCESS_HIERARCHY


def test_hierarchy_ordering():
    """Higher levels should see everything lower levels see."""
    levels = sorted(AccessLevel)
    for i, level in enumerate(levels):
        accessible = get_accessible_levels(level)
        for lower in levels[:i + 1]:
            assert lower in accessible


def test_get_accessible_level_names():
    names = get_accessible_level_names(AccessLevel.FACULTY)
    assert "public" in names
    assert "student" in names
    assert "faculty" in names
    assert "admin_staff" not in names


def test_build_chroma_filter_no_user():
    f = build_chroma_filter(AccessLevel.STUDENT)
    assert f == {"access_level": {"$in": ["public", "student"]}}


def test_build_chroma_filter_with_user():
    f = build_chroma_filter(AccessLevel.STUDENT, user_id="user-123")
    assert "$or" in f
    assert len(f["$or"]) == 2
    assert f["$or"][0] == {"access_level": {"$in": ["public", "student"]}}
    assert f["$or"][1] == {"owner_id": "user-123"}


def test_build_chroma_filter_executive():
    f = build_chroma_filter(AccessLevel.EXECUTIVE_BOARD)
    expected = ["public", "student", "faculty", "admin_staff", "executive_board"]
    assert f == {"access_level": {"$in": expected}}


def test_can_access_collection():
    assert can_access_collection(AccessLevel.FACULTY, AccessLevel.STUDENT)
    assert can_access_collection(AccessLevel.STUDENT, AccessLevel.STUDENT)
    assert not can_access_collection(AccessLevel.STUDENT, AccessLevel.FACULTY)
    assert can_access_collection(AccessLevel.EXECUTIVE_BOARD, AccessLevel.ADMIN_STAFF)
    assert not can_access_collection(AccessLevel.PUBLIC, AccessLevel.STUDENT)
