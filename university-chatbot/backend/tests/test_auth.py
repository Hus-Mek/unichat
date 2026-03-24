"""Tests for authentication and security."""

from datetime import timedelta

from jose import jwt

from backend.core.config import get_settings
from backend.core.security import (
    ALGORITHM,
    create_access_token,
    get_password_hash,
    verify_password,
    verify_token,
)


def test_password_hashing():
    password = "secure-password-123"
    hashed = get_password_hash(password)
    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("wrong-password", hashed)


def test_create_and_verify_token():
    data = {"sub": "user-id-123", "email": "test@uni.edu", "access_level": "student"}
    token = create_access_token(data)
    payload = verify_token(token)
    assert payload["sub"] == "user-id-123"
    assert payload["email"] == "test@uni.edu"
    assert payload["access_level"] == "student"
    assert "exp" in payload


def test_token_with_custom_expiry():
    data = {"sub": "user-456"}
    token = create_access_token(data, expires_delta=timedelta(minutes=5))
    payload = verify_token(token)
    assert payload["sub"] == "user-456"


def test_token_algorithm():
    data = {"sub": "test"}
    token = create_access_token(data)
    settings = get_settings()
    header = jwt.get_unverified_header(token)
    assert header["alg"] == ALGORITHM
