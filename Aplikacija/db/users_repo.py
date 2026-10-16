import hashlib
import hmac
import os
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from db.connection import get_conn_string

_PBKDF2_ITERATIONS = 260_000


class InvalidCredentials(Exception):
    pass


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, _, digest_hex = stored_hash.partition("$")
    if not digest_hex:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return hmac.compare_digest(actual, expected)


def get_user_by_username(username: str) -> Optional[dict]:
    with psycopg.connect(get_conn_string(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, display_name, password_hash, created_at FROM users WHERE username = %s",
                (username,),
            )
            return cur.fetchone()


def create_user(username: str, password: str, display_name: Optional[str] = None) -> dict:
    password_hash = _hash_password(password)
    with psycopg.connect(get_conn_string(), row_factory=dict_row, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, display_name, password_hash) VALUES (%s, %s, %s) "
                "RETURNING id, username, display_name, created_at",
                (username, display_name or username, password_hash),
            )
            return cur.fetchone()


def authenticate_user(username: str, password: str, display_name: Optional[str] = None) -> dict:
    user = get_user_by_username(username)
    if user is None:
        return create_user(username, password, display_name)
    if user["password_hash"] is None:
        # Account predates the password_hash column: claim it with this password.
        set_password(user["id"], password)
        return user
    if not _verify_password(password, user["password_hash"]):
        raise InvalidCredentials(f"Wrong password for user '{username}'.")
    return user


_EVAL_SCRIPT_PASSWORD = "eval-script-account"


def get_or_create_user(username: str, display_name: Optional[str] = None) -> dict:
    return authenticate_user(username, _EVAL_SCRIPT_PASSWORD, display_name)


def set_password(user_id: int, password: str) -> None:
    with psycopg.connect(get_conn_string(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (_hash_password(password), user_id),
            )
