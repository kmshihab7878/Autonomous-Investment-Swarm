"""Tests for HMAC key rotation in risk token signing and verification.

Covers the dual-key overlap window that allows zero-downtime key rotation.
During rotation, tokens signed with the previous key remain valid until
their TTL (300s) expires naturally.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import time

import pytest

from aiswarm.risk.limits import (
    TOKEN_TTL_SECONDS,
    _DEFAULT_KEY_ID,
    sign_risk_token,
    verify_risk_token,
)


class TestHmacKeyRotation:
    """HMAC key rotation: dual-key verification during overlap window."""

    def setup_method(self) -> None:
        """Ensure a clean key environment for each test."""
        import os

        os.environ["AIS_RISK_HMAC_SECRET"] = "current-secret-key"
        os.environ.pop("AIS_RISK_HMAC_SECRET_PREVIOUS", None)
        os.environ.pop("AIS_RISK_HMAC_KEY_ID", None)

    def teardown_method(self) -> None:
        """Restore the default test secret used by the rest of the suite."""
        import os

        os.environ["AIS_RISK_HMAC_SECRET"] = "test-secret-for-ci"
        os.environ.pop("AIS_RISK_HMAC_SECRET_PREVIOUS", None)
        os.environ.pop("AIS_RISK_HMAC_KEY_ID", None)

    # ---------------------------------------------------------------
    # 1. Normal case: sign with current key, verify with current key
    # ---------------------------------------------------------------

    def test_sign_and_verify_with_current_key(self) -> None:
        """Token signed with the current key verifies successfully."""
        token = sign_risk_token("ord_normal")
        assert verify_risk_token(token, "ord_normal")

    # ---------------------------------------------------------------
    # 2. Rotation case: token signed with previous key still verifies
    # ---------------------------------------------------------------

    def test_previous_key_token_verifies_during_rotation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """During key rotation, a token signed with the old key can still
        be verified when the old key is configured as the previous key."""

        # Step 1: sign a token with the "old" key
        old_secret = "old-secret-key"
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET", old_secret)
        monkeypatch.setenv("AIS_RISK_HMAC_KEY_ID", "v1")
        token = sign_risk_token("ord_rotate")

        # Step 2: rotate — new key becomes current, old becomes previous
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET", "new-secret-key")
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET_PREVIOUS", old_secret)
        monkeypatch.setenv("AIS_RISK_HMAC_KEY_ID", "v2")

        # Token signed with old key should still verify via fallback
        assert verify_risk_token(token, "ord_rotate")

    # ---------------------------------------------------------------
    # 3. Unknown key: token signed with a completely unknown key rejects
    # ---------------------------------------------------------------

    def test_unknown_key_token_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A token signed with a key that is neither current nor previous
        must be rejected."""
        # Sign with an unknown secret by crafting a token manually
        order_id = "ord_unknown"
        timestamp = str(int(time.time()))
        key_id = "v99"
        payload = f"{order_id}:{timestamp}:{key_id}"
        unknown_secret = "totally-unknown-secret"
        sig = hmac_mod.new(
            unknown_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        forged_token = f"{payload}:{sig}"

        # Current key is different, no previous key configured
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET", "current-secret-key")
        monkeypatch.delenv("AIS_RISK_HMAC_SECRET_PREVIOUS", raising=False)

        assert not verify_risk_token(forged_token, order_id)

    def test_unknown_key_rejected_even_with_previous_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A token signed with an unknown key is rejected even when a
        previous key is configured (neither key matches)."""
        order_id = "ord_unknown2"
        timestamp = str(int(time.time()))
        key_id = "v99"
        payload = f"{order_id}:{timestamp}:{key_id}"
        unknown_secret = "totally-unknown-secret"
        sig = hmac_mod.new(
            unknown_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        forged_token = f"{payload}:{sig}"

        monkeypatch.setenv("AIS_RISK_HMAC_SECRET", "current-secret-key")
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET_PREVIOUS", "old-secret-key")

        assert not verify_risk_token(forged_token, order_id)

    # ---------------------------------------------------------------
    # 4. key_id is included in the token payload
    # ---------------------------------------------------------------

    def test_key_id_in_token_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The key_id field is embedded in the token and matches the
        configured AIS_RISK_HMAC_KEY_ID."""
        monkeypatch.setenv("AIS_RISK_HMAC_KEY_ID", "v3")
        token = sign_risk_token("ord_keyid")
        parts = token.split(":")
        assert len(parts) == 4
        assert parts[2] == "v3"

    def test_default_key_id(self) -> None:
        """When AIS_RISK_HMAC_KEY_ID is not set, the default key_id is used."""
        import os

        os.environ.pop("AIS_RISK_HMAC_KEY_ID", None)
        token = sign_risk_token("ord_default_kid")
        parts = token.split(":")
        assert parts[2] == _DEFAULT_KEY_ID

    # ---------------------------------------------------------------
    # 5. No previous key = single-key behavior (backward compatible)
    # ---------------------------------------------------------------

    def test_single_key_behavior_without_previous(self) -> None:
        """Without AIS_RISK_HMAC_SECRET_PREVIOUS, the system operates
        in single-key mode — only the current key is tried."""
        import os

        os.environ.pop("AIS_RISK_HMAC_SECRET_PREVIOUS", None)

        token = sign_risk_token("ord_single")
        assert verify_risk_token(token, "ord_single")

    def test_single_key_rejects_wrong_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In single-key mode, a token signed with a different key is
        rejected (no fallback to try)."""
        # Sign with one secret
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET", "secret-a")
        monkeypatch.delenv("AIS_RISK_HMAC_SECRET_PREVIOUS", raising=False)
        token = sign_risk_token("ord_rej")

        # Switch to a different secret
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET", "secret-b")

        assert not verify_risk_token(token, "ord_rej")

    # ---------------------------------------------------------------
    # Edge cases
    # ---------------------------------------------------------------

    def test_expired_token_rejected_even_with_previous_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An expired token is rejected regardless of key validity.
        The TTL check applies before key fallback."""

        old_secret = "old-secret"
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET", old_secret)

        # Craft an expired token manually
        order_id = "ord_expired"
        expired_timestamp = str(int(time.time()) - TOKEN_TTL_SECONDS - 10)
        key_id = _DEFAULT_KEY_ID
        payload = f"{order_id}:{expired_timestamp}:{key_id}"
        sig = hmac_mod.new(
            old_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        expired_token = f"{payload}:{sig}"

        # Even with the old key as previous, TTL rejects it
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET", "new-secret")
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET_PREVIOUS", old_secret)

        assert not verify_risk_token(expired_token, order_id)

    def test_new_tokens_always_signed_with_current_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even during rotation, new tokens are always signed with the
        current (primary) key, not the previous key."""
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET", "current-key")
        monkeypatch.setenv("AIS_RISK_HMAC_SECRET_PREVIOUS", "old-key")
        monkeypatch.setenv("AIS_RISK_HMAC_KEY_ID", "v2")

        token = sign_risk_token("ord_new")

        # Verify succeeds with current key (no fallback needed)
        # Remove previous key to prove it did not use the old key
        monkeypatch.delenv("AIS_RISK_HMAC_SECRET_PREVIOUS", raising=False)
        assert verify_risk_token(token, "ord_new")
