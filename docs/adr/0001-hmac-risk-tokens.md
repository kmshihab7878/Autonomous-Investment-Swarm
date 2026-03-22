# ADR-0001: HMAC-SHA256 Risk Approval Tokens

## Status

Accepted

## Date

2026-03-22

## Context

The Autonomous Investment Swarm executes orders autonomously across multiple
exchanges. A core safety requirement is that no order reaches an exchange without
cryptographic proof that the risk engine has evaluated and approved it. Without
this guarantee, a bug in the coordinator, a race condition in the execution
pipeline, or a compromised agent could bypass risk controls entirely and submit
unvalidated orders to live markets.

The risk engine integrates five independent guards (KillSwitch, ExposureManager,
DrawdownGuard, LeverageGuard, LiquidityGuard) plus per-mandate constraints. The
output of a successful validation needs to be a tamper-evident artifact that
travels with the order through the execution pipeline. This artifact must be
verifiable at the point of submission without requiring a callback to the risk
engine, because the executor operates asynchronously and may be on a different
process boundary.

Additionally, the system needs to support zero-downtime key rotation for the
signing secret. In a production environment, rotating HMAC keys must not
invalidate tokens that were signed with the previous key but have not yet been
submitted.

## Decision

Risk approval is represented as an HMAC-SHA256 signed, TTL-bound token with the
format:

```
{order_id}:{timestamp}:{key_id}:{signature}
```

The token lifecycle works as follows:

1. **Signing** (`sign_risk_token`): When `RiskEngine.validate()` approves an
   order, it calls `sign_risk_token(order_id)`. The function constructs a
   payload of `{order_id}:{unix_timestamp}:{key_id}`, computes
   `HMAC-SHA256(payload, AIS_RISK_HMAC_SECRET)`, and returns the full
   `payload:signature` string. The approved order is returned as a Pydantic
   `model_copy` with the token in `risk_approval_token` and status set to
   `APPROVED`.

2. **Verification** (`verify_risk_token`): Before submitting to any exchange,
   the executor calls `verify_risk_token(token, order_id)`. Verification
   checks:
   - Token has exactly 4 colon-separated parts.
   - The embedded `order_id` matches the order being submitted.
   - The timestamp is within the 300-second (5-minute) TTL window and is not
     in the future.
   - The HMAC signature matches using the current key.
   - If the current key fails, the previous key (`AIS_RISK_HMAC_SECRET_PREVIOUS`)
     is tried, enabling a rotation window.

3. **Key rotation**: The system supports two concurrent keys. The current key
   (`AIS_RISK_HMAC_SECRET`) always signs new tokens. Verification tries the
   current key first, then falls back to `AIS_RISK_HMAC_SECRET_PREVIOUS`. A
   `key_id` field (defaulting to `v1`, configurable via `AIS_RISK_HMAC_KEY_ID`)
   makes tokens self-identifying for auditability. When a token verifies against
   the previous key, a warning is logged to track rotation progress.

4. **Fail-closed semantics**: If `AIS_RISK_HMAC_SECRET` is unset or empty, the
   `_hmac_secret()` helper raises `RuntimeError`. There is no fallback secret
   and no default key. The system refuses to start rather than operate without
   signing capability.

The signing and verification functions live in `src/aiswarm/risk/limits.py`
alongside the `RiskEngine` class, keeping the cryptographic boundary co-located
with the risk decision boundary.

## Consequences

### Positive

- **Tamper evidence**: An order cannot be modified after risk approval without
  invalidating the token, because the order_id is embedded in the signed payload.
- **Replay prevention**: The 300-second TTL prevents stale approvals from being
  reused after market conditions have changed.
- **Zero-downtime rotation**: Two-key verification means secrets can be rotated
  without a maintenance window. Operators set the new key as primary, move the
  old key to `_PREVIOUS`, and wait for the TTL window to drain.
- **Auditability**: The `key_id` field in every token creates a clear audit trail
  of which key signed which order, useful during incident response.
- **No network dependency**: Verification is a pure function (HMAC comparison)
  with no database or network calls, keeping the hot path fast.

### Negative

- **Clock sensitivity**: The TTL check relies on system clock accuracy. A clock
  skew greater than 300 seconds between the risk engine and executor would cause
  valid tokens to be rejected. Mitigated by running both in the same process and
  using NTP.
- **Single-use gap**: The token does not enforce single-use semantics. A token
  could theoretically be submitted twice within the TTL window. Mitigated by the
  OrderStore tracking submission state and the exchange deduplicating by
  client order ID.
- **Secret management burden**: Operators must manage two environment variables
  during rotation windows. Incorrect configuration (e.g., setting both to the
  same value) is harmless but wasteful.

### Neutral

- Token format is string-based rather than JWT. This is intentional: JWTs add
  parsing overhead and library dependencies for a token that never leaves the
  process boundary.

## Alternatives Considered

### JWT-based Risk Tokens

JSON Web Tokens with RS256 or ES256 signatures. Rejected because: (1) JWTs are
designed for cross-system token exchange; our tokens never leave the process,
(2) JWT parsing libraries add dependencies and attack surface, (3) the
`header.payload.signature` format is heavier than our 4-field format for a
token that is created and consumed millions of times.

### Database-backed Approval Records

Write an approval record to the EventStore and have the executor query it before
submission. Rejected because: (1) introduces a database read on the critical
execution path, adding latency and a failure mode, (2) the EventStore is
append-only and not designed for point lookups under load, (3) the approval
check becomes a distributed coordination problem if the executor runs in a
separate process.

### Shared-memory Flag on the Order Object

Set a boolean `risk_approved = True` on the order without cryptographic proof.
Rejected because: (1) any code path that constructs an Order can set the flag,
(2) provides no tamper detection, (3) violates the invariant that risk approval
must be cryptographically verifiable.

### Asymmetric Signing (RSA/ECDSA)

Use public-key cryptography so the executor only needs the public key. Rejected
because: (1) the risk engine and executor run in the same trust domain, so there
is no separation-of-privilege benefit, (2) asymmetric operations are ~100x
slower than HMAC for equivalent security, (3) key management is more complex
(certificates, key pairs, rotation involves both keys).

## References

- `src/aiswarm/risk/limits.py` -- `sign_risk_token`, `verify_risk_token`, `RiskEngine`
- `src/aiswarm/execution/aster_executor.py` -- token verification before submission
- `src/aiswarm/types/orders.py` -- `Order.risk_approval_token` field
