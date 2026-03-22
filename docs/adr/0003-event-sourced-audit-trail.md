# ADR-0003: Event-Sourced Audit Trail with SQLite EventStore

## Status

Accepted

## Date

2026-03-22

## Context

An autonomous trading system that manages real capital must maintain a complete,
tamper-resistant record of every decision, order, risk event, fill, and
reconciliation result. This audit trail serves three purposes: (1) regulatory
compliance -- demonstrating that risk controls were applied to every order,
(2) crash recovery -- reconstructing system state from persisted events and
checkpoints rather than requiring the trading loop to remain running, and
(3) performance attribution -- replaying decisions to understand which agents
and strategies contributed to portfolio returns.

The system operates as a single-process Python application (the trading loop)
with an optional FastAPI control plane. It runs on modest hardware and should
not require external infrastructure (databases, message queues) for local
development or single-node deployment. However, the persistence layer must be
durable enough that a process crash does not lose any events that were confirmed
as written.

Additionally, the system needs checkpoint capability. The SharedMemory state
(agent beliefs, market regimes, portfolio snapshots) must be periodically
saved so that a restart can pick up from the last checkpoint rather than
replaying all events from the beginning of time.

## Decision

All state-changing operations are recorded in an append-only `EventStore`
backed by SQLite, defined in `src/aiswarm/data/event_store.py`.

**Schema** -- two tables:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL,    -- JSON-serialized event data
    source TEXT DEFAULT '',   -- originating component
    created_at TEXT NOT NULL
);

CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL,    -- JSON-serialized state snapshot
    created_at TEXT NOT NULL
);
```

Both tables have indexes on `event_type`/`checkpoint_type` and `timestamp` for
efficient filtered queries.

**Event operations:**

- `append(event_type, payload, source)` -- generic append. Returns the event ID.
- `append_decision(decision)` -- convenience for `event_type="decision"`.
- `append_order(order)` -- convenience for `event_type="order"`.
- `append_risk_event(risk_event)` -- for risk validation outcomes.
- `append_fill(fill)` -- for execution fills from the exchange.
- `append_reconciliation(result)` -- for reconciliation pass results.

All writes are append-only. Events are never updated or deleted. The payload is
JSON-serialized with `default=str` to handle datetime and UUID types. Each event
carries both a logical `timestamp` (when the event occurred in the domain) and a
`created_at` (when it was persisted).

**Checkpoint operations:**

- `save_checkpoint(checkpoint_type, payload)` -- generic checkpoint write.
- `load_latest_checkpoint(checkpoint_type)` -- loads the most recent checkpoint
  of a given type (ORDER BY id DESC LIMIT 1).
- `save_portfolio_checkpoint(snapshot)` / `load_portfolio_checkpoint()` --
  convenience wrappers for portfolio state.
- `save_memory_checkpoint(memory_state)` / `load_memory_checkpoint()` --
  convenience wrappers for SharedMemory state.

Checkpoints are also append-only. Loading always returns the most recent. Old
checkpoints are retained for historical analysis but are not used during
recovery.

**Query operations:**

- `get_events(event_type, since, limit)` -- filtered query with optional type
  and time range. Returns events in reverse chronological order (most recent
  first).
- `get_decisions(limit)` / `get_orders(limit)` -- convenience wrappers.
- `count_events(event_type)` -- count with optional type filter.

**Connection management:**

Each operation opens a fresh `sqlite3.connect()`, uses a context manager for
commit/close, and sets `row_factory = sqlite3.Row` for dict-like access. This
avoids connection pooling complexity and is safe for the single-writer pattern
of the trading loop.

**File location:**

Default path is `data/ais_events.db` (configurable via `AIS_DB_PATH`). The
parent directory is created automatically via `Path.mkdir(parents=True)`.

## Consequences

### Positive

- **Full replay capability**: Every decision, order, risk event, and fill is
  recorded with enough context to reconstruct the decision-making process.
  Useful for post-mortem analysis and strategy tuning.
- **Checkpoint-based recovery**: On restart, the system loads the latest
  portfolio and SharedMemory checkpoints, then replays only events since the
  checkpoint timestamp. Recovery time is proportional to the checkpoint
  interval, not total event history.
- **Zero external dependencies**: SQLite is in the Python standard library. No
  database server, no connection strings, no migrations framework. The entire
  audit trail is a single file that can be copied, backed up, or analyzed with
  any SQLite tool.
- **Append-only integrity**: Events cannot be retroactively modified. The
  autoincrement ID provides a total ordering guarantee. This simplifies
  reasoning about event sequences during replay.
- **Structured logging complement**: The EventStore captures structured domain
  events; the JSON log captures operational telemetry. Together they provide
  both business-level and infrastructure-level observability.

### Negative

- **Single-writer limitation**: SQLite supports one writer at a time. If the
  FastAPI control plane writes events concurrently with the trading loop,
  write contention can cause `SQLITE_BUSY` errors. Mitigated by the control
  plane being read-heavy and the trading loop being the sole writer in normal
  operation.
- **No built-in replication**: The EventStore is a local file. If the host
  fails, the audit trail is lost unless external backup is configured.
  Mitigated by running the database on a durable volume and periodic backup
  to object storage.
- **Unbounded growth**: The events table grows without bound. A year of
  production trading at 60-second cycles with 5-10 events per cycle produces
  ~2.6M-5.2M rows. SQLite handles this comfortably but the file size will
  grow to hundreds of MB. No automatic archival or compaction is implemented.
- **JSON payload overhead**: Storing payloads as JSON text is less space-efficient
  and slower to query than normalized columns. Mitigated by the payloads being
  relatively small (typically < 1KB) and queries primarily filtering on indexed
  columns (`event_type`, `timestamp`).

### Neutral

- The EventStore is not a full event-sourcing implementation (no projections, no
  event handlers, no saga support). It is a persistence layer with event-sourcing
  semantics. The distinction is intentional: full event sourcing adds complexity
  that is not justified for a single-process trading system.

## Alternatives Considered

### PostgreSQL Event Store

Use PostgreSQL with a proper events table, LISTEN/NOTIFY for event consumers,
and JSONB for queryable payloads. Rejected because: (1) adds an external
dependency that must be running for the system to start, (2) PostgreSQL is
overkill for a single-process application with one writer, (3) operational
burden of managing a database server for local development, (4) SQLite's
single-file portability is a significant advantage for development and testing.

### Apache Kafka / Event Streaming

Use Kafka or Redpanda for durable, ordered event streams with consumer groups.
Rejected because: (1) massive operational overhead for a single-process system,
(2) introduces network latency on every event append, (3) requires JVM
infrastructure (Kafka) or additional binary (Redpanda), (4) consumer group
semantics are unnecessary when there is one consumer (the replay engine).

### In-Memory Event Log with Periodic Flush

Keep events in a Python list and flush to disk periodically. Rejected because:
(1) events between the last flush and a crash are lost, which is unacceptable
for a financial system, (2) memory consumption grows unboundedly during long
sessions, (3) no query capability without loading the full log.

### File-based Append Log (JSONL)

Append JSON lines to a flat file, one event per line. Rejected because:
(1) no indexed queries -- finding all risk events requires scanning the entire
file, (2) no atomic writes -- a crash during write can corrupt the last line,
(3) no checkpoint table -- would need a separate mechanism for state snapshots,
(4) SQLite provides all of these features with minimal additional complexity.

## References

- `src/aiswarm/data/event_store.py` -- `EventStore` class
- `src/aiswarm/orchestration/memory.py` -- `SharedMemory` (checkpoint consumer)
- `src/aiswarm/loop/` -- trading loop (checkpoint producer)
- `src/aiswarm/review/` -- session review generator (event consumer)
