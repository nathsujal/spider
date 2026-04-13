# Brainstorming: spider-py

## Project Overview

- **Purpose**: PyO3 Python bindings for Spider — a bio-inspired AI agent memory graph database engine written in pure Rust. spider-py exposes `spider-core`'s synchronous, blocking database API to Python, enabling Python applications and AI agent frameworks to use Spider's graph storage, document ingestion, bio-inspired scoring, and graph traversal directly from Python.
- **Current State**: Empty `lib.rs` — a skeleton crate exists with PyO3 and spider-core as dependencies, configured as a `cdylib` + `rlib` crate. The `[lib]` name is `spider` (so the compiled `.so`/`.dll`/`.dylib` becomes `spider` — the Python importable module name).
- **Goals**:
  1. Provide a Pythonic, idiomatic Python API (`import spider`) that wraps all core spider-core functionality
  2. Maintain minimal-copy data transfer between Rust and Python where possible
  3. Support non-blocking Python usage via `py.allow_threads()` (GIL release during I/O)
  4. Enable pip-installable distribution via maturin
  5. Fit into the parent workspace ecosystem — same version, same release cadence

## Explored Approaches

### Approach 1: Direct PyO3 Wrapping (Recommended)
- **Description**: Wrap each spider-core public type (`Spider`, `Node`, `Edge`, etc.) as a PyO3 `#[pyclass]` with `#[pymethods]`. The Python classes directly mirror the Rust types. Database operations run on the GIL-released thread pool via `py.allow_threads()`.
- **Pros**:
  - Straightforward mapping — one Rust struct → one Python class
  - PyO3 has mature GIL management, `allow_threads` for releasing the GIL during blocking I/O
  - Minimal overhead — direct FFI calls into Rust
  - Well-documented ecosystem (maturin, pyo3-build-config)
  - Can use `#[pyfunction]` for free functions like `find_by_label`, `index`, etc.
- **Cons**:
  - Requires careful lifetime management — `Spider` holds `&mut` to all file handles; Python may hold references across calls
  - PyO3 `pyclass` types cannot have generic parameters or lifetimes — must own or use `Arc<Mutex<>>`
  - Error translation layer needed: `DbError` → Python exceptions
- **Complexity**: Medium
- **Recommended for**: This is the standard and most maintainable approach for Rust→Python bindings

### Approach 2: Opaque Handle Pattern
- **Description**: Instead of exposing Rust structs directly, use opaque integer handles (like database connection IDs). Python passes handles to functions that look up the real Rust objects from a global registry.
- **Pros**:
  - Avoids lifetime/GIL complexity entirely
  - Easy to serialize/deserialize handles for IPC
  - Clean separation between Rust internals and Python API
- **Cons**:
  - Requires global mutable state (`HashMap<u64, Spider>`)
  - Handle lifecycle management (leak prevention, cleanup on GC)
  - Less Pythonic — feels like a C API wrapper
  - Debugging is harder (opaque handles in tracebacks)
- **Complexity**: Medium-High
- **Recommended for**: When Rust types have complex lifetimes that can't be expressed in PyO3 — not needed here since `Spider` can be `Arc<Mutex<>>`

### Approach 3: IPC/FFI Bridge (gRPC, Unix sockets)
- **Description**: Run spider-daemon as a background process; spider-py communicates via gRPC or Unix domain sockets.
- **Pros**:
  - Complete isolation — Python crashes don't corrupt Rust state
  - Can share the daemon across multiple Python processes
  - Natural fit for the existing spider-daemon HTTP API
- **Cons**:
  - Significant overhead for a database that's designed for O(1) direct access
  - Defeats the purpose of spider-core being a synchronous, in-process engine
  - Complexity of managing daemon lifecycle from Python
  - Serialization overhead on every call
- **Complexity**: High
- **Recommended for**: Multi-process scenarios or when the daemon is already running — not for direct bindings

### Approach 4: Runtime-generated Python Code (cffi/FFI gen)
- **Description**: Use cffi or generate Python stubs from Rust types automatically.
- **Pros**:
  - Could auto-generate bindings from schema definitions
  - Less manual boilerplate
- **Cons**:
  - Mature tooling is limited (cbindgen exists but not for Python)
  - Generated code lacks Pythonic ergonomics
  - Debugging generated code is difficult
- **Complexity**: High
- **Recommended for**: Large APIs with frequent schema changes — overkill for this project

## Architecture Considerations

### Module Structure (FLAT — ~15 public functions, splitting across subdirectories adds cognitive overhead)

```
spider-py/
├── src/
│   ├── lib.rs              # #[pymodule], exception registration
│   ├── error.rs            # DbError → PyErr mapping, exception definitions
│   ├── spider.rs           # PySpider class (Arc<parking_lot::Mutex<Spider>>) — ALL methods here
│   ├── types.rs            # PyNodeId, PyEdgeId, PyBioTier, PyDirection
│   └── ingest.rs           # PyIngestRequest, PyProposition, PyEntity, PyIngestResult
├── python/
│   └── spider/
│       ├── __init__.py     # Re-export from native module, add Python helpers
│       └── types.py        # Python type hints / stubs
├── tests/                  # Python integration tests
├── Cargo.toml
└── pyproject.toml          # maturin build config
```

**Rationale for flat structure**: With only ~15 public functions, splitting across `types/` and `operations/` subdirectories adds cognitive overhead. All Spider methods live in `spider.rs`; types go in `types.rs`; ingestion types go in `ingest.rs`.

### Public API Design

The Python API should feel natural to Python developers while faithfully representing spider-core's semantics:

```python
import spider

# Database lifecycle (context manager)
with spider.Spider.open("./my_graph") as db:
    # Ingestion
    from spider import IngestRequest, Proposition, Entity

    req = IngestRequest(
        title="My Document",
        propositions=[
            Proposition(
                text="Mumbai is the financial capital of India",
                entities=[
                    Entity(name="Mumbai", entity_type="LOCATION"),
                    Entity(name="India", entity_type="LOCATION"),
                ],
            ),
        ],
    )
    result = db.index(req)
    print(f"Document ID: {result.document_id}")

    # Queries
    nodes = db.find_by_label("ENTITY")
    mumbai_nodes = db.find_by_property("name", "Mumbai")

    # Traversal
    neighbors = db.get_neighbors(node_id, direction="outgoing")
    for neighbor in neighbors:
        print(f"Node {neighbor.node_id} via edge {neighbor.edge_id}")

    # Bio scoring
    tier = db.get_bio_tier(node_id)  # Hot, Warm, Cold, Pruned
```

### Error Handling Strategy

- **Single `SpiderError` base exception** with 5 sub-exceptions:
  - `SpiderNotFoundError` — NodeNotFound, EdgeNotFound, BlobNotFound, etc.
  - `SpiderCorruptError` — CorruptMetadata, CorruptRecordFile
  - `SpiderIOError` — FileOpen, Io
  - `SpiderIngestionError` — DocumentNodeNotFound, NoPropositions
  - `SpiderTraversalError` — TraversalDepthExceeded
- Implement `impl From<spider_core::error::DbError> for PyErr` in `error.rs`
- All sub-exceptions inherit from `SpiderError` for catch-all error handling
- Error messages reuse `DbError::Display` output for consistency with spider-core

### Dependency Decisions

| Dependency | Decision | Rationale |
|---|---|---|
| `pyo3` | **Include** | Standard Rust→Python bindings |
| `parking_lot` | **Include** (new) | Non-poisoning mutex, smaller footprint (1 byte vs 48 bytes), better performance under contention |
| `maturin` | **Include** (dev/build) | Standard PyO3 build tool, handles wheel building |
| `thiserror` | **Not needed** | spider-core already has `DbError`; we translate to PyErr directly |
| `anyhow` | **Not needed** | We have a concrete error type (`DbError`) — no dynamic error chains |
| `serde`/`serde_json` | **Include** | For serializing ingestion request types from Python dicts |
| `tokio` | **Deferred** | Not in MVP. spider-core is sync-only. Can add as optional feature flag later. |

### Feature Flags

```toml
[features]
default = []
extension-module = ["pyo3/extension-module"]
```

**Note**: `extension-module` is **NOT** in `default` because it breaks `cargo test`. Users building wheels must explicitly enable it: `maturin build --features extension-module`.

### GIL Release Pattern (Critical Coding Standard)

**ALL I/O operations MUST release the GIL** using `py.allow_threads()`. This is not optional — forgetting it blocks all Python threads during Rust I/O.

```rust
// CORRECT pattern — convert BEFORE GIL release
fn index(&self, py: Python<'_>, request: &PyIngestRequest) -> PyResult<PyIngestResult> {
    let inner = Arc::clone(&self.inner);
    let rust_request = request.to_rust()?;  // Convert BEFORE GIL release

    py.allow_threads(move || {
        let mut db = inner.lock().unwrap();
        spider_core::db::ingest::index(&mut db, &rust_request)
            .map(PyIngestResult::from)
            .map_err(Into::into)
    })
}
```

**Key rules**:
1. Convert Python types to Rust **before** calling `py.allow_threads()`
2. Access Python objects **before** GIL release — accessing them inside `allow_threads` causes segfaults
3. Return only Rust-native types (or types that don't hold Python GIL references) from the closure

### Rust-Python Interop Patterns

1. **`Spider` handle**: `Arc<parking_lot::Mutex<spider_core::db::Spider>>` inside a `#[pyclass]`. `parking_lot::Mutex` is used instead of `std::sync::Mutex` because:
   - Non-poisoning: if a panic occurs, subsequent calls still work
   - Smaller footprint: 1 byte vs 48 bytes
   - Better performance under contention

2. **IngestRequest conversion**: Python types own `String`, Rust types use borrowed `&str`. Convert at call time by building Rust structs with `&str` slices pointing into Python-owned strings (valid for duration of method call).

3. **String handling**: Python `str` → Rust `&str` via PyO3's `Bound<'_, PyString>`. Short strings are zero-copy within the GIL context.

4. **Numeric types**: `u32`, `f64`, `i64` map directly. `NodeId` and `EdgeId` are transparent wrappers.

5. **Collection types**: `Vec<T>` ↔ Python `list`. PyO3 handles this automatically.

6. **Enum types**: `Direction` → Python `str` ("outgoing", "incoming", "both") or `enum.IntEnum`. `BioTier` → Python `enum.Enum` with string names.

### Context Manager

The `Spider` class implements `__enter__`/`__exit__` for idiomatic Python usage:

```python
with spider.Spider.open("./my_graph") as db:
    db.index(request)
# auto-closed on exit
```

## Key Decisions Made

1. **Approach 1: Direct PyO3 Wrapping with `Arc<parking_lot::Mutex<>>`** — The `Spider` handle wraps the inner `spider_core::db::Spider` in `Arc<parking_lot::Mutex<...>>`. `parking_lot` chosen over `std::sync::Mutex` for non-poisoning behavior, smaller footprint, and better contention performance.

2. **FLAT module structure** — With ~15 public functions, all Spider methods live in `spider.rs`. Types in `types.rs`, ingestion types in `ingest.rs`. No `types/` or `operations/` subdirectories.

3. **Single `SpiderError` base + 5 sub-exceptions** — `SpiderNotFoundError`, `SpiderCorruptError`, `SpiderIOError`, `SpiderIngestionError`, `SpiderTraversalError`. All inherit from `SpiderError`.

4. **Synchronous-only for MVP** — Async is deferred entirely. spider-core is sync-only, and `allow_threads()` already achieves non-blocking Python. Can add as optional feature flag later when users demand it.

5. **Module name = `spider`** — The `[lib]` name in Cargo.toml is `spider`, so Python users `import spider`.

6. **Methods on Spider class** — spider-core's `db/` functions take `&mut Spider` as first argument. In Python, these become methods on the `Spider` class for a natural OOP feel.

7. **No Python-side caching** — All data lives in Rust. Python is a thin pass-through layer. This avoids consistency issues.

8. **GIL release is mandatory** — Every I/O operation uses `py.allow_threads()`. Forgetting this is a critical bug that blocks all Python threads.

9. **`extension-module` NOT in default features** — Breaks `cargo test`. Must be explicitly enabled during wheel building.

10. **Context manager support** — `__enter__`/`__exit__` implemented on `Spider` class.

## Common Pitfalls (Documented for Contributors)

1. **Forgetting `py.allow_threads()` before I/O** — Holds the GIL, blocks all Python threads. This is the #1 performance bug in PyO3 projects.

2. **Accessing Python objects after GIL release** — Causes segfaults. Convert all Python types to Rust before calling `py.allow_threads()`.

3. **`extension-module` feature breaks `cargo test`** — Use `cargo test` without features, `maturin build --features extension-module` for wheels. Separate build commands.

4. **Need to test on Python 3.9–3.13** — Different Python versions have different GIL behavior. Test across the full range.

5. **Mutex poisoning** — `parking_lot::Mutex` doesn't poison, but if spider-core panics internally, the `Spider` state may be inconsistent. Document this and test resilience.

6. **Memory leaks from Arc cycles** — Python's GC doesn't collect `Arc` cycles. The `Spider` class should implement `__del__` or `close()` to break the cycle. Test with `gc.collect()` and memory profiling.

## Open Questions

1. **Should we expose raw `Node`/`Edge` structs to Python?** — These are internal 29/33-byte records. Python users likely don't need direct access — they need IDs, labels, properties, and traversal results. Exposing them would require copying the full struct definition.

2. **Property read/write API** — spider-core's `ingest.rs` has `set_string_property` and `set_int_property` as private functions. Should spider-py expose a general `db.set_property(node_id, key, value)` method? **Deferred to Phase 5** — depends on whether spider-core exposes these publicly.

3. **Content-addressable blob storage** — `db/content.rs` exists in spider-core. Should this be exposed? It's useful for storing raw document text alongside the graph. **Deferred to Phase 5 (post-MVP)** — can be added once core API is stable.

4. **Bio parameter tuning** — Should Python users be able to modify bio scoring parameters (w_sig, w_freq, gravity), or only read them from metadata?

## Recommendations

Proceed with **Approach 1 (Direct PyO3 Wrapping)** with the following implementation priorities:

**Phase 1 (Foundation)**: Build setup with `parking_lot`, error types, Spider handle with open/close, context manager
**Phase 2 (Core Types & Ingestion)**: NodeId/EdgeId wrappers, ingestion types, index() method with explicit GIL release
**Phase 3 (Queries & Traversal)**: Find queries, graph traversal, bio scoring
**Phase 4 (Polish & Packaging)**: Type stubs, integration tests, GIL release verification, memory leak detection, mutex poisoning resilience, wheel publishing
**Phase 5 (Deferred/Post-MVP)**: Blob storage, property read/write API, async support (only if users demand it)

The API should be Pythonic but faithful to spider-core's design. Every Python method should have a direct counterpart in spider-core. The `Arc<parking_lot::Mutex<>>` wrapper is the key architectural decision that makes this feasible — it resolves the `&mut self` lifetime problem that PyO3 `pyclass` types cannot express, with non-poisoning behavior and minimal overhead.
