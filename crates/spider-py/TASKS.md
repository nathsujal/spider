# Tasks: spider-py

## Overview

Implementation plan for PyO3 Python bindings exposing spider-core's database engine to Python. The API provides `import spider` with a `Spider` class for database operations, ingestion, querying, and graph traversal. The architecture uses `Arc<parking_lot::Mutex<spider_core::db::Spider>>` inside a single `PySpider` class (flat module structure), with mandatory GIL release (`py.allow_threads()`) for all I/O operations. Async is deferred from MVP.

## Task List

### Phase 1: Foundation

#### Task 1: Project scaffolding and build configuration
- **Status**: ✅ COMPLETED
- **Description**: Set up maturin build configuration, pyproject.toml, and the Rust module skeleton with PyO3 initialization. Update Cargo.toml with correct dependencies.
- **Scope**:
  - Include: `pyproject.toml`, `Cargo.toml` updates, `src/lib.rs` PyO3 module setup
  - Out of scope: Actual binding implementations, Python helper modules
- **Technical Details**:
  - Create `pyproject.toml` with maturin backend:
    ```toml
    [build-system]
    requires = ["maturin>=1.0,<2.0"]
    build-backend = "maturin"

    [project]
    name = "spider-py"
    version = "0.1.0"
    description = "Python bindings for Spider — bio-inspired AI agent memory graph"
    requires-python = ">=3.9"
    license = { text = "MIT" }
    ```
  - Update `Cargo.toml`:
    - Add `parking_lot = "0.12"` dependency
    - **Remove** `anyhow` (not needed — spider-core has concrete `DbError`)
    - Keep `pyo3`, `serde`, `serde_json`
    - Set feature flags: `default = []`, `extension-module = ["pyo3/extension-module"]`, `gil-refs = ["pyo3/gil-refs"]`
    - **Do NOT** include `extension-module` in default (breaks `cargo test`)
  - Create `src/lib.rs` with `#[pymodule] fn spider(m: &Bound<'_, PyModule>) -> PyResult<()>` that registers the module and exceptions
  - Create empty `src/error.rs`, `src/spider_handle.rs`, `src/types.rs`, `src/ingest.rs` module declarations
- **Acceptance Criteria**:
  - [x] `maturin develop` builds successfully in a virtual environment
  - [x] `python -c "import spider"` works without errors
  - [x] `cargo check` passes (without `extension-module` feature)
  - [x] `cargo check --features extension-module` passes
  - [x] `cargo clippy` reports no warnings
  - [x] `Cargo.toml` has `parking_lot`, no `anyhow`, correct feature flags
- **Dependencies**: None
- **Estimated Complexity**: S
- **Implementation Notes**: PyO3 uses `Bound<'_, PyModule>`. Remember: `cargo test` requires `--no-default-features` or simply no feature flags since `extension-module` is not in default. Renamed `spider.rs` to `spider_handle.rs` to avoid naming conflict with `#[pymodule] fn spider`.

#### Task 2: Python exception hierarchy
- **Status**: ✅ COMPLETED
- **Description**: Define the Python exception classes that map from `spider_core::error::DbError`.
- **Scope**:
  - Include: `src/error.rs` with `SpiderError` base exception and 5 sub-exceptions
  - Out of scope: Error mapping functions (covered in later tasks)
- **Technical Details**:
  - Create base exception:
    ```rust
    create_exception!(spider, SpiderError, PyException);
    ```
  - Create 5 sub-exceptions, all inheriting from `SpiderError`:
    - `SpiderNotFoundError` — NodeNotFound, EdgeNotFound, SourceNodeNotFound, TargetNodeNotFound, BlobNotFound, NotFound, IdOutOfRange, NodeDeleted, PropertyNotFound, DocumentNodeNotFound
    - `SpiderCorruptError` — CorruptMetadata, CorruptRecordFile, BlobHashMismatch
    - `SpiderIOError` — FileOpen, Io, PathNotFound, VersionMismatch
    - `SpiderIngestionError` — NoPropositions
    - `SpiderTraversalError` — TraversalDepthExceeded
    - NodeError, EdgeError, PropertyError, TokenError, DynamicError → SpiderNotFoundError (schema errors treated as not found)
  - Implement `pub fn db_error_to_pyerr(err: spider_core::error::DbError) -> PyErr` that maps each variant to the appropriate Python exception with a descriptive message
  - Register all exceptions in the PyO3 module init in `lib.rs`
- **Acceptance Criteria**:
  - [x] `spider.SpiderError` is accessible from Python
  - [x] `spider.SpiderNotFoundError` is a subclass of `spider.SpiderError`
  - [x] `spider.SpiderCorruptError`, `SpiderIOError`, `SpiderIngestionError`, `SpiderTraversalError` all accessible and subclass `SpiderError`
  - [x] `db_error_to_pyerr(DbError::NodeNotFound(42))` produces a `SpiderNotFoundError` with message containing "42"
  - [x] Unit tests verify the error mapping for each variant
  - [x] All sub-exceptions inherit from `SpiderError` (verified via exception names in error messages)
- **Dependencies**: Task 1
- **Estimated Complexity**: S
- **Implementation Notes**: Use `pyo3::exceptions::PyException` as the base. The `create_exception!` macro generates the Python exception class. Error messages reuse `DbError::Display` output for consistency with spider-core. Can't implement `From<DbError> for PyErr` due to orphan rules (both types are external), so using a helper function instead.

#### Task 3: Spider database handle class with context manager
- **Description**: Implement the `PySpider` class wrapping `Arc<parking_lot::Mutex<spider_core::db::Spider>>` with open/close lifecycle methods and context manager support.
- **Scope**:
  - Include: `src/spider.rs` — `PySpider` class with `open()`, `open_default()`, `close()`, `__enter__`/`__exit__`
  - Out of scope: Operational methods (index, find, traverse) — covered in later phases
- **Technical Details**:
  ```rust
  #[pyclass]
  pub struct PySpider {
      inner: Arc<parking_lot::Mutex<spider_core::db::Spider>>,
  }
  ```
  - `#[classmethod] fn open(_cls: &Bound<'_, PyType>, path: &str, py: Python<'_>) -> PyResult<Self>` — wraps `Spider::open(Path::new(path))`
  - `#[classmethod] fn open_default(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self>` — wraps `Spider::open_default()`
  - `fn close(&self, py: Python<'_>) -> PyResult<()>` — calls `inner.lock().close()` inside `py.allow_threads()`
  - Context manager: `__enter__` returns `self`, `__exit__` calls `close()` (silently ignores errors to match spider-core's Drop behavior)
  - `__repr__` for debugging (includes database path)
  - `path()` method returning the database path as a Python string
  - `parking_lot::Mutex` does NOT poison — if a panic occurs inside spider-core, subsequent calls to the same `Spider` instance will still work (though state may be inconsistent)
- **Acceptance Criteria**:
  - [ ] `db = spider.Spider.open("/tmp/test_db")` opens a database
  - [ ] `db = spider.Spider.open_default()` opens at platform default
  - [ ] `db.close()` flushes data and is idempotent (double-close is safe)
  - [ ] `with spider.Spider.open("/tmp/test_db") as db:` works as context manager
  - [ ] `db.path()` returns the database path string
  - [ ] Database is auto-closed when Python object is garbage collected
  - [ ] Opening a non-existent directory creates it automatically
  - [ ] Opening a corrupt database raises `SpiderCorruptError`
  - [ ] GIL is released during open/close operations (verified by running in a thread)
  - [ ] `parking_lot::Mutex` is used (not `std::sync::Mutex`) — verified in code review
- **Dependencies**: Tasks 1, 2
- **Estimated Complexity**: M
- **Implementation Notes**: Use `parking_lot::Mutex` — call `.lock()` (returns `MutexGuard`), NOT `.lock().unwrap()` (that's `std::sync::Mutex`). The `allow_threads` call releases the GIL so Python can do other work during I/O. Register `PySpider` in `lib.rs`.

### Phase 2: Core Types & Ingestion

#### Task 4: NodeId and EdgeId Python wrappers
- **Description**: Implement transparent wrapper types for `NodeId` and `EdgeId` that Python can use as integers but Rust can convert back.
- **Scope**:
  - Include: `PyNodeId`, `PyEdgeId` in `src/types.rs`
  - Out of scope: Node/Edge struct exposure (IDs are sufficient for the API)
- **Technical Details**:
  - `PyNodeId`: `#[pyclass]` with `__int__`, `__repr__`, `__eq__`, `__hash__`
  - Constructor: `fn __new__(raw: u32) -> PyResult<Self>` — validates non-zero
  - `as_int()` method returning the raw u32
  - Internal conversion: `impl From<&PyNodeId> for spider_core::db::nodes::NodeId`
  - Same pattern for `PyEdgeId` wrapping `spider_core::db::rels::EdgeId`
- **Acceptance Criteria**:
  - [ ] `node_id = spider.NodeId(42)` creates a NodeId
  - [ ] `int(node_id) == 42`
  - [ ] `spider.NodeId(0)` raises `SpiderError` (invalid ID)
  - [ ] `node_id == spider.NodeId(42)` is True
  - [ ] NodeId can be used as a dictionary key (hashable)
  - [ ] `repr(node_id)` returns `"NodeId(42)"` or similar
- **Dependencies**: Tasks 1, 2
- **Estimated Complexity**: S
- **Implementation Notes**: These are simple value types. The main concern is making them feel like integers in Python while maintaining type safety in Rust. PyO3's `IntoPy` and `FromPyObject` traits handle the conversion. Place in `types.rs` alongside `PyDirection` and `PyBioTier`.

#### Task 5: Direction and BioTier enums
- **Description**: Implement Python enum types for traversal direction and bio storage tier classification.
- **Scope**:
  - Include: `PyBioTier`, `PyDirection` in `src/types.rs`
  - Out of scope: Bio scoring calculation (covered in Task 9)
- **Technical Details**:
  - `PyDirection`: `#[pyclass]` enum or class with class attributes:
    - Variants: `Outgoing`, `Incoming`, `Both`
    - Accept both enum values and strings ("outgoing", "incoming", "both") in method parameters
  - `PyBioTier`: `#[pyclass]` enum:
    - Variants: `Hot`, `Warm`, `Cold`, `Pruned`
    - `is_prunable()`, `is_active()` methods
    - `__str__` returns human-readable name
  - Conversion: `impl TryFrom<&str> for Direction` for accepting string parameters
  - Conversion: `impl From<spider_core::bio::tier::BioTier> for PyBioTier`
- **Acceptance Criteria**:
  - [ ] `spider.Direction.OUTGOING` accessible
  - [ ] `spider.Direction.INCOMING` accessible
  - [ ] `spider.Direction.BOTH` accessible
  - [ ] `spider.BioTier.HOT` accessible
  - [ ] `BioTier.from_score(100.0) == BioTier.HOT`
  - [ ] `BioTier.HOT.is_active()` returns True
  - [ ] `BioTier.PRUNED.is_prunable()` returns True
- **Dependencies**: Tasks 1, 2
- **Estimated Complexity**: S
- **Implementation Notes**: PyO3 doesn't have native enum support that maps perfectly to Python enums. Consider using `pyo3::types::PyEnum` or a class with class attributes. The string-accepting approach (accepting "outgoing" as well as `Direction.OUTGOING`) is more Pythonic.

#### Task 6: Ingestion types — PyProposition, PyEntity, PyIngestRequest, PyIngestResult
- **Description**: Implement PyO3 classes for the ingestion request/response types.
- **Scope**:
  - Include: `src/ingest.rs` — all four types
  - Out of scope: The `index()` function itself (covered in Task 7)
- **Technical Details**:
  - `PyEntity`: `#[pyclass]` with `name: String`, `entity_type: String`
    - Constructor: `fn __new__(name: &str, entity_type: &str) -> Self`
    - Properties: `name`, `entity_type` (read-only)
  - `PyProposition`: `#[pyclass]` with `text: String`, `entities: Vec<PyEntity>`
    - Constructor: `fn __new__(text: &str, entities: Option<Vec<PyEntity>>) -> Self`
    - Properties: `text`, `entities`
  - `PyIngestRequest`: `#[pyclass]` with `title: String`, `propositions: Vec<PyProposition>`
    - Constructor: `fn __new__(title: &str, propositions: Option<Vec<PyProposition>>) -> Self`
    - Properties: `title`, `propositions`
    - **Critical**: Implement `fn to_rust(&self) -> spider_core::db::ingest::IngestRequest<'_>` — builds borrowed Rust struct with `&str` slices pointing into Python-owned `String` fields. The borrow lifetime is valid for the duration of the Python method call.
  - `PyIngestResult`: `#[pyclass]` — returned from `index()`, not constructed by user
    - Properties: `document_id: PyNodeId`, `proposition_count: usize`, `entity_count: usize`, `edge_count: usize`
- **Acceptance Criteria**:
  - [ ] `Entity("Mumbai", "LOCATION")` creates an entity
  - [ ] `Proposition("text", [Entity(...)])` creates a proposition
  - [ ] `IngestRequest("title", [Proposition(...)])` creates a request
  - [ ] `IngestResult` properties are accessible after ingestion
  - [ ] `IngestRequest` with empty propositions list is valid
  - [ ] Entity with empty string name is accepted (validation is server-side)
  - [ ] `to_rust()` produces a valid `IngestRequest<'_>` with correct borrowed lifetimes
- **Dependencies**: Tasks 1, 2, 4
- **Estimated Complexity**: M
- **Implementation Notes**: The conversion from Python `PyIngestRequest` to Rust's `IngestRequest<'_>` is the trickiest part. The Rust version uses borrowed `&str`, but PyO3 owns the strings. Solution: the `to_rust()` method builds `&str` slices from the Python-owned `String` fields — the borrow is valid for the duration of the method call since the Python object is held alive by the caller. Document this lifetime relationship explicitly in code comments.

#### Task 7: Ingestion — `Spider.index()` method with explicit GIL release
- **Description**: Implement the `index()` method on the `PySpider` class that wraps `spider_core::db::ingest::index()`. **This is the critical method that establishes the GIL release pattern for all subsequent I/O methods.**
- **Scope**:
  - Include: `index()` method in `src/spider.rs`
  - Out of scope: Advanced ingestion options (batching, async)
- **Technical Details**:
  ```python
  def index(self, request: IngestRequest) -> IngestResult
  ```
  Implementation pattern (must be followed for ALL I/O methods):
  ```rust
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
  Key steps:
  1. Clone `Arc` before GIL release (can't access `self` inside `allow_threads`)
  2. Convert Python `PyIngestRequest` → Rust `IngestRequest<'_>` **BEFORE** calling `py.allow_threads()`
  3. Inside `allow_threads`: lock mutex, call spider-core, convert result
  4. `DbError` → `PyErr` via `Into::into` (uses the `From<DbError>` from Task 2)
- **Acceptance Criteria**:
  - [ ] `result = db.index(request)` returns a `PyIngestResult`
  - [ ] `result.document_id` is a valid `PyNodeId`
  - [ ] `result.proposition_count` matches the number of propositions in the request
  - [ ] `result.entity_count` matches new entities created (excluding deduplicated)
  - [ ] `result.edge_count` matches expected CONTAINS + MENTIONS edges
  - [ ] Ingesting a document with zero propositions succeeds (returns zero counts)
  - [ ] Ingesting duplicate entity names deduplicates correctly (second ingestion returns entity_count=0)
  - [ ] Ingestion errors raise appropriate Python exceptions
  - [ ] **GIL is released during ingestion** — verified by running index() in a Python thread while another thread does work
- **Dependencies**: Tasks 3, 4, 6
- **Estimated Complexity**: M
- **Implementation Notes**: This method establishes the coding pattern for ALL subsequent I/O methods. The borrow conversion (step 2) must happen before GIL release. Never access Python objects inside `allow_threads`. Use this as the template for find, traverse, and bio methods.

### Phase 3: Queries & Traversal

#### Task 8: Find queries — `find_by_label()`, `find_by_property()`, `find_one_by_property()`
- **Description**: Implement label-based and property-based query methods on the `PySpider` class.
- **Scope**:
  - Include: `find_by_label`, `find_by_property`, `find_one_by_property` in `src/spider.rs`
  - Out of scope: Full-text search, indexed queries (future spider-core features)
- **Technical Details**:
  ```python
  def find_by_label(self, label: str) -> List[NodeId]
  def find_by_property(self, key: str, value: str) -> List[NodeId]
  def find_one_by_property(self, key: str, value: str) -> Optional[NodeId]
  ```
  - Each method wraps the corresponding `spider_core::db::find` function
  - Follow the GIL release pattern from Task 7: convert inputs → `allow_threads` → lock → call → convert result
  - Return `Vec<PyNodeId>` → Python `list` of `PyNodeId` objects
  - `find_one_by_property` returns `Option<PyNodeId>` → Python `NodeId | None`
- **Acceptance Criteria**:
  - [ ] `db.find_by_label("DOCUMENT")` returns list of NodeIds
  - [ ] `db.find_by_label("NONEXISTENT")` returns empty list
  - [ ] `db.find_by_property("name", "Mumbai")` finds ingested entities
  - [ ] `db.find_one_by_property("name", "Mumbai")` returns first match or None
  - [ ] `find_by_property` with unknown key returns empty list (not error)
  - [ ] Query results are consistent with ingested data
  - [ ] GIL is released during queries (verified by running in a thread)
- **Dependencies**: Tasks 3, 4, 7
- **Estimated Complexity**: S
- **Implementation Notes**: These are straightforward wrappers following the GIL release pattern established in Task 7. The `find` module in spider-core already handles all the scanning logic.

#### Task 9: Graph traversal — `get_neighbors()`, `get_relationships()`, `count_relationships()`
- **Description**: Implement graph traversal methods that walk edge chains and return connected nodes.
- **Scope**:
  - Include: All three traversal functions + `PyNeighbor` class in `src/spider.rs` and `src/types.rs`
  - Out of scope: Multi-hop traversal, pathfinding (future features)
- **Technical Details**:
  ```python
  def get_neighbors(self, node_id: NodeId, direction: Direction = Direction.BOTH) -> List[Neighbor]
  def get_relationships(self, node_id: NodeId, direction: Direction = Direction.BOTH) -> List[dict]
  def count_relationships(self, node_id: NodeId, direction: Direction = Direction.BOTH) -> int
  ```
  - `PyNeighbor` class: `node_id: PyNodeId`, `edge_id: PyEdgeId`
  - `get_relationships` returns a list of dicts with edge details: `{source_id, target_id, type}`
  - Direction parameter accepts both `PyDirection` enum and string ("outgoing", "incoming", "both")
  - All calls follow GIL release pattern from Task 7
- **Acceptance Criteria**:
  - [ ] `db.get_neighbors(doc_id, Direction.OUTGOING)` returns proposition neighbors after ingestion
  - [ ] `db.get_neighbors(prop_id, Direction.OUTGOING)` returns entity neighbors
  - [ ] `db.get_neighbors(entity_id, Direction.INCOMING)` returns the proposition that mentions it
  - [ ] `db.get_neighbors(node_id, Direction.BOTH)` returns all connected nodes
  - [ ] `db.count_relationships(node_id)` returns correct count matching `len(get_relationships(...))`
  - [ ] Query on non-existent node raises `SpiderNotFoundError`
  - [ ] Traversal depth exceeded raises `SpiderTraversalError`
  - [ ] Neighbor objects have correct `node_id` and `edge_id`
- **Dependencies**: Tasks 3, 4, 5, 7
- **Estimated Complexity**: M
- **Implementation Notes**: The `spider_core::query::traverse` module returns `Neighbor` with `NodeId` and `EdgeId`. The Python `PyNeighbor` class mirrors this. For `get_relationships`, return a list of simple dicts rather than a full Edge class — edges are complex (33-byte records with chain pointers) and Python users rarely need the raw details.

#### Task 10: Bio scoring — `get_bio_score()` and `get_bio_tier()`
- **Description**: Implement bio-inspired vitality scoring methods that calculate a node's "memory strength" and storage tier classification.
- **Scope**:
  - Include: Scoring methods in `src/spider.rs`
  - Out of scope: Bio parameter tuning (reads from Metadata, doesn't modify)
- **Technical Details**:
  ```python
  def get_bio_score(self, node_id: NodeId) -> float
  def get_bio_tier(self, node_id: NodeId) -> BioTier
  ```
  - Read node from `db.nodes.get(node_id - 1)` inside lock
  - Compute `spider_core::bio::score::calculate_with_params(&node, &BioParams { w_sig, w_freq, gravity })`
  - BioParams read from `db.metadata` (bio_w_sig, bio_w_freq, bio_gravity)
  - Tier from `spider_core::bio::tier::BioTier::from_score(score)` → `PyBioTier`
  - All calls follow GIL release pattern from Task 7
- **Acceptance Criteria**:
  - [ ] `db.get_bio_score(node_id)` returns a positive float for a live node
  - [ ] `db.get_bio_tier(node_id)` returns appropriate `PyBioTier` enum
  - [ ] Score decreases as `last_accessed_at` gets older (test by creating node with old timestamp)
  - [ ] Score increases with higher `access_count`
  - [ ] Score increases with higher `significance`
  - [ ] Non-existent node raises `SpiderNotFoundError`
  - [ ] Score computation matches Rust-side `bio::score::calculate()` output
- **Dependencies**: Tasks 3, 4, 5, 7
- **Estimated Complexity**: S
- **Implementation Notes**: The bio score function depends on `now_unix_secs()` which changes over time. For testing, create nodes with specific `last_accessed_at` timestamps and verify relative score ordering. The BioParams are stored in Metadata — use the actual values rather than defaults.

#### Task 11: Node operations — `node_count()`, `node_touch()`, `set_significance()`
- **Description**: Implement basic node management operations for controlling node metadata.
- **Scope**:
  - Include: Node count, touch, significance setters in `src/spider.rs`
  - Out of scope: Node CRUD (create/delete are internal to ingestion)
- **Technical Details**:
  ```python
  def node_count(self) -> int
  def node_touch(self, node_id: NodeId) -> int  # returns new access_count
  def set_significance(self, node_id: NodeId, significance: int) -> None
  ```
  - `node_count()` returns `metadata.next_node_id - 1`
  - `node_touch()` wraps `NodeAccess::touch()` — increments access_count, updates last_accessed_at
  - `set_significance()` wraps `NodeAccess::set_significance()` — sets significance (0-255)
  - All operations follow GIL release pattern from Task 7
- **Acceptance Criteria**:
  - [ ] `db.node_count()` returns correct count after ingestion
  - [ ] `db.node_touch(node_id)` increments access_count by 1
  - [ ] `db.set_significance(node_id, 200)` updates node's significance
  - [ ] `db.set_significance(node_id, 300)` raises error (out of range)
  - [ ] Touching non-existent node raises `SpiderNotFoundError`
  - [ ] Bio score changes after touch (verified by calling get_bio_score before/after)
- **Dependencies**: Tasks 3, 4, 7
- **Estimated Complexity**: S
- **Implementation Notes**: These methods use `spider_core::db::nodes::NodeAccess` which is a short-lived borrow pattern. In spider-core, this is accessed via `db.nodes` directly (public field). Lock the Spider mutex, then call methods on the inner fields.

### Phase 4: Polish & Packaging

#### Task 12: Python type stubs (`.pyi` files)
- **Description**: Generate or write type stub files for IDE autocompletion and type checking.
- **Scope**:
  - Include: `python/spider/__init__.pyi` with full type annotations
  - Out of scope: Runtime type checking enforcement
- **Technical Details**:
  - Write `.pyi` stub files manually or use `maturin generate-imports`
  - Include all classes, methods, properties, and return types
  - Use `typing` module for `List`, `Optional`, `Union`
  - Example:
    ```python
    class Spider:
        @classmethod
        def open(cls, path: str) -> Spider: ...
        @classmethod
        def open_default(cls) -> Spider: ...
        def close(self) -> None: ...
        def path(self) -> str: ...
        def index(self, request: IngestRequest) -> IngestResult: ...
        def find_by_label(self, label: str) -> List[NodeId]: ...
        def find_by_property(self, key: str, value: str) -> List[NodeId]: ...
        def find_one_by_property(self, key: str, value: str) -> NodeId | None: ...
        def get_neighbors(self, node_id: NodeId, direction: Direction = ...) -> List[Neighbor]: ...
        def get_bio_score(self, node_id: NodeId) -> float: ...
        def get_bio_tier(self, node_id: NodeId) -> BioTier: ...
    ```
- **Acceptance Criteria**:
  - [ ] `python/spider/__init__.pyi` exists with complete type annotations
  - [ ] `mypy --strict python/spider/` passes without errors (against stub file)
  - [ ] IDE autocompletion works in VS Code / PyCharm for `import spider`
  - [ ] All method signatures match the actual Rust implementation
- **Dependencies**: Tasks 3–11 (all public API must be finalized first)
- **Estimated Complexity**: S
- **Implementation Notes**: PyO3's `#[pyclass]` and `#[pymethods]` don't auto-generate `.pyi` files. Manual stubs are the most reliable approach. Alternatively, use `stubgen` from `mypy` to generate initial stubs, then refine.

#### Task 13: Python integration tests
- **Description**: Write comprehensive Python-side integration tests that exercise the full API from a Python perspective.
- **Scope**:
  - Include: `tests/` directory with pytest-based tests
  - Out of scope: Rust unit tests (already in spider-core)
- **Technical Details**:
  - Use `pytest` as the test framework
  - Test file structure:
    ```
    tests/
    ├── test_lifecycle.py      # Open, close, context manager, default path
    ├── test_ingest.py         # Document ingestion, deduplication, edge cases
    ├── test_find.py           # find_by_label, find_by_property queries
    ├── test_traverse.py       # get_neighbors, get_relationships, count
    ├── test_bio.py            # Bio scoring, tier classification
    └── conftest.py            # Fixtures for temp database
    ```
  - `conftest.py` fixture: `tmp_db` that creates a temporary directory, opens Spider, yields it, and closes
  - Test error handling: verify correct exceptions are raised
  - Test thread safety: verify GIL release by running operations in `concurrent.futures.ThreadPoolExecutor`
- **Acceptance Criteria**:
  - [ ] `pytest tests/` passes with all tests green
  - [ ] Tests cover happy paths and error cases for every public method
  - [ ] At least one test verifies GIL release (operation in thread doesn't block main thread)
  - [ ] Test coverage > 90% of Python-exposed API surface
  - [ ] Tests are independent and can run in any order
- **Dependencies**: Tasks 3–11
- **Estimated Complexity**: M
- **Implementation Notes**: Use `tempfile.TemporaryDirectory()` for isolated test databases. The fixture pattern ensures cleanup even on test failure. Thread safety tests should use `ThreadPoolExecutor` with 2+ threads and verify concurrent operations don't deadlock.

#### Task 14: GIL release verification tests
- **Description**: Write dedicated tests that verify ALL I/O methods properly release the GIL. This is the #1 performance pitfall in PyO3 projects.
- **Scope**:
  - Include: Python tests that prove GIL release for every I/O method
  - Out of scope: Performance benchmarking (that's a separate concern)
- **Technical Details**:
  - Pattern: Start a long-running I/O operation in one thread, verify another thread can execute Python code concurrently:
    ```python
    import threading
    import time

    def test_gil_release_on_index():
        db = Spider.open(temp_dir)
        db.index(big_request)  # should release GIL

        gil_released = threading.Event()
        def check_gil():
            gil_released.set()  # if GIL were held, this wouldn't run

        t = threading.Thread(target=check_gil)
        t.start()
        t.join(timeout=1.0)
        assert gil_released.is_set(), "GIL was not released during index()"
    ```
  - Test every I/O method: `open`, `close`, `index`, `find_by_label`, `find_by_property`, `find_one_by_property`, `get_neighbors`, `get_relationships`, `count_relationships`, `get_bio_score`, `get_bio_tier`, `node_count`, `node_touch`, `set_significance`
  - Alternative approach: Use `time.time()` to measure that concurrent operations actually overlap (not sequential)
- **Acceptance Criteria**:
  - [ ] Every I/O method has a GIL release test
  - [ ] Tests fail if `py.allow_threads()` is removed from any method (proves the test works)
  - [ ] Concurrent execution is verified (operations overlap in time, not sequential)
  - [ ] Tests pass on Python 3.9–3.13
- **Dependencies**: Tasks 7–11 (all I/O methods must be implemented)
- **Estimated Complexity**: M
- **Implementation Notes**: This is a verification safety net. If a contributor forgets `py.allow_threads()` on a new method, this test suite should catch it. Consider making this part of CI.

#### Task 15: Memory leak detection tests
- **Description**: Write tests that detect memory leaks from Arc cycles, unclosed database handles, and Python object retention.
- **Scope**:
  - Include: Python tests for memory leak detection
  - Out of scope: Heap profiling, flame graphs
- **Technical Details**:
  - Test Arc cycle collection:
    ```python
    import gc
    import sys

    def test_spider_gc_cleanup():
        before = len(gc.get_objects())
        for _ in range(100):
            db = Spider.open(temp_dir)
            db.close()
        gc.collect()
        after = len(gc.get_objects())
        assert after - before < 10, f"Possible Arc leak: {after - before} objects retained"
    ```
  - Test context manager cleanup: Verify `with Spider.open(...) as db:` doesn't leak
  - Test unclosed Spider: Verify that forgetting to call `close()` doesn't leak (Drop should handle it)
  - Use `tracemalloc` for Python-side memory tracking
  - Test with large ingestion payloads to verify no memory growth
- **Acceptance Criteria**:
  - [ ] 100 open/close cycles don't grow object count by more than a small threshold
  - [ ] Context manager pattern doesn't leak
  - [ ] Forgetting to call `close()` doesn't leak (Drop handles it)
  - [ ] Large ingestion (1000+ propositions) doesn't grow memory disproportionately
  - [ ] Tests use `gc.collect()` and `tracemalloc` for detection
- **Dependencies**: Tasks 7, 13
- **Estimated Complexity**: M
- **Implementation Notes**: Rust's `Arc` is not cycle-collected by Python's GC. The `PySpider` class should implement `__del__` or rely on `close()` being called. If `close()` isn't called, the `Drop` impl should break the Arc cycle. Test this explicitly.

#### Task 16: Mutex poisoning resilience tests
- **Description**: Write tests that verify the `parking_lot::Mutex` handles edge cases correctly — particularly that panics inside spider-core don't poison the mutex and subsequent calls still work.
- **Scope**:
  - Include: Rust unit tests for mutex resilience
  - Out of scope: Simulating actual spider-core panics (those are spider-core's responsibility)
- **Technical Details**:
  - Test that `parking_lot::Mutex` doesn't poison (unlike `std::sync::Mutex`):
    ```rust
    #[test]
    fn test_mutex_not_poisoned() {
        let mutex = parking_lot::Mutex::new(42);
        // Simulate: if a previous call panicked, can we still lock?
        let guard = mutex.lock();
        assert_eq!(*guard, 42);
        // Drop guard, lock again — should work fine
        drop(guard);
        let guard2 = mutex.lock();
        assert_eq!(*guard2, 42);
    }
    ```
  - Test that `PySpider` can handle consecutive operations after an error (e.g., opening a corrupt DB, then closing it)
  - Test concurrent access doesn't deadlock: multiple threads calling methods on the same `Spider` instance
  - Verify that `close()` is idempotent and safe to call multiple times
- **Acceptance Criteria**:
  - [ ] `parking_lot::Mutex` behavior documented and tested (no poisoning)
  - [ ] `Spider` can be used after an error-returning operation (e.g., find on empty DB)
  - [ ] Concurrent access from multiple Python threads doesn't deadlock
  - [ ] Double-close is safe (idempotent)
  - [ ] Opening a corrupt database, then closing it, works without panicking
- **Dependencies**: Tasks 3, 7
- **Estimated Complexity**: S
- **Implementation Notes**: `parking_lot::Mutex` is non-poisoning by design, but if spider-core panics while holding the lock, the `Spider` state may be inconsistent. Document this in the Python API docs: "After a panic, the Spider instance may return inconsistent results."

#### Task 17: Documentation and examples
- **Description**: Write user-facing documentation including README, API reference, and example scripts.
- **Scope**:
  - Include: `README.md` in spider-py, `examples/` directory, API docstrings
  - Out of scope: Full tutorial website, video content
- **Technical Details**:
  - README structure:
    - Project description and relationship to spider-core
    - Installation instructions (`pip install spider-py` or `maturin develop`)
    - Quick start example (5-10 lines of Python)
    - API overview table
    - Development setup (building from source)
    - Link to parent project
  - Examples:
    - `examples/quick_start.py` — basic open, ingest, query
    - `examples/knowledge_graph.py` — building a knowledge graph from text
    - `examples/bio_scoring.py` — exploring bio-inspired memory decay
  - Add `#[doc]` strings to all `#[pyclass]` and `#[pymethods]` for `help(spider.Spider)` to work
- **Acceptance Criteria**:
  - [ ] `README.md` exists with installation, quick start, and API overview
  - [ ] `examples/quick_start.py` runs end-to-end without errors
  - [ ] `help(spider.Spider)` shows docstrings in Python REPL
  - [ ] `help(spider.Spider.index)` shows method documentation
  - [ ] Examples demonstrate all major API features
- **Dependencies**: Tasks 3–13
- **Estimated Complexity**: S
- **Implementation Notes**: PyO3 docstrings are set via the `name` and `text` parameters on `#[pymethods]` or the `///` doc comments on Rust functions. Test `help()` output by running `python -c "import spider; help(spider.Spider)"`.

#### Task 18: CI/CD and wheel building configuration
- **Description**: Configure automated building and publishing of Python wheels for all major platforms.
- **Scope**:
  - Include: CI workflow configuration, maturin build settings
  - Out of scope: Publishing to PyPI (manual first step)
- **Technical Details**:
  - GitHub Actions workflow (or equivalent CI):
    - Test matrix: Python 3.9, 3.10, 3.11, 3.12, 3.13 on Linux (x86_64), macOS (x86_64 + aarch64), Windows (x86_64)
    - Steps: checkout → setup Python → `pip install maturin pytest` → `maturin develop` → `pytest tests/`
    - Build step: `maturin build --release --features extension-module` for each platform
  - `pyproject.toml` classifiers for PyPI
  - `.gitignore` updates for build artifacts (`target/wheels/`, `*.so`, `*.pyd`)
  - **Important**: Test command uses `maturin develop` (no `extension-module` feature needed for local dev). Build command uses `--features extension-module`.
- **Acceptance Criteria**:
  - [ ] CI workflow exists and runs on push/PR
  - [ ] Tests pass on all platform/Python version combinations
  - [ ] `maturin build --release --features extension-module` produces a wheel file
  - [ ] Wheel is installable via `pip install dist/spider_py-*.whl`
  - [ ] `import spider` works after wheel installation
  - [ ] `cargo test` works without `--features extension-module`
- **Dependencies**: Tasks 1, 13–16
- **Estimated Complexity**: M
- **Implementation Notes**: maturin handles cross-compilation well. For macOS aarch64 (Apple Silicon), ensure the Rust toolchain includes `aarch64-apple-darwin` target. Use `cibuildwheel` for automated multi-platform wheel building if needed. Remember: `extension-module` feature must be explicitly enabled for wheel building but NOT for `cargo test`.

### Phase 5: Deferred Features (Post-MVP)

These features were explicitly deferred from MVP to keep the initial release focused. They can be added once Phases 1–4 are stable and users demand additional functionality.

#### Task 19: Content-addressable blob storage API
- **Description**: Expose spider-core's blob storage (`db/content.rs`) for storing raw document text and binary data alongside the graph.
- **Scope**:
  - Include: `store_blob()`, `get_blob()`, `has_blob()` methods in `src/spider.rs`
  - Out of scope: Streaming large blobs, blob deletion, blob listing
- **Technical Details**:
  ```python
  def store_blob(self, data: bytes) -> str  # returns SHA-256 hex digest
  def get_blob(self, hash_hex: str) -> bytes
  def has_blob(self, hash_hex: str) -> bool
  ```
  - Read the `db/content.rs` module in spider-core to understand the exact API
  - Python `bytes` → Rust `&[u8]` for storage
  - SHA-256 hex digest → Python `str`
  - All calls follow GIL release pattern from Task 7
- **Acceptance Criteria**:
  - [ ] `hash_hex = db.store_blob(b"hello world")` stores blob and returns hash
  - [ ] `db.get_blob(hash_hex) == b"hello world"`
  - [ ] `db.has_blob(hash_hex)` returns True
  - [ ] `db.get_blob("nonexistent")` raises `SpiderNotFoundError`
  - [ ] Stored blob persists across close/reopen
  - [ ] Large blobs (>1MB) are stored and retrieved correctly
- **Dependencies**: Tasks 1–4 (Spider handle, exceptions)
- **Estimated Complexity**: M
- **Implementation Notes**: Blob storage uses SHA-256 content addressing with deduplication. The API likely takes `&[u8]` and returns a hash string. Useful for storing raw document text alongside the graph.

#### Task 20: Property read/write API
- **Description**: Expose generic property setters/getters for nodes, beyond what ingestion provides.
- **Scope**:
  - Include: `set_property()`, `get_property()`, `delete_property()` methods in `src/spider.rs`
  - Out of scope: Bulk property operations, indexed property queries
- **Technical Details**:
  ```python
  def set_property(self, node_id: NodeId, key: str, value: str | int) -> None
  def get_property(self, node_id: NodeId, key: str) -> str | int | None
  def delete_property(self, node_id: NodeId, key: str) -> None
  ```
  - spider-core's `ingest.rs` has `set_string_property` and `set_int_property` as private functions
  - These need to be exposed or reimplemented in spider-py
  - Value type: accept both `str` and `int`, dispatch to correct setter internally
- **Acceptance Criteria**:
  - [ ] `db.set_property(node_id, "name", "Mumbai")` sets a string property
  - [ ] `db.set_property(node_id, "population", 20000000)` sets an int property
  - [ ] `db.get_property(node_id, "name")` returns the value
  - [ ] `db.get_property(node_id, "nonexistent")` returns None
  - [ ] Properties persist across close/reopen
  - [ ] Setting property on non-existent node raises `SpiderNotFoundError`
- **Dependencies**: Tasks 1–4
- **Estimated Complexity**: M
- **Implementation Notes**: May require exposing spider-core's private property functions or reimplementing them. The value type dispatch (`str` vs `int`) should be handled in Rust, not Python.

#### Task 21: Async Python API (optional feature flag)
- **Description**: Add async support for Python applications that use `asyncio`, via `tokio` and `pyo3-async-runtimes`.
- **Scope**:
  - Include: `async def index_async()`, `async def find_by_label_async()`, etc. behind a feature flag
  - Out of scope: Native async in spider-core (that's a spider-core concern)
- **Technical Details**:
  - Add feature flag: `async = ["pyo3-async-runtimes/tokio-runtime", "tokio"]`
  - Pattern for async methods:
    ```rust
    async fn index_async(&self, request: PyIngestRequest) -> PyResult<PyIngestResult> {
        let inner = Arc::clone(&self.inner);
        let rust_request = request.to_rust();

        tokio::task::spawn_blocking(move || {
            let mut db = inner.lock().unwrap();
            spider_core::db::ingest::index(&mut db, &rust_request)
        })
        .await
        .unwrap()
        .map(PyIngestResult::from)
        .map_err(Into::into)
    }
    ```
  - Use `spawn_blocking` because spider-core is sync — this runs blocking I/O on tokio's thread pool without blocking the async event loop
  - Sync methods remain available — async is opt-in via feature flag
- **Acceptance Criteria**:
  - [ ] `async with spider.Spider.open_async("./my_graph") as db:` works
  - [ ] `await db.index_async(request)` returns `IngestResult`
  - [ ] Async methods don't block the Python event loop (verified by concurrent asyncio.gather)
  - [ ] Sync methods still work when `async` feature is not enabled
  - [ ] `maturin build --features async,extension-module` produces a wheel with async support
- **Dependencies**: Tasks 1–7 (all sync methods must be stable first)
- **Estimated Complexity**: L
- **Implementation Notes**: This adds significant build complexity (`tokio` + `pyo3-async-runtimes` dependency chain). Only add when users actually demand async. The `spawn_blocking` approach is the correct pattern for wrapping sync I/O in async — don't try to make spider-core natively async.

## Implementation Order

```
Phase 1: Foundation
  Task 1 (scaffold) → Task 2 (exceptions) → Task 3 (Spider handle + context manager)

Phase 2: Core Types & Ingestion
  Task 4 (NodeId/EdgeId) + Task 5 (Direction/BioTier) [parallel]
  → Task 6 (ingestion types with to_rust conversion)
  → Task 7 (index ingestion — establishes GIL release pattern)

Phase 3: Queries & Traversal
  Task 8 (find queries) → Task 9 (traversal) → Task 10 (bio scoring) → Task 11 (node ops)

Phase 4: Polish & Packaging
  Task 12 (type stubs) + Task 13 (integration tests) [parallel]
  → Task 14 (GIL release verification)
  → Task 15 (memory leak detection)
  → Task 16 (mutex poisoning resilience)
  → Task 17 (documentation) + Task 18 (CI/CD) [parallel]

Phase 5: Deferred Features (Post-MVP)
  Task 19 (blob storage) + Task 20 (property API) [parallel]
  → Task 21 (async support) [optional, only if users demand it]
```

**Critical path**: Tasks 1→2→3→4→5→6→7→8→9→13→14. This path covers the core database lifecycle, ingestion, querying, testing, and GIL verification — the minimum viable product.

**Parallelizable work**:
- Tasks 4 and 5 can be developed in parallel (independent type definitions)
- Tasks 10 and 11 can be developed in parallel (independent operations)
- Tasks 12 and 13 can be developed in parallel (stubs don't block tests)
- Tasks 17 and 18 can be developed in parallel once the API is stable
- Tasks 19 and 20 can be developed in parallel (independent features)

## Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| **Forgetting `py.allow_threads()` before I/O** | High | Task 14 (GIL release verification tests) catches this. Document the pattern in BRAINSTORMING.md. Code review checklist includes "GIL released?" |
| **Accessing Python objects after GIL release (segfaults)** | High | Convert all Python types to Rust BEFORE `py.allow_threads()`. Code review. Task 14 tests will hang/crash if violated. |
| **Borrowed data lifetime issues in ingestion conversion** | Medium | Build Rust `IngestRequest` at call time with `&str` slices from Python-owned `String` fields. Lifetime is valid for the duration of the method call. Document explicitly in code comments. |
| **`extension-module` feature breaks `cargo test`** | Medium | Feature is NOT in default. Document separate build commands: `cargo test` for tests, `maturin build --features extension-module` for wheels. CI enforces both. |
| **Mutex contention under concurrent Python access** | Medium | `parking_lot::Mutex` is held only during individual operations, not across method calls. Each method locks, does work, and releases. Document the limitation and recommend multiple `Spider` instances for high-concurrency scenarios. |
| **Platform-specific wheel building (macOS aarch64)** | Medium | Use `cibuildwheel` which handles cross-compilation automatically. Test on actual hardware if possible. |
| **API changes in spider-core** | Medium | spider-py is tightly coupled to spider-core's public API. Any changes to `Spider::open()`, `index()`, etc. will require corresponding updates. Version lock spider-py to spider-core versions. |
| **Memory leaks from Arc cycles in Python** | Medium | Task 15 (memory leak detection tests) catches this. Implement `__del__` or ensure `close()` breaks the Arc cycle. Test with `gc.collect()` and `tracemalloc`. |
| **Python 3.9–3.13 GIL behavior differences** | Low | Test on all versions in CI (Task 18). Document known differences. |
| **Mutex poisoning resilience** | Low | `parking_lot::Mutex` doesn't poison (Task 16 verifies). But if spider-core panics internally, state may be inconsistent. Document this in API docs. |
| **Scope creep from deferred features** | Medium | Phase 5 features (blob storage, property API, async) are explicitly out of MVP. Resist adding them until Phases 1–4 are shipped and tested. |
