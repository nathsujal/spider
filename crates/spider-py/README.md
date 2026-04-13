# spider-py

Python bindings for [Spider](https://github.com/nathsujal/spider) — a bio-inspired AI agent memory graph database engine written in pure Rust.

`spider-py` exposes `spider-core`'s synchronous, blocking database API to Python, enabling Python applications and AI agent frameworks to use Spider's graph storage, document ingestion, bio-inspired scoring, and graph traversal directly from Python.

## Features

- **Graph Database**: Store and query nodes, edges, and properties with O(1) direct access
- **Document Ingestion**: Ingest documents with propositions and named entities, automatic deduplication
- **Graph Traversal**: Walk edge chains to find neighbors and relationships in any direction
- **Bio-Inspired Scoring**: Nodes have vitality scores based on significance, access frequency, and recency
- **Thread-Safe**: All I/O operations release the GIL, enabling true Python multithreading
- **Context Manager**: Idiomatic Python usage with automatic resource cleanup

## Installation

### From Source (Development)

```bash
# Clone the parent project
git clone https://github.com/nathsujal/spider.git
cd spider/crates/spider-py

# Install maturin
pip install maturin

# Build and install in development mode
maturin develop
```

### From Wheel (Production)

```bash
pip install spider-py
```

## Quick Start

```python
import spider

# Open a database (creates it if it doesn't exist)
with spider.Spider.open("./my_graph") as db:
    # Ingest a document with propositions and entities
    request = spider.IngestRequest(
        title="My Knowledge",
        propositions=[
            spider.Proposition(
                "Mumbai is the financial capital of India",
                entities=[
                    spider.Entity("Mumbai", "LOCATION"),
                    spider.Entity("India", "LOCATION"),
                ],
            ),
        ],
    )
    result = db.index(request)
    print(f"Document ID: {result.document_id}")

    # Query by label
    docs = db.find_by_label("DOCUMENT")
    print(f"Found {len(docs)} documents")

    # Traverse the graph
    for doc_id in docs:
        neighbors = db.get_neighbors(doc_id, spider.Direction.OUTGOING)
        for n in neighbors:
            print(f"  -> {n.node_id} via edge {n.edge_id}")

    # Bio scoring
    for doc_id in docs:
        score = db.get_bio_score(doc_id)
        tier = db.get_bio_tier(doc_id)
        print(f"  Score: {score:.2f}, Tier: {tier}")
```

## API Overview

### Database Lifecycle

| Method | Description |
|--------|-------------|
| `Spider.open(path)` | Open/create a database at the given path |
| `Spider.open_default()` | Open/create at platform-default location |
| `db.close()` | Flush and close the database (idempotent) |
| `db.path` | Get the database directory path |

### Ingestion

| Method | Description |
|--------|-------------|
| `db.index(request)` | Ingest an IngestRequest, returns IngestResult |

### Queries

| Method | Description |
|--------|-------------|
| `db.find_by_label(label)` | Find all nodes with a given label |
| `db.find_by_property(key, value)` | Find nodes with a matching property |
| `db.find_one_by_property(key, value)` | Find first matching node or None |

### Graph Traversal

| Method | Description |
|--------|-------------|
| `db.get_neighbors(node_id, direction)` | Get neighbor nodes in a direction |
| `db.get_relationships(node_id, direction)` | Get relationship dicts with source/target |
| `db.count_relationships(node_id, direction)` | Count relationships efficiently |

### Bio Scoring

| Method | Description |
|--------|-------------|
| `db.get_bio_score(node_id)` | Calculate node vitality score |
| `db.get_bio_tier(node_id)` | Get storage tier (Hot/Warm/Cold/Pruned) |

### Node Operations

| Method | Description |
|--------|-------------|
| `db.node_count()` | Get total node count |
| `db.node_touch(node_id)` | Touch node (increment access count) |
| `db.set_significance(node_id, sig)` | Set significance (0-255) |

### Types

| Type | Description |
|------|-------------|
| `NodeId` | Unique node identifier (1-based integer) |
| `EdgeId` | Unique edge identifier (1-based integer) |
| `Neighbor` | Node-Edge pair from traversal |
| `Direction` | OUTGOING, INCOMING, BOTH |
| `BioTier` | HOT, WARM, COLD, PRUNED |
| `Entity` | Named entity with name and type |
| `Proposition` | Factual statement with entities |
| `IngestRequest` | Document with propositions for ingestion |
| `IngestResult` | Counts of created nodes/edges |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `SpiderError` | Base exception for all Spider errors |
| `SpiderNotFoundError` | Node, edge, or blob not found |
| `SpiderCorruptError` | Database file corruption |
| `SpiderIOError` | File I/O errors |
| `SpiderIngestionError` | Ingestion failures |
| `SpiderTraversalError` | Traversal depth exceeded |

## Development

### Building from Source

```bash
# Build for development (no extension-module feature)
cargo check -p spider-py
cargo clippy -p spider-py

# Build wheel for distribution
maturin build --release --features extension-module
```

### Running Tests

Tests require the built extension module. First build with maturin, then run pytest:

```bash
# Build the extension
maturin develop

# Run Python integration tests
pytest crates/spider-py/tests/ -v
```

**Note**: `cargo test` cannot run for this crate because it's a `cdylib` Python extension module. Use `maturin develop` + `pytest` for testing instead.

### Project Structure

```
spider-py/
├── src/
│   ├── lib.rs              # PyO3 module and exception registration
│   ├── error.rs            # DbError → PyErr mapping
│   ├── spider_handle.rs    # PySpider class (all Spider methods)
│   ├── types.rs            # PyNodeId, PyEdgeId, PyDirection, PyBioTier, PyNeighbor
│   └── ingest.rs           # PyEntity, PyProposition, PyIngestRequest, PyIngestResult
├── python/spider/
│   └── __init__.pyi        # Type stubs for IDE autocompletion
├── tests/                  # Python integration tests
├── examples/               # Example scripts
├── Cargo.toml
└── pyproject.toml
```

## Architecture

`spider-py` uses PyO3 to wrap `spider-core`'s database engine. The `Spider` handle wraps the inner database in `Arc<parking_lot::Mutex<...>>` for thread-safe access. All I/O operations release Python's GIL via `py.allow_threads()`, enabling true multithreading.

Key design decisions:
- **Direct PyO3 Wrapping**: One Rust struct → one Python class
- **Flat Module Structure**: All methods in one file for ~15 public functions
- **Synchronous-only**: spider-core is sync-only, `allow_threads()` achieves non-blocking Python
- **Context Manager Support**: `__enter__`/`__exit__` for idiomatic Python usage

See [BRAINSTORMING.md](BRAINSTORMING.md) for the full architecture exploration.

## License

MIT
