# Spider

**Bio-inspired AI agent memory graph database** — written in pure Rust.

Spider stores knowledge as a graph of **Documents → Propositions → Entities**, with vitality scoring that mimics how biological memories strengthen through use and fade with time.

## Quick Start

```bash
# Build the workspace
cargo build --release

# Inspect a database with the TUI
cargo run -p spider-inspect -- ./my_graph

# Run the HTTP daemon
cargo run -p spider-daemon -- --db-path ./my_graph --addr 0.0.0.0:3000

# Run tests
cargo test
```

## Workspace Structure

| Crate | Description |
|-------|-------------|
| [`spider-core`](crates/spider-core/) | Storage engine — sync, blocking, direct file access. The heart of the system. |
| [`spider-daemon`](crates/spider-daemon/) | HTTP/WebSocket API server (Axum + Tokio) with async job queue for LLM-powered ingestion. |
| [`spider-inspect`](crates/spider-inspect/) | Interactive TUI for database inspection — graph visualization, integrity checks, REPL commands. |
| [`spider-py`](crates/spider-py/) | PyO3 Python bindings — `import spider` for Python applications (in development). |

## How It Works

A document enters Spider as pre-extracted **propositions** (factual statements) with **entities** (named concepts). Spider wires them into a graph:

```
[Document] ──CONTAINS──► [Proposition] ──MENTIONS──► [Entity]
           ──CONTAINS──► [Proposition] ──MENTIONS──► [Entity]
```

Every node receives a **bio score** based on significance, access frequency, and time decay — mimicking how human memories strengthen or fade.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Complete system architecture and document lifecycle walkthrough
- **[crates/spider-core/README.md](crates/spider-core/README.md)** — Storage engine API and module overview
- **[crates/spider-inspect/README.md](crates/spider-inspect/README.md)** — TUI usage and commands
- **[crates/spider-py/BRAINSTORMING.md](crates/spider-py/BRAINSTORMING.md)** — Python bindings design (in development)
- **[crates/spider-py/TASKS.md](crates/spider-py/TASKS.md)** — Implementation task list

## Key Characteristics

- **Synchronous core** — no async, no network, no LLM calls in spider-core
- **Fixed-size records** — O(1) random access (Node: 29 bytes, Edge: 33 bytes)
- **Embedded doubly-linked edge chains** — no separate adjacency index needed
- **Content-addressable blobs** — SHA-256 deduplication
- **Bio-inspired vitality scoring** — Hot / Warm / Cold / Pruned tiers
- **Tombstone deletion** — nodes/edges are soft-deleted, enabling graceful corruption handling

## License

MIT
