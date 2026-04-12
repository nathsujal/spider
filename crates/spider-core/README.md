# spider-core

Pure Rust database engine for Spider — a bio-inspired AI agent memory graph.

## What It Does

Spider-core is the on-disk storage engine. It owns the `.db` files in a database directory and provides the low-level API for all other Spider components (daemon, Python bindings, inspect REPL).

Key characteristics:
- **Synchronous, blocking** — no async, no HTTP, no network
- **Fixed-size records** — O(1) random access (Node: 29 bytes, Edge: 33 bytes, Property: 40 bytes)
- **Content-addressable blobs** — SHA-256 deduplication under `blobs/`
- **Bio-inspired vitality scoring** — every node gets a score based on significance, access frequency, and time decay
- **Zero external dependencies for core logic** — LLM calls and HTTP live in other crates

## Quick Start

```rust
use std::path::Path;
use spider_core::db::lifecycle::Spider;

// Open or create a database
let mut db = Spider::open(Path::new("./my_graph"))?;

// Ingest a document with pre-extracted propositions
use spider_core::db::ingest::{index, IngestRequest, Proposition, Entity};

let result = index(&mut db, &IngestRequest {
    title: "My Document",
    propositions: vec![
        Proposition {
            text: "Mumbai is the financial capital of India",
            entities: vec![
                Entity { name: "Mumbai", entity_type: "LOCATION" },
                Entity { name: "India", entity_type: "LOCATION" },
            ],
        },
    ],
})?;

println!("Document node ID: {}", result.document_id.get());
println!("Propositions created: {}", result.proposition_count);
println!("New entities: {}", result.entity_count);
```

## Database Layout

A Spider database is a directory containing multiple files:

```
my_graph/
├── meta.db            44-byte header (ID counters, bio tuning params)
├── nodes.db           Node records (29 bytes each)
├── edges.db           Edge records (33 bytes each)
├── properties.db      Property records (40 bytes each)
├── strings.db         Dynamic string records (128 bytes each)
├── arrays.db          Dynamic array records (128 bytes each)
├── labels.tokens      Interned label names
├── edge_types.tokens  Interned edge type names
├── prop_keys.tokens   Interned property key names
└── blobs/             Content-addressable blob storage (SHA-256 named files)
```

## Graph Shape After Ingestion

```
[File node]
     │
  SOURCED_FROM
     │
     ▼
[Document] ──CONTAINS──► [Proposition] ──MENTIONS──► [Entity]
           ──CONTAINS──► [Proposition]
           ──CONTAINS──► [Proposition] ──MENTIONS──► [Entity]
```

## Bio Score Formula

```
Score = ((S × Ws × 100) + (F × Wf)) / (Δdays + 2)^G

Where:
  S          = significance (0.0–1.0, stored as u8 0–255)
  F          = log-dampened frequency: ln(1 + access_count) × 10
  Δdays      = days since last access
  Ws, Wf, G  = tuning parameters stored in meta.db
```

## Module Overview

| Module | Purpose |
|---|---|
| `db::lifecycle` | [`Spider`](https://docs.rs/spider-core/latest/spider_core/db/lifecycle/struct.Spider.html) — open, close, Drop |
| `db::ingest` | [`index()`](https://docs.rs/spider-core/latest/spider_core/db/ingest/fn.index.html) — Document → Proposition → Entity wiring |
| `db::content` | Blob storage — `file()`, `read_blob()`, `remove_blob()` |
| `db::find` | `find_by_label()`, `find_by_property()` |
| `query::traverse` | `get_neighbors()`, `get_relationships()`, `count_relationships()` |
| `property` | `get()`, `set()`, `delete()`, `list_all()` with `PropertyValue` enum |
| `bio::score` | `calculate_bio_score()` — the vitality formula |
| `bio::tier` | `BioTier` — Hot / Warm / Cold / Pruned |
| `schema` | On-disk record layouts |
| `store` | `RecordFile<T>` — fixed-size record I/O |
| `error` | `DbError` — single error enum for all operations |

## Running Tests

```bash
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

## License

MIT
