//! # spider-core
//!
//! Pure Rust database engine for **Spider** — a bio-inspired AI agent memory graph.
//!
//! Spider mimics how biological memory works: memories form when information is added,
//! strong memories persist, weak memories decay, and related memories strengthen each
//! other. The `spider-core` crate owns the on-disk `.db` files and provides the
//! low-level API for all other Spider components.
//!
//! ## Architecture
//!
//! ```text
//! spider-core (this crate)
//!       ↑                          ↑
//! spider-py (cdylib)         spider-daemon (bin)
//!       ↑                          ↑
//! python/spider/             spider-inspect (bin)
//! ```
//!
//! `spider-core` has **no** HTTP, no PyO3, no async runtime. It is a synchronous,
//! blocking database engine that every other Spider component depends on.
//!
//! ## Quick Start
//!
//! ```no_run
//! use std::path::Path;
//! use spider_core::db::lifecycle::Spider;
//!
//! let mut db = Spider::open(Path::new("./my_graph")).unwrap();
//! // ... use db ...
//! db.close().unwrap();
//! ```
//!
//! ## Key Modules
//!
//! | Module | Purpose |
//! |---|---|
//! | [`db::lifecycle`] | Database open/close/Drop, [`Spider`](db::lifecycle::Spider) handle |
//! | [`db::ingest`] | [`index()`](db::ingest::index) — wire Document → Proposition → Entity graph |
//! | [`db::content`] | Content-addressable blob storage with SHA-256 deduplication |
//! | [`db::find`] | Query nodes by label or property value |
//! | [`query::traverse`] | Graph traversal: neighbors, relationships, edge counting |
//! | [`property`] | Typed property get/set/list with [`PropertyValue`](property::PropertyValue) enum |
//! | [`bio::score`] | Vitality scoring — significance + frequency + time decay |
//! | [`bio::tier`] | Hot / Warm / Cold / Pruned classification |
//! | [`schema`] | On-disk record layouts (Node: 29 bytes, Edge: 33 bytes, Property: 40 bytes) |
//! | [`store`] | [`RecordFile<T>`](store::record::RecordFile) — fixed-size record I/O |
//! | [`error`] | Single [`DbError`](error::DbError) enum for all operations |
//!
//! ## Design Principles
//!
//! - **No external dependencies** — LLM calls, HTTP, and Python bindings live outside this crate
//! - **Deterministic** — no randomness, no network, fully testable without APIs
//! - **Fixed-size records** — O(1) random access: `offset = HEADER + (id - 1) * record_size`
//! - **Memory-mapped** — the OS handles page caching via `memmap2`
//! - **Content-addressable blobs** — SHA-256 deduplication, stored under `blobs/` directory

pub mod store;
pub mod schema;
pub mod error;
pub mod bio;
pub mod db;
pub mod query;
pub mod property;