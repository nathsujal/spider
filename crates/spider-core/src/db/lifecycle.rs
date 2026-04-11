//! Database lifecycle: open, close, Drop.
//!
//! Manages the lifecycle of a Spider database: initialization (`open()`),
//! graceful shutdown (`close()`), and automatic cleanup via `Drop`. All
//! database file handling happens here.
//!
//! ## Database directory layout
//!
//! ```text
//! my_graph/
//! ├── meta.db            44-byte metadata header (ID counters + bio params)
//! ├── nodes.db           RecordFile<Node>          (29 bytes/record)
//! ├── edges.db           RecordFile<Edge>          (33 bytes/record)
//! ├── properties.db      RecordFile<PropertyRecord> (40 bytes/record)
//! ├── strings.db         RecordFile<DynamicStringRecord> (128 bytes/record)
//! ├── arrays.db          RecordFile<DynamicArrayRecord>  (128 bytes/record)
//! ├── labels.tokens      TokenStore (variable-length wire format)
//! ├── edge_types.tokens  TokenStore (variable-length wire format)
//! └── prop_keys.tokens   TokenStore (variable-length wire format)
//! ```
//!
//! ## Responsibilities
//! - Open or create `.db` files
//! - Persist and load metadata
//! - Persist and load token stores
//! - Flush data on shutdown
//! - Clean up resources when the `Spider` handle is dropped

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::SpiderResult;
use crate::schema::edge::Edge;
use crate::schema::node::Node;
use crate::schema::property::PropertyRecord;
use crate::schema::dynamic::{DynamicStringRecord, DynamicArrayRecord};
use crate::schema::token::TokenStore;
use crate::store::record::RecordFile;

/// Returns the platform-default database directory path.
///
/// | Platform | Path |
/// |---|---|
/// | Linux   | `~/.local/share/spider/default/` |
/// | macOS   | `~/Library/Application Support/spider/default/` |
/// | Windows | `%APPDATA%\spider\default\` |
///
/// The directory is **not** created by this function — callers should
/// use [`Spider::open()`](Spider::open) which creates it automatically.
pub fn default_db_path() -> PathBuf {
    let proj = directories::ProjectDirs::from("dev", "spider", "spider")
        .expect("could not determine home directory");
    proj.data_dir().join("default")
}

// Metadata

/// 44-byte metadata header stored in `meta.db`.
///
/// Contains allocation counters for ID generation (nodes, edges, properties,
/// strings, arrays) and bio score tuning parameters.
///
/// ## Wire format (44 bytes, little-endian)
///
/// ```text
/// Offset  Size  Field
/// ──────  ────  ──────────────
///  0       4    next_node_id
///  4       4    next_rel_id
///  8       4    next_prop_id
/// 12       4    next_string_id
/// 16       4    next_array_id
/// 20       8    bio_w_sig    (f64)
/// 28       8    bio_w_freq   (f64)
/// 36       8    bio_gravity  (f64)
/// ──────  ────
/// Total   44
/// ```
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Metadata {
    /// Monotonically increasing counter for next node ID.
    pub next_node_id: u32,
    /// Monotonically increasing counter for next relationship ID.
    pub next_rel_id: u32,
    /// Monotonically increasing counter for next property ID.
    pub next_prop_id: u32,
    /// Monotonically increasing counter for next dynamic string ID.
    pub next_string_id: u32,
    /// Monotonically increasing counter for next dynamic array ID.
    pub next_array_id: u32,
    /// Significance weight in bio score formula (default 1.0).
    pub bio_w_sig: f64,
    /// Frequency weight in bio score formula (default 1.0).
    pub bio_w_freq: f64,
    /// Decay exponent in bio score formula (default 1.0).
    pub bio_gravity: f64,
}

impl Metadata {
    /// Serialized size on disk in bytes.
    pub const SIZE: usize = 44;

    /// Serialize to 44 bytes, little-endian.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.next_node_id.to_le_bytes());
        buf[4..8].copy_from_slice(&self.next_rel_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.next_prop_id.to_le_bytes());
        buf[12..16].copy_from_slice(&self.next_string_id.to_le_bytes());
        buf[16..20].copy_from_slice(&self.next_array_id.to_le_bytes());
        buf[20..28].copy_from_slice(&self.bio_w_sig.to_le_bytes());
        buf[28..36].copy_from_slice(&self.bio_w_freq.to_le_bytes());
        buf[36..44].copy_from_slice(&self.bio_gravity.to_le_bytes());
        buf
    }

    /// Deserialize from 44 bytes, little-endian.
    ///
    /// # Panics
    /// Panics if `bytes.len() != 44`. This indicates on-disk corruption —
    /// the caller should validate size before calling.
    pub fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        Self {
            next_node_id:   u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            next_rel_id:    u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            next_prop_id:   u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            next_string_id: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            next_array_id:  u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            bio_w_sig:      f64::from_le_bytes(bytes[20..28].try_into().unwrap()),
            bio_w_freq:     f64::from_le_bytes(bytes[28..36].try_into().unwrap()),
            bio_gravity:    f64::from_le_bytes(bytes[36..44].try_into().unwrap()),
        }
    }
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            next_node_id: 1,
            next_rel_id: 1,
            next_prop_id: 1,
            next_string_id: 1,
            next_array_id: 1,
            bio_w_sig: 1.0,
            bio_w_freq: 1.0,
            bio_gravity: 1.0,
        }
    }
}

// Spider

/// Main database handle.
///
/// Owns all record file handles, token stores, and metadata. Dropping this
/// struct closes the database automatically via [`Drop`].
///
/// ## Usage
///
/// ```no_run
/// # use std::path::Path;
/// # use spider_core::db::lifecycle::Spider;
/// let mut db = Spider::open(Path::new("./my_graph")).unwrap();
/// // ... use db ...
/// db.close().unwrap(); // explicit flush; also happens on Drop
/// ```
pub struct Spider {
    /// Database directory path.
    path: PathBuf,

    /// Global metadata — ID counters and bio params.
    pub metadata: Metadata,

    /// Node record file (`nodes.db`).
    pub nodes: RecordFile<Node>,
    /// Edge record file (`edges.db`).
    pub edges: RecordFile<Edge>,
    /// Property record file (`properties.db`).
    pub properties: RecordFile<PropertyRecord>,
    /// Dynamic string record file (`strings.db`).
    pub strings: RecordFile<DynamicStringRecord>,
    /// Dynamic array record file (`arrays.db`).
    pub arrays: RecordFile<DynamicArrayRecord>,

    /// Token store for node labels.
    pub label_tokens: TokenStore,
    /// Token store for edge types.
    pub edge_type_tokens: TokenStore,
    /// Token store for property keys.
    pub prop_key_tokens: TokenStore,

    /// `true` after `close()` has been called — prevents double-flush in `Drop`.
    closed: bool,
}

impl std::fmt::Debug for Spider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Spider")
            .field("path", &self.path)
            .field("metadata", &self.metadata)
            .field("label_tokens", &self.label_tokens.len())
            .field("edge_type_tokens", &self.edge_type_tokens.len())
            .field("prop_key_tokens", &self.prop_key_tokens.len())
            .field("closed", &self.closed)
            .finish()
    }
}

impl Spider {
    // File names within the database directory.
    const META_FILE: &'static str = "meta.db";
    const NODES_FILE: &'static str = "nodes.db";
    const EDGES_FILE: &'static str = "edges.db";
    const PROPERTIES_FILE: &'static str = "properties.db";
    const STRINGS_FILE: &'static str = "strings.db";
    const ARRAYS_FILE: &'static str = "arrays.db";
    const LABELS_TOKEN_FILE: &'static str = "labels.tokens";
    const EDGE_TYPES_TOKEN_FILE: &'static str = "edge_types.tokens";
    const PROP_KEYS_TOKEN_FILE: &'static str = "prop_keys.tokens";

    /// Creates a short-lived [`EdgeOps`] handle for edge CRUD operations.
    ///
    /// Borrows `nodes`, `edges`, and `metadata` mutably — no other
    /// operations on those fields are possible while the handle is alive.
    pub fn edge_ops(&mut self) -> crate::db::rels::EdgeOps<'_> {
        crate::db::rels::EdgeOps {
            nodes: &mut self.nodes,
            edges: &mut self.edges,
            metadata: &mut self.metadata,
        }
    }

    /// Opens or creates a database at the given directory path.
    ///
    /// - Creates the directory (and parents) if it doesn't exist.
    /// - Opens all record files, creating them if missing.
    /// - Loads metadata and token stores from existing files, or initializes
    ///   defaults for a fresh database.
    ///
    /// # Errors
    /// Returns `Err` if any file cannot be opened, created, or read.
    pub fn open(path: &Path) -> SpiderResult<Self> {
        fs::create_dir_all(path)?;

        let metadata = Self::load_metadata(path)?;
        let nodes = Self::open_or_create(&path.join(Self::NODES_FILE))?;
        let edges = Self::open_or_create(&path.join(Self::EDGES_FILE))?;
        let properties = Self::open_or_create(&path.join(Self::PROPERTIES_FILE))?;
        let strings = Self::open_or_create(&path.join(Self::STRINGS_FILE))?;
        let arrays = Self::open_or_create(&path.join(Self::ARRAYS_FILE))?;
        let label_tokens = Self::load_token_store(&path.join(Self::LABELS_TOKEN_FILE))?;
        let edge_type_tokens = Self::load_token_store(&path.join(Self::EDGE_TYPES_TOKEN_FILE))?;
        let prop_key_tokens = Self::load_token_store(&path.join(Self::PROP_KEYS_TOKEN_FILE))?;

        Ok(Self {
            path: path.to_path_buf(),
            metadata,
            nodes,
            edges,
            properties,
            strings,
            arrays,
            label_tokens,
            edge_type_tokens,
            prop_key_tokens,
            closed: false,
        })
    }

    /// Opens or creates a database at the platform-default location.
    ///
    /// | Platform | Default Path |
    /// |---|---|
    /// | Linux   | `~/.local/share/spider/default/` |
    /// | macOS   | `~/Library/Application Support/spider/default/` |
    /// | Windows | `%APPDATA%\spider\default\` |
    ///
    /// The directory is created automatically if it doesn't exist.
    /// This is equivalent to `Spider::open(&default_db_path())`.
    ///
    /// # Errors
    /// Returns `Err` if the default path cannot be resolved or any
    /// file cannot be opened, created, or read.
    pub fn open_default() -> SpiderResult<Self> {
        Self::open(&default_db_path())
    }

    /// The database directory path.
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Gracefully closes the database and flushes all data to disk.
    ///
    /// Writes metadata and all token stores back to their files, then syncs
    /// all record files. Safe to call multiple times — subsequent calls are
    /// no-ops.
    ///
    /// # Errors
    /// Returns `Err` if any flush or write fails. Even on error, the
    /// instance is marked closed to prevent partial double-writes in `Drop`.
    pub fn close(&mut self) -> SpiderResult<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;

        fs::write(
            self.path.join(Self::META_FILE),
            self.metadata.to_bytes(),
        )?;

        let token_files = [
            (Self::LABELS_TOKEN_FILE, &self.label_tokens),
            (Self::EDGE_TYPES_TOKEN_FILE, &self.edge_type_tokens),
            (Self::PROP_KEYS_TOKEN_FILE, &self.prop_key_tokens),
        ];
        for (name, store) in token_files {
            Self::save_token_store(&self.path.join(name), store)?;
        }

        Ok(())
    }

    // Private helpers

    /// Reads or initializes the metadata header.
    fn load_metadata(path: &Path) -> SpiderResult<Metadata> {
        let meta_path = path.join(Self::META_FILE);
        if meta_path.exists() {
            let data = fs::read(&meta_path)?;
            if data.len() != Metadata::SIZE {
                return Err(crate::error::DbError::CorruptMetadata {
                    expected_bytes: Metadata::SIZE,
                    got_bytes: data.len(),
                });
            }
            Ok(Metadata::from_bytes(data.try_into().unwrap()))
        } else {
            let meta = Metadata::default();
            fs::write(&meta_path, meta.to_bytes())?;
            Ok(meta)
        }
    }

    /// Opens a `RecordFile<T>` — creates if missing, opens if existing.
    fn open_or_create<T: crate::store::record::Record>(
        path: &Path,
    ) -> SpiderResult<RecordFile<T>> {
        if path.exists() {
            RecordFile::open(path)
        } else {
            RecordFile::create(path)
        }
    }

    /// Loads a `TokenStore` from disk — returns empty store if file missing.
    fn load_token_store(path: &Path) -> SpiderResult<TokenStore> {
        if path.exists() {
            let data = fs::read(path)?;
            Ok(TokenStore::from_bytes(&data))
        } else {
            Ok(TokenStore::new())
        }
    }

    /// Writes a `TokenStore` to disk.
    fn save_token_store(path: &Path, store: &TokenStore) -> SpiderResult<()> {
        let mut file = fs::File::create(path)?;
        file.write_all(&store.to_bytes())?;
        file.sync_all()?;
        Ok(())
    }
}

impl Drop for Spider {
    fn drop(&mut self) {
        // Ensure cleanup even if user doesn't call close() explicitly.
        if let Err(e) = self.close() {
            eprintln!("warning: error closing spider database during Drop: {}", e);
        }
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_db_path_contains_spider() {
        let path = default_db_path();
        let components: Vec<_> = path.components().collect();
        // Last component should be "default"
        assert_eq!(components.last().unwrap().as_os_str(), "default");
        // Parent should contain "spider" somewhere
        let parent = path.parent().unwrap();
        assert!(parent.to_string_lossy().contains("spider"));
    }

    #[test]
    fn open_default_creates_database() {
        let db = Spider::open_default().expect("open_default should succeed");
        assert!(db.path().ends_with("default"));
        assert!(db.path().join(Spider::META_FILE).exists());
        assert!(db.path().join(Spider::NODES_FILE).exists());
    }

    #[test]
    fn metadata_default_sane() {
        let m = Metadata::default();
        assert_eq!(m.next_node_id, 1);
        assert_eq!(m.next_rel_id, 1);
        assert_eq!(m.next_prop_id, 1);
        assert_eq!(m.next_string_id, 1);
        assert_eq!(m.next_array_id, 1);
        assert_eq!(m.bio_w_sig, 1.0);
        assert_eq!(m.bio_w_freq, 1.0);
        assert_eq!(m.bio_gravity, 1.0);
    }

    #[test]
    fn metadata_round_trip() {
        let m = Metadata {
            next_node_id: 42, next_rel_id: 100, next_prop_id: 7,
            next_string_id: 3, next_array_id: 5,
            bio_w_sig: 2.5, bio_w_freq: 1.3, bio_gravity: 0.8,
        };
        let restored = Metadata::from_bytes(m.to_bytes());
        assert_eq!(restored.next_node_id, 42);
        assert_eq!(restored.next_rel_id, 100);
        assert_eq!(restored.next_prop_id, 7);
        assert_eq!(restored.next_string_id, 3);
        assert_eq!(restored.next_array_id, 5);
        assert!((restored.bio_w_sig - 2.5).abs() < f64::EPSILON);
        assert!((restored.bio_w_freq - 1.3).abs() < f64::EPSILON);
        assert!((restored.bio_gravity - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn metadata_size_constant() {
        assert_eq!(Metadata::default().to_bytes().len(), Metadata::SIZE);
    }

    #[test]
    fn open_creates_fresh_database() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        let db = Spider::open(&db_path).unwrap();

        assert!(db_path.exists());
        assert!(db_path.join(Spider::META_FILE).exists());
        assert!(db_path.join(Spider::NODES_FILE).exists());
        assert!(db_path.join(Spider::EDGES_FILE).exists());
        assert!(db_path.join(Spider::PROPERTIES_FILE).exists());
        assert!(db_path.join(Spider::STRINGS_FILE).exists());
        assert!(db_path.join(Spider::ARRAYS_FILE).exists());
        assert_eq!(db.metadata.next_node_id, 1);
        assert!(db.label_tokens.is_empty());
        assert!(db.edge_type_tokens.is_empty());
        assert!(db.prop_key_tokens.is_empty());
    }

    #[test]
    fn close_persists_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        {
            let mut db = Spider::open(&db_path).unwrap();
            db.metadata.next_node_id = 99;
            db.metadata.bio_w_sig = 3.14;
            db.close().unwrap();
        }
        let db = Spider::open(&db_path).unwrap();
        assert_eq!(db.metadata.next_node_id, 99);
        assert!((db.metadata.bio_w_sig - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn close_persists_token_stores() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        {
            let mut db = Spider::open(&db_path).unwrap();
            db.label_tokens.get_or_create("Person").unwrap();
            db.label_tokens.get_or_create("Document").unwrap();
            db.edge_type_tokens.get_or_create("CONTAINS").unwrap();
            db.prop_key_tokens.get_or_create("name").unwrap();
            db.close().unwrap();
        }
        let db = Spider::open(&db_path).unwrap();
        assert_eq!(db.label_tokens.len(), 2);
        assert!(db.label_tokens.contains("Person"));
        assert!(db.label_tokens.contains("Document"));
        assert_eq!(db.edge_type_tokens.len(), 1);
        assert!(db.edge_type_tokens.contains("CONTAINS"));
        assert_eq!(db.prop_key_tokens.len(), 1);
        assert!(db.prop_key_tokens.contains("name"));
    }

    #[test]
    fn drop_flushes_without_explicit_close() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        {
            let mut db = Spider::open(&db_path).unwrap();
            db.metadata.next_node_id = 77;
            db.label_tokens.get_or_create("Entity").unwrap();
        }
        let db = Spider::open(&db_path).unwrap();
        assert_eq!(db.metadata.next_node_id, 77);
        assert!(db.label_tokens.contains("Entity"));
    }

    #[test]
    fn double_close_is_safe() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        let mut db = Spider::open(&db_path).unwrap();
        db.close().unwrap();
        db.close().unwrap();
    }

    #[test]
    fn reopen_preserves_empty_database() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        {
            let mut db = Spider::open(&db_path).unwrap();
            db.close().unwrap();
        }
        let db = Spider::open(&db_path).unwrap();
        assert_eq!(db.metadata.next_node_id, 1);
        assert!(db.label_tokens.is_empty());
    }

    #[test]
    fn corrupt_metadata_detected() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        {
            let mut db = Spider::open(&db_path).unwrap();
            db.close().unwrap();
        }
        fs::write(db_path.join(Spider::META_FILE), b"short").unwrap();
        let result = Spider::open(&db_path);
        assert!(
            matches!(result, Err(crate::error::DbError::CorruptMetadata { .. })),
            "expected CorruptMetadata, got: {result:?}",
        );
    }

    #[test]
    fn path_accessor() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        let db = Spider::open(&db_path).unwrap();
        assert_eq!(db.path(), db_path);
    }

    #[test]
    fn nodes_append_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        let node = Node::new(1, &[], 1_700_000_000, None).unwrap();
        {
            let mut db = Spider::open(&db_path).unwrap();
            db.nodes.append(&[node]).unwrap();
            db.close().unwrap();
        }
        {
            let mut db = Spider::open(&db_path).unwrap();
            let read_back = db.nodes.get(0).unwrap();
            assert_eq!(read_back.id, 1);
            assert_eq!(read_back.significance, Node::DEFAULT_SIGNIFICANCE);
        }
    }
}
