//! Top-level error type for all `spider-core` database operations.
//!
//! [`DbError`] is the single error type returned by every public method on
//! [`Spider`](crate::db::Spider). Lower-level schema errors
//! ([`NodeError`], [`EdgeError`], etc.) are wrapped via [`From`] impls so
//! the `?` operator works naturally throughout the `db/` layer.
//!
//! ## Design
//!
//! - One error type for all public API — callers pattern-match on a single enum
//! - Schema errors wrapped, not flattened — caller can inspect the inner cause
//! - I/O errors wrapped from `std::io::Error`
//! - All variants carry enough context to produce a useful error message
//! without the caller needing to know internal IDs or offsets

use crate::schema::{
    node::NodeError,
    edge::EdgeError,
    property::PropertyError,
    token::TokenError,
    dynamic::DynamicError,
};

/// Convenience alias — every public `db/` method returns this.
pub type SpiderResult<T> = Result<T, DbError>;

/// All errors that can arise from database operations.
#[derive(Debug)]
pub enum DbError {

    /// The database directory or a required file does not exist.
    PathNotFound(std::path::PathBuf),

    /// A database file could not be opened or created.
    FileOpen {
        path: std::path::PathBuf,
        source: std::io::Error,
    },

    /// The metadata file (`meta.db`) is corrupt or has an unexpected size.
    CorruptMetadata { expected_bytes: usize, got_bytes: usize },

    /// The database was opened by a newer version of spider-core and cannot
    /// be read by this version.
    VersionMismatch { file_version: u32, supported_version: u32 },

    /// A low-level I/O error from reading or writing a record file.
    Io(std::io::Error),

    /// A record file is corrupt — the byte length is not a multiple of the
    /// record size.
    CorruptRecordFile {
        path: std::path::PathBuf,
        file_bytes: u64,
        record_size: usize,
    },

    /// An ID that was expected to be in range was not.
    IdOutOfRange { id: u32, max: u32 },

    /// No live node exists with the given ID.
    NodeNotFound(u32),

    /// The node with this ID has already been deleted (tombstone slot).
    NodeDeleted(u32),

    /// A node-level schema constraint was violated.
    NodeError(NodeError),

    /// No live edge exists with the given ID.
    EdgeNotFound(u32),

    /// The source node referenced by an edge does not exist.
    SourceNodeNotFound(u32),

    /// The target node referenced by an edge does not exist.
    TargetNodeNotFound(u32),

    /// An edge-level schema constraint was violated.
    EdgeError(EdgeError),

    /// The requested property key does not exist on this node or edge.
    PropertyNotFound { owner_id: u32, key: String },

    /// A property-level schema constraint was violated (e.g. int out of range).
    PropertyError(PropertyError),

    /// A token store operation failed (e.g. store full, name too long).
    TokenError(TokenError),

    /// A dynamic record operation failed (e.g. length overflow).
    DynamicError(DynamicError),

    /// No blob exists with the given SHA-256 hash.
    BlobNotFound { hash: String },

    /// The stored blob's content does not match its recorded SHA-256 hash.
    /// Indicates on-disk corruption.
    BlobHashMismatch {
        expected: String,
        actual: String,
    },

    /// The document node referenced during ingestion does not exist.
    DocumentNodeNotFound(u32),

    /// Ingestion produced zero propositions — nothing to wire into the graph.
    NoPropositions,

    /// A traversal depth limit was exceeded.
    TraversalDepthExceeded { limit: usize },

    /// A query returned no results. Not always fatal — callers decide.
    NotFound,
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PathNotFound(p) =>
                write!(f, "database path not found: {}", p.display()),
            Self::FileOpen { path, source } =>
                write!(f, "failed to open {}: {}", path.display(), source),
            Self::CorruptMetadata { expected_bytes, got_bytes } =>
                write!(f, "corrupt metadata: expected {expected_bytes} bytes, got {got_bytes}"),
            Self::VersionMismatch { file_version, supported_version } =>
                write!(f, "database version {file_version} is not supported by this build (supports {supported_version})"),

            Self::Io(e) =>
                write!(f, "I/O error: {e}"),
            Self::CorruptRecordFile { path, file_bytes, record_size } =>
                write!(f, "corrupt record file {}: {file_bytes} bytes is not a multiple of {record_size}", path.display()),
            Self::IdOutOfRange { id, max } =>
                write!(f, "id {id} is out of range (max {max})"),

            Self::NodeNotFound(id) =>
                write!(f, "node {id} not found"),
            Self::NodeDeleted(id) =>
                write!(f, "node {id} has been deleted"),
            Self::NodeError(e) =>
                write!(f, "node error: {e}"),

            Self::EdgeNotFound(id) =>
                write!(f, "edge {id} not found"),
            Self::SourceNodeNotFound(id) =>
                write!(f, "source node {id} not found"),
            Self::TargetNodeNotFound(id) =>
                write!(f, "target node {id} not found"),
            Self::EdgeError(e) =>
                write!(f, "edge error: {e}"),

            Self::PropertyNotFound { owner_id, key } =>
                write!(f, "property '{key}' not found on node/edge {owner_id}"),
            Self::PropertyError(e) =>
                write!(f, "property error: {e}"),

            Self::TokenError(e) =>
                write!(f, "token error: {e}"),

            Self::DynamicError(e) =>
                write!(f, "dynamic storage error: {e}"),

            Self::BlobNotFound { hash } =>
                write!(f, "blob not found: {hash}"),
            Self::BlobHashMismatch { expected, actual } =>
                write!(f, "blob hash mismatch: expected {expected}, got {actual}"),

            Self::DocumentNodeNotFound(id) =>
                write!(f, "document node {id} not found during ingestion"),
            Self::NoPropositions =>
                write!(f, "ingestion produced zero propositions"),

            Self::TraversalDepthExceeded { limit } =>
                write!(f, "traversal depth limit of {limit} exceeded"),
            Self::NotFound =>
                write!(f, "not found"),
        }
    }
}

impl std::error::Error for DbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FileOpen { source, .. } => Some(source),
            Self::Io(e) => Some(e),
            Self::NodeError(e) => Some(e),
            Self::EdgeError(e) => Some(e),
            Self::PropertyError(e) => Some(e),
            Self::TokenError(e) => Some(e),
            Self::DynamicError(e) => Some(e),
            _ => None,
        }
    }
}

// These allow `?` to work directly in any db/ function that returns
// SpiderResult, without manual wrapping at every call site.
impl From<NodeError> for DbError {
    fn from(e: NodeError) -> Self {
        Self::NodeError(e)
    }
}

impl From<EdgeError> for DbError {
    fn from(e: EdgeError) -> Self {
        Self::EdgeError(e)
    }
}

impl From<PropertyError> for DbError {
    fn from(e: PropertyError) -> Self {
        Self::PropertyError(e)
    }
}

impl From<TokenError> for DbError {
    fn from(e: TokenError) -> Self {
        Self::TokenError(e)
    }
}

impl From<DynamicError> for DbError {
    fn from(e: DynamicError) -> Self {
        Self::DynamicError(e)
    }
}

impl From<std::io::Error> for DbError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn node_not_found_message() {
        let e = DbError::NodeNotFound(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn edge_not_found_message() {
        let e = DbError::EdgeNotFound(7);
        assert!(e.to_string().contains("7"));
    }

    #[test]
    fn path_not_found_message() {
        let e = DbError::PathNotFound(PathBuf::from("/tmp/spider.db"));
        assert!(e.to_string().contains("spider.db"));
    }

    #[test]
    fn corrupt_metadata_message() {
        let e = DbError::CorruptMetadata { expected_bytes: 44, got_bytes: 32 };
        let s = e.to_string();
        assert!(s.contains("44"));
        assert!(s.contains("32"));
    }

    #[test]
    fn corrupt_record_file_message() {
        let e = DbError::CorruptRecordFile {
            path: PathBuf::from("nodes.db"),
            file_bytes: 100,
            record_size: 29,
        };
        let s = e.to_string();
        assert!(s.contains("nodes.db"));
        assert!(s.contains("100"));
        assert!(s.contains("29"));
    }

    #[test]
    fn blob_hash_mismatch_message() {
        let e = DbError::BlobHashMismatch {
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("abc123"));
        assert!(s.contains("def456"));
    }

    #[test]
    fn property_not_found_message() {
        let e = DbError::PropertyNotFound { owner_id: 5, key: "name".to_string() };
        let s = e.to_string();
        assert!(s.contains("name"));
        assert!(s.contains("5"));
    }

    #[test]
    fn version_mismatch_message() {
        let e = DbError::VersionMismatch { file_version: 3, supported_version: 1 };
        let s = e.to_string();
        assert!(s.contains("3"));
        assert!(s.contains("1"));
    }

    #[test]
    fn traversal_depth_message() {
        let e = DbError::TraversalDepthExceeded { limit: 10 };
        assert!(e.to_string().contains("10"));
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let db_err: DbError = io_err.into();
        assert!(matches!(db_err, DbError::Io(_)));
        assert!(db_err.to_string().contains("file missing"));
    }

    #[test]
    fn from_node_error() {
        let ne = NodeError::InvalidNodeId(0);
        let db_err: DbError = ne.into();
        assert!(matches!(db_err, DbError::NodeError(_)));
    }

    #[test]
    fn from_edge_error() {
        let ee = EdgeError::SelfLoop(5);
        let db_err: DbError = ee.into();
        assert!(matches!(db_err, DbError::EdgeError(_)));
    }

    #[test]
    fn from_property_error() {
        let pe = PropertyError::IntOutOfRange { value: 999, min: -1, max: 1 };
        let db_err: DbError = pe.into();
        assert!(matches!(db_err, DbError::PropertyError(_)));
    }

    #[test]
    fn from_token_error() {
        let te = TokenError::StoreFull;
        let db_err: DbError = te.into();
        assert!(matches!(db_err, DbError::TokenError(_)));
    }

    #[test]
    fn from_dynamic_error() {
        let de = DynamicError::LengthOverflow { value: 99, max: 50 };
        let db_err: DbError = de.into();
        assert!(matches!(db_err, DbError::DynamicError(_)));
    }

    #[test]
    fn source_some_for_wrapped_errors() {
        use std::error::Error;

        let db_err = DbError::NodeError(NodeError::InvalidNodeId(0));
        assert!(db_err.source().is_some());

        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let db_err = DbError::Io(io_err);
        assert!(db_err.source().is_some());
    }

    #[test]
    fn source_none_for_leaf_errors() {
        use std::error::Error;

        assert!(DbError::NodeNotFound(1).source().is_none());
        assert!(DbError::EdgeNotFound(1).source().is_none());
        assert!(DbError::NotFound.source().is_none());
        assert!(DbError::NoPropositions.source().is_none());
        assert!(DbError::BlobNotFound { hash: "x".to_string() }.source().is_none());
    }

    #[test]
    fn question_mark_from_node_error_compiles() {
        fn inner() -> Result<(), NodeError> {
            Err(NodeError::LabelSlotsFull)
        }
        fn outer() -> SpiderResult<()> {
            inner()?; // NodeError → DbError via From
            Ok(())
        }
        assert!(matches!(outer(), Err(DbError::NodeError(NodeError::LabelSlotsFull))));
    }

    #[test]
    fn question_mark_from_io_error_compiles() {
        fn inner() -> Result<(), std::io::Error> {
            Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe"))
        }
        fn outer() -> SpiderResult<()> {
            inner()?; // io::Error → DbError via From
            Ok(())
        }
        assert!(matches!(outer(), Err(DbError::Io(_))));
    }
}
