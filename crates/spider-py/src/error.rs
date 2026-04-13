// Error mapping: spider_core::error::DbError → Python exceptions

use pyo3::PyErr;

use crate::{
    SpiderNotFoundError,
    SpiderCorruptError,
    SpiderIOError,
    SpiderIngestionError,
    SpiderTraversalError,
};

/// Convert a `spider_core::error::DbError` into the appropriate Python exception.
///
/// This function is needed because we can't implement `From<DbError> for PyErr`
/// directly due to Rust's orphan rules (both types are external to this crate).
///
/// Mapping:
/// - NodeNotFound, EdgeNotFound, SourceNodeNotFound, TargetNodeNotFound,
///   BlobNotFound, NotFound, IdOutOfRange, NodeDeleted, PropertyNotFound,
///   DocumentNodeNotFound → SpiderNotFoundError
/// - CorruptMetadata, CorruptRecordFile, BlobHashMismatch → SpiderCorruptError
/// - FileOpen, Io, PathNotFound, VersionMismatch → SpiderIOError
/// - NoPropositions → SpiderIngestionError
/// - TraversalDepthExceeded → SpiderTraversalError
/// - NodeError, EdgeError, PropertyError, TokenError, DynamicError → SpiderNotFoundError
pub fn db_error_to_pyerr(err: spider_core::error::DbError) -> PyErr {
    use spider_core::error::DbError;

    let error_message = err.to_string();

    match err {
        // Not Found errors
        DbError::NodeNotFound(_)
        | DbError::NodeDeleted(_)
        | DbError::EdgeNotFound(_)
        | DbError::SourceNodeNotFound(_)
        | DbError::TargetNodeNotFound(_)
        | DbError::IdOutOfRange { .. }
        | DbError::PropertyNotFound { .. }
        | DbError::BlobNotFound { .. }
        | DbError::NotFound
        | DbError::DocumentNodeNotFound(_) => {
            SpiderNotFoundError::new_err(error_message)
        }

        // Corruption errors
        DbError::CorruptMetadata { .. }
        | DbError::CorruptRecordFile { .. }
        | DbError::BlobHashMismatch { .. } => {
            SpiderCorruptError::new_err(error_message)
        }

        // I/O errors
        DbError::FileOpen { .. }
        | DbError::Io(_)
        | DbError::PathNotFound(_)
        | DbError::VersionMismatch { .. } => {
            SpiderIOError::new_err(error_message)
        }

        // Ingestion errors
        DbError::NoPropositions => {
            SpiderIngestionError::new_err(error_message)
        }

        // Traversal errors
        DbError::TraversalDepthExceeded { .. } => {
            SpiderTraversalError::new_err(error_message)
        }

        // Schema/sub-system errors
        DbError::NodeError(_)
        | DbError::EdgeError(_)
        | DbError::PropertyError(_)
        | DbError::TokenError(_)
        | DbError::DynamicError(_) => {
            SpiderNotFoundError::new_err(error_message)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spider_core::error::DbError;
    use std::path::PathBuf;

    #[test]
    fn node_not_found_error_message_contains_id() {
        let db_err = DbError::NodeNotFound(42);
        let py_err = db_error_to_pyerr(db_err);
        // Check exception type name in error
        assert!(py_err.to_string().contains("SpiderNotFoundError"));
    }

    #[test]
    fn edge_not_found_error_message_contains_id() {
        let db_err = DbError::EdgeNotFound(7);
        let py_err = db_error_to_pyerr(db_err);
        assert!(py_err.to_string().contains("SpiderNotFoundError"));
    }

    #[test]
    fn corrupt_metadata_error_message() {
        let db_err = DbError::CorruptMetadata {
            expected_bytes: 44,
            got_bytes: 32,
        };
        let py_err = db_error_to_pyerr(db_err);
        assert!(py_err.to_string().contains("SpiderCorruptError"));
        assert!(py_err.to_string().contains("44"));
        assert!(py_err.to_string().contains("32"));
    }

    #[test]
    fn corrupt_record_file_error_message() {
        let db_err = DbError::CorruptRecordFile {
            path: PathBuf::from("nodes.db"),
            file_bytes: 100,
            record_size: 29,
        };
        let py_err = db_error_to_pyerr(db_err);
        assert!(py_err.to_string().contains("SpiderCorruptError"));
        assert!(py_err.to_string().contains("nodes.db"));
    }

    #[test]
    fn blob_hash_mismatch_error_message() {
        let db_err = DbError::BlobHashMismatch {
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        let py_err = db_error_to_pyerr(db_err);
        assert!(py_err.to_string().contains("SpiderCorruptError"));
        assert!(py_err.to_string().contains("abc123"));
        assert!(py_err.to_string().contains("def456"));
    }

    #[test]
    fn file_open_error_message() {
        let db_err = DbError::FileOpen {
            path: PathBuf::from("/tmp/test.db"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        };
        let py_err = db_error_to_pyerr(db_err);
        assert!(py_err.to_string().contains("SpiderIOError"));
    }

    #[test]
    fn io_error_message() {
        let db_err = DbError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "denied",
        ));
        let py_err = db_error_to_pyerr(db_err);
        assert!(py_err.to_string().contains("SpiderIOError"));
        assert!(py_err.to_string().contains("denied"));
    }

    #[test]
    fn no_propositions_error_message() {
        let db_err = DbError::NoPropositions;
        let py_err = db_error_to_pyerr(db_err);
        assert!(py_err.to_string().contains("SpiderIngestionError"));
        assert!(py_err.to_string().contains("zero propositions"));
    }

    #[test]
    fn traversal_depth_exceeded_error_message() {
        let db_err = DbError::TraversalDepthExceeded { limit: 10 };
        let py_err = db_error_to_pyerr(db_err);
        assert!(py_err.to_string().contains("SpiderTraversalError"));
        assert!(py_err.to_string().contains("10"));
    }

    #[test]
    fn error_message_contains_db_error_display() {
        let db_err = DbError::NodeNotFound(42);
        let py_err = db_error_to_pyerr(db_err);
        assert!(py_err.to_string().contains("42"));
    }
}
