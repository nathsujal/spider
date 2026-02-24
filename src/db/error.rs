//! # Database Errors
//!
//! Error types for Spider operations.

use std::path::PathBuf;

use crate::store::StoreError;

/// Result type for database operations.
pub type Result<T> = std::result::Result<T, DbError>;

/// Errors that can occur during database operations.
#[derive(Debug)]
pub enum DbError {
    // Store Errors
    /// Wrapped storage layer error.
    Store(StoreError),

    // Node Errors
    /// Node with given ID not found or deleted.
    NodeNotFound(u32),

    /// Too many labels (max 4).
    TooManyLabels { max: usize },

    // Relationship Errors
    /// Source node does not exist.
    SourceNodeNotFound(u32),

    /// Target node does not exist.
    TargetNodeNotFound(u32),

    /// Relationship with given ID not found or deleted.
    RelNotFound(u32),

    // Token Errors
    /// Token store is full (max 256 tokens).
    TokenStoreExhausted { store: &'static str },

    // Value Errors
    /// Value exceeds maximum size.
    ValueTooLarge { max_bytes: usize },

    // Database Errors
    /// Database is already open by another process.
    DatabaseLocked(PathBuf),

    /// Database files are corrupted.
    Corrupted(String),
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(e) => write!(f, "Store error: {}", e),
            Self::NodeNotFound(id) => write!(f, "Node {} not found", id),
            Self::TooManyLabels { max } => write!(f, "Too many labels (max {})", max),
            Self::SourceNodeNotFound(id) => write!(f, "Source node {} not found", id),
            Self::TargetNodeNotFound(id) => write!(f, "Target node {} not found", id),
            Self::RelNotFound(id) => write!(f, "Relationship {} not found", id),
            Self::TokenStoreExhausted { store } => write!(f, "{} store exhausted (max 256)", store),
            Self::ValueTooLarge { max_bytes } => write!(f, "Value too large (max {} bytes)", max_bytes),
            Self::DatabaseLocked(path) => write!(f, "Database locked: {:?}", path),
            Self::Corrupted(reason) => write!(f, "Corrupted: {}", reason),
        }
    }
}

impl std::error::Error for DbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StoreError> for DbError {
    fn from(err: StoreError) -> Self {
        Self::Store(err)
    }
}

impl From<std::io::Error> for DbError {
    fn from(err: std::io::Error) -> Self {
        Self::Store(StoreError::from(err))
    }
}
