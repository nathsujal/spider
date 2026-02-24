//! # Store Errors
//!
//! Error types for store operations.
//!
//! All store operations return [`Result<T>`] which uses [`StoreError`].

use std::io;
use std::path::PathBuf;

/// Result type for store operations.
pub type Result<T> = std::result::Result<T, StoreError>;

/// Errors that can occur during store operations.
#[derive(Debug)]
pub enum StoreError {
    /// I/O error from the operating system.
    Io(io::Error),

    /// File is corrupted or invalid.
    Corrupted { path: PathBuf, reason: String },

    /// Requested ID is out of bounds.
    OutOfBounds { id: u32, capacity: u32 },

    /// Store is full and cannot grow.
    Full,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Corrupted { path, reason } => {
                write!(f, "Corrupted file {:?}: {}", path, reason)
            }
            Self::OutOfBounds { id, capacity } => {
                write!(f, "ID {} out of bounds (capacity: {})", id, capacity)
            }
            Self::Full => write!(f, "Store is full"),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for StoreError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}
