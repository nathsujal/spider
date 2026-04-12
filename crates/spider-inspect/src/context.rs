//! Database context — owns the Spider handle and provides shared access.

use anyhow::Result;
use spider_core::db::lifecycle::Spider;
use std::path::Path;

/// Holds the open database connection.
///
/// All REPL commands receive `&mut Context` and access the database
/// through `ctx.db`.
pub struct Context {
    pub db: Spider,
}

impl Context {
    /// Opens a database at the given path, or the platform default.
    pub fn open(path: Option<&Path>) -> Result<Self> {
        let db = match path {
            Some(p) => Spider::open(p),
            None => Spider::open_default(),
        }
        .map_err(|e| anyhow::anyhow!("failed to open database: {e}"))?;

        Ok(Self { db })
    }

    /// Returns a human-readable label for the database location.
    #[allow(dead_code)]
    pub fn db_label(&self) -> &str {
        self.db.path().file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
    }
}
