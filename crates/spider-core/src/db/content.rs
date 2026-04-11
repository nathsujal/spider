//! Blob storage — content-addressable filing with SHA-256 deduplication.
//!
//! Blobs are stored as individual files under `blobs/` within the database
//! directory, named by their SHA-256 hex digest. This gives automatic
//! deduplication: the same content written twice produces one file.
//!
//! ## Directory layout
//!
//! ```text
//! my_graph/
//! ├── nodes.db
//! ├── edges.db
//! └── blobs/
//!     ├── a1b2c3...   # full SHA-256 hex as filename
//!     ├── d4e5f6...
//!     └── ...
//! ```
//!
//! ## Responsibilities
//! - `file()` — store raw bytes, return the SHA-256 hex digest
//! - `read_blob()` — read a blob by its hash
//! - `remove_blob()` — delete a blob file
//! - `has_blob()` — check if a blob exists

use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{DbError, SpiderResult};

/// Subdirectory name for blob storage within the database directory.
const BLOBS_DIR: &str = "blobs";

/// Hex-encode bytes to a lowercase hex string.
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Returns the path to the blobs directory, creating it if missing.
fn ensure_blobs_dir(db_path: &Path) -> SpiderResult<PathBuf> {
    let dir = db_path.join(BLOBS_DIR);
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|e| DbError::FileOpen {
            path: dir.clone(),
            source: e,
        })?;
    }
    Ok(dir)
}

/// Computes the SHA-256 hex digest of the given bytes.
fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex_encode(&result)
}

/// Computes the SHA-256 hex digest of a file's contents.
fn sha256_file_hex(path: &Path) -> SpiderResult<String> {
    let mut file = fs::File::open(path).map_err(|e| DbError::FileOpen {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf).map_err(DbError::Io)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

/// Returns the path to a blob file by its hex digest.
fn blob_path(blobs_dir: &Path, hash: &str) -> PathBuf {
    blobs_dir.join(hash)
}

// --- Public API ---

/// Stores raw bytes as a content-addressable blob.
///
/// Returns the SHA-256 hex digest of the stored content. If a blob with
/// the same content already exists, this is a no-op (deduplication).
///
/// # Errors
/// - [`DbError::Io`] — if the blob directory or file cannot be created
pub fn file(db_path: &Path, data: &[u8]) -> SpiderResult<String> {
    let hash = sha256_hex(data);
    let blobs_dir = ensure_blobs_dir(db_path)?;
    let path = blob_path(&blobs_dir, &hash);

    // If the blob already exists, skip (deduplication).
    if path.exists() {
        return Ok(hash);
    }

    fs::write(&path, data).map_err(DbError::Io)?;
    Ok(hash)
}

/// Reads a blob's contents by its SHA-256 hex digest.
///
/// # Errors
/// - [`DbError::BlobNotFound`] — no blob exists with the given hash
/// - [`DbError::Io`] — if the file cannot be read
pub fn read_blob(db_path: &Path, hash: &str) -> SpiderResult<Vec<u8>> {
    let blobs_dir = ensure_blobs_dir(db_path)?;
    let path = blob_path(&blobs_dir, hash);

    if !path.exists() {
        return Err(DbError::BlobNotFound { hash: hash.to_string() });
    }

    fs::read(&path).map_err(DbError::Io)
}

/// Removes a blob file by its SHA-256 hex digest.
///
/// Returns `true` if the blob was deleted, `false` if it did not exist.
///
/// # Errors
/// - [`DbError::Io`] — if the file cannot be removed
pub fn remove_blob(db_path: &Path, hash: &str) -> SpiderResult<bool> {
    let blobs_dir = ensure_blobs_dir(db_path)?;
    let path = blob_path(&blobs_dir, hash);

    if !path.exists() {
        return Ok(false);
    }

    fs::remove_file(&path).map_err(DbError::Io)?;
    Ok(true)
}

/// Checks whether a blob with the given SHA-256 hex digest exists.
///
/// # Errors
/// - [`DbError::Io`] — if the blob directory cannot be accessed
pub fn has_blob(db_path: &Path, hash: &str) -> SpiderResult<bool> {
    let blobs_dir = ensure_blobs_dir(db_path)?;
    Ok(blob_path(&blobs_dir, hash).exists())
}

/// Files a blob from an existing file on disk.
///
/// Reads the file, stores a copy in the blob store, and returns the
/// SHA-256 hex digest. Also verifies the content hash after writing.
///
/// # Errors
/// - [`DbError::Io`] — if the source file or blob directory cannot be read
/// - [`DbError::BlobHashMismatch`] — if the written blob's hash differs from
///   the source file's hash (indicates a write error)
pub fn file_from_path(db_path: &Path, source: &Path) -> SpiderResult<String> {
    // Read source file.
    let data = fs::read(source).map_err(|e| DbError::FileOpen {
        path: source.to_path_buf(),
        source: e,
    })?;

    // Store in blob store.
    let hash = file(db_path, &data)?;

    // Verify integrity.
    let stored = read_blob(db_path, &hash)?;
    if stored != data {
        return Err(DbError::BlobHashMismatch {
            expected: hash.clone(),
            actual: sha256_hex(&stored),
        });
    }

    Ok(hash)
}

/// Verifies the integrity of a blob against its expected SHA-256 hash.
///
/// Re-hashes the stored blob and compares it to the expected value.
///
/// # Errors
/// - [`DbError::BlobNotFound`] — no blob exists with the given hash
/// - [`DbError::BlobHashMismatch`] — stored content hash doesn't match
/// - [`DbError::Io`] — if the blob cannot be read
pub fn verify_blob(db_path: &Path, expected_hash: &str) -> SpiderResult<()> {
    let blobs_dir = ensure_blobs_dir(db_path)?;
    let path = blob_path(&blobs_dir, expected_hash);

    if !path.exists() {
        return Err(DbError::BlobNotFound { hash: expected_hash.to_string() });
    }

    let actual_hash = sha256_file_hex(&path)?;
    if actual_hash != expected_hash {
        return Err(DbError::BlobHashMismatch {
            expected: expected_hash.to_string(),
            actual: actual_hash,
        });
    }

    Ok(())
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_db_path() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn file_stores_and_returns_hash() {
        let dir = temp_db_path();
        let data = b"hello world";

        let hash = file(dir.path(), data).unwrap();
        assert_eq!(hash.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn file_deduplicates_identical_content() {
        let dir = temp_db_path();
        let data = b"same content";

        let hash1 = file(dir.path(), data).unwrap();
        let hash2 = file(dir.path(), data).unwrap();

        assert_eq!(hash1, hash2);

        // Only one file should exist on disk.
        let blobs_dir = dir.path().join(BLOBS_DIR);
        let count = fs::read_dir(blobs_dir).unwrap().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn read_blob_returns_stored_data() {
        let dir = temp_db_path();
        let data = b"test data for reading";

        let hash = file(dir.path(), data).unwrap();
        let read_back = read_blob(dir.path(), &hash).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    fn read_blob_not_found() {
        let dir = temp_db_path();
        let result = read_blob(dir.path(), "nonexistent_hash");
        assert!(matches!(result, Err(DbError::BlobNotFound { .. })));
    }

    #[test]
    fn remove_blob_deletes_file() {
        let dir = temp_db_path();
        let data = b"to be deleted";

        let hash = file(dir.path(), data).unwrap();
        assert!(has_blob(dir.path(), &hash).unwrap());

        let removed = remove_blob(dir.path(), &hash).unwrap();
        assert!(removed);

        assert!(!has_blob(dir.path(), &hash).unwrap());
        assert!(matches!(
            read_blob(dir.path(), &hash),
            Err(DbError::BlobNotFound { .. })
        ));
    }

    #[test]
    fn remove_blob_nonexistent_returns_false() {
        let dir = temp_db_path();
        let result = remove_blob(dir.path(), "no_such_hash").unwrap();
        assert!(!result);
    }

    #[test]
    fn has_blob_true_after_file() {
        let dir = temp_db_path();
        let data = b"check existence";

        let hash = file(dir.path(), data).unwrap();
        assert!(has_blob(dir.path(), &hash).unwrap());
    }

    #[test]
    fn has_blob_false_before_file() {
        let dir = temp_db_path();
        assert!(!has_blob(dir.path(), "fake_hash").unwrap());
    }

    #[test]
    fn file_from_path_roundtrip() {
        let dir = temp_db_path();

        // Create a source file.
        let source_path = dir.path().join("source.bin");
        let source_data = b"file content to blob";
        fs::write(&source_path, source_data).unwrap();

        let hash = file_from_path(dir.path(), &source_path).unwrap();

        // Verify we can read it back.
        let read_back = read_blob(dir.path(), &hash).unwrap();
        assert_eq!(read_back, source_data);
    }

    #[test]
    fn file_from_path_deduplicates() {
        let dir = temp_db_path();

        let source1 = dir.path().join("src1.bin");
        let source2 = dir.path().join("src2.bin");
        let data = b"identical files";
        fs::write(&source1, data).unwrap();
        fs::write(&source2, data).unwrap();

        let hash1 = file_from_path(dir.path(), &source1).unwrap();
        let hash2 = file_from_path(dir.path(), &source2).unwrap();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn file_from_path_not_found() {
        let dir = temp_db_path();
        let missing = dir.path().join("does_not_exist.bin");
        let result = file_from_path(dir.path(), &missing);
        assert!(matches!(result, Err(DbError::FileOpen { .. })));
    }

    #[test]
    fn verify_blob_valid_hash_passes() {
        let dir = temp_db_path();
        let data = b"verify me";

        let hash = file(dir.path(), data).unwrap();
        verify_blob(dir.path(), &hash).unwrap();
    }

    #[test]
    fn verify_blob_not_found() {
        let dir = temp_db_path();
        let result = verify_blob(dir.path(), "bad_hash");
        assert!(matches!(result, Err(DbError::BlobNotFound { .. })));
    }

    #[test]
    fn verify_blob_corrupted_content() {
        let dir = temp_db_path();
        let data = b"original data";

        let hash = file(dir.path(), data).unwrap();

        // Corrupt the blob file.
        let blobs_dir = dir.path().join(BLOBS_DIR);
        let blob_path = blob_path(&blobs_dir, &hash);
        fs::write(&blob_path, b"corrupted!").unwrap();

        let result = verify_blob(dir.path(), &hash);
        assert!(matches!(result, Err(DbError::BlobHashMismatch { .. })));
    }

    #[test]
    fn different_content_different_hashes() {
        let dir = temp_db_path();

        let hash1 = file(dir.path(), b"data one").unwrap();
        let hash2 = file(dir.path(), b"data two").unwrap();

        assert_ne!(hash1, hash2);

        // Two files should exist.
        let blobs_dir = dir.path().join(BLOBS_DIR);
        let count = fs::read_dir(blobs_dir).unwrap().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn empty_blob() {
        let dir = temp_db_path();
        let hash = file(dir.path(), &[]).unwrap();

        let read_back = read_blob(dir.path(), &hash).unwrap();
        assert!(read_back.is_empty());
    }

    #[test]
    fn large_blob() {
        let dir = temp_db_path();
        let data: Vec<u8> = (0..255).cycle().take(1_000_000).collect();

        let hash = file(dir.path(), &data).unwrap();
        let read_back = read_blob(dir.path(), &hash).unwrap();
        assert_eq!(read_back, data);
    }

    #[test]
    fn sha256_hex_is_deterministic() {
        let data = b"deterministic hash test";
        let h1 = sha256_hex(data);
        let h2 = sha256_hex(data);
        assert_eq!(h1, h2);
    }
}
