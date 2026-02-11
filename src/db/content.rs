//! Content-addressed blob storage for SpiderDB.
//!
//! Stores binary data (images, audio, PDFs, etc.) as files keyed by
//! SHA256 hash, with 2-char prefix directories (like Git objects).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Default max blob size: 100 MB.
pub const DEFAULT_MAX_BLOB_SIZE: u64 = 100 * 1024 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// MIME Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detect MIME type from magic bytes (file signature).
fn detect_mime_from_bytes(data: &[u8]) -> Option<&'static str> {
    if data.len() < 4 {
        return None;
    }
    // Images
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) { return Some("image/png"); }
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) { return Some("image/jpeg"); }
    if data.starts_with(b"GIF8") { return Some("image/gif"); }
    if data.starts_with(b"BM") { return Some("image/bmp"); }
    if data.len() >= 12 {
        if data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" { return Some("image/webp"); }
        if data.starts_with(b"RIFF") && &data[8..12] == b"WAVE" { return Some("audio/wav"); }
    }
    // Audio
    if data.starts_with(&[0xFF, 0xFB]) || data.starts_with(&[0xFF, 0xF3])
        || data.starts_with(b"ID3") { return Some("audio/mpeg"); }
    if data.starts_with(b"OggS") { return Some("audio/ogg"); }
    if data.starts_with(b"fLaC") { return Some("audio/flac"); }
    // Video
    if data.len() >= 8 && &data[4..8] == b"ftyp" { return Some("video/mp4"); }
    // Documents
    if data.starts_with(b"%PDF") { return Some("application/pdf"); }
    // Archives
    if data.starts_with(&[0x50, 0x4B, 0x03, 0x04]) { return Some("application/zip"); }
    if data.starts_with(&[0x1F, 0x8B]) { return Some("application/gzip"); }

    None
}

/// Detect MIME type from file extension.
fn detect_mime_from_name(name: &str) -> Option<&'static str> {
    let ext = name.rsplit('.').next()?.to_lowercase();
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "bmp" => Some("image/bmp"),
        "svg" => Some("image/svg+xml"),
        "ico" => Some("image/x-icon"),
        "tiff" | "tif" => Some("image/tiff"),
        "wav" => Some("audio/wav"),
        "mp3" => Some("audio/mpeg"),
        "ogg" => Some("audio/ogg"),
        "flac" => Some("audio/flac"),
        "mp4" | "m4v" => Some("video/mp4"),
        "webm" => Some("video/webm"),
        "mov" => Some("video/quicktime"),
        "avi" => Some("video/x-msvideo"),
        "pdf" => Some("application/pdf"),
        "html" | "htm" => Some("text/html"),
        "md" | "markdown" => Some("text/markdown"),
        "txt" => Some("text/plain"),
        "csv" => Some("text/csv"),
        "json" => Some("application/json"),
        "xml" => Some("application/xml"),
        "rs" => Some("text/x-rust"),
        "py" => Some("text/x-python"),
        "js" => Some("text/javascript"),
        "ts" => Some("text/typescript"),
        "c" => Some("text/x-c"),
        "cpp" | "cc" => Some("text/x-c++"),
        "java" => Some("text/x-java"),
        "go" => Some("text/x-go"),
        "toml" => Some("application/toml"),
        "yaml" | "yml" => Some("application/yaml"),
        "zip" => Some("application/zip"),
        "gz" | "gzip" => Some("application/gzip"),
        "tar" => Some("application/x-tar"),
        _ => None,
    }
}

/// Detect MIME type: magic bytes first, then filename, then fallback.
pub fn detect_mime(data: &[u8], name: &str) -> String {
    if let Some(m) = detect_mime_from_bytes(data) { return m.to_string(); }
    if let Some(m) = detect_mime_from_name(name) { return m.to_string(); }
    "application/octet-stream".to_string()
}

/// Get file extension for a MIME type.
fn ext_for_mime(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/bmp" => "bmp",
        "audio/wav" => "wav",
        "audio/mpeg" => "mp3",
        "audio/ogg" => "ogg",
        "audio/flac" => "flac",
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        "application/pdf" => "pdf",
        "application/json" => "json",
        "application/zip" => "zip",
        "text/plain" => "txt",
        "text/html" => "html",
        "text/markdown" => "md",
        "text/x-rust" => "rs",
        "text/x-python" => "py",
        "text/javascript" => "js",
        _ => "bin",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Blob Metadata & Manifest
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata for a single stored blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobMeta {
    pub hash: String,
    pub mime_type: String,
    pub size_bytes: u64,
    pub created_at: u32,
    pub ref_count: u32,
    pub original_name: String,
}

/// Manifest: index of all stored blobs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct BlobManifest {
    version: u32,
    blobs: HashMap<String, BlobMeta>,
}

impl BlobManifest {
    fn new() -> Self {
        Self { version: 1, blobs: HashMap::new() }
    }

    fn load(base: &Path) -> Self {
        let bin_path = base.join("manifest.bin");
        if bin_path.exists() {
            if let Ok(data) = fs::read(&bin_path) {
                if let Ok(m) = bincode::deserialize::<BlobManifest>(&data) {
                    return m;
                }
            }
        }
        Self::new()
    }

    fn save(&self, base: &Path) -> io::Result<()> {
        // Binary (primary)
        let data = bincode::serialize(self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        fs::write(base.join("manifest.bin"), data)?;
        // JSON (debug)
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        fs::write(base.join("manifest.debug.json"), json)?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContentStore
// ─────────────────────────────────────────────────────────────────────────────

/// Content-addressed blob store.
pub struct ContentStore {
    base_path: PathBuf,
    manifest: BlobManifest,
    pub max_blob_size: u64,
}

impl ContentStore {
    /// Open or create a content store at `{db_path}/blobs/`.
    pub fn open(db_path: &Path) -> io::Result<Self> {
        let base_path = db_path.join("blobs");
        fs::create_dir_all(&base_path)?;
        let manifest = BlobManifest::load(&base_path);
        Ok(Self { base_path, manifest, max_blob_size: DEFAULT_MAX_BLOB_SIZE })
    }

    /// Store binary data. Returns `(sha256_hash, mime_type)`.
    /// Deduplicates: if content already exists, increments ref_count.
    pub fn store(&mut self, data: &[u8], name: &str) -> io::Result<(String, String)> {
        let size = data.len() as u64;
        if size > self.max_blob_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Blob size {} exceeds max {} bytes", size, self.max_blob_size),
            ));
        }

        // SHA256 hash
        let hash = {
            let mut h = Sha256::new();
            h.update(data);
            format!("{:x}", h.finalize())
        };

        // Dedup check
        if let Some(meta) = self.manifest.blobs.get_mut(&hash) {
            meta.ref_count += 1;
            return Ok((hash, meta.mime_type.clone()));
        }

        // Detect MIME
        let mime_type = detect_mime(data, name);

        // Write file: blobs/{prefix}/{hash}.{ext}
        let prefix = &hash[..2];
        let ext = ext_for_mime(&mime_type);
        let dir = self.base_path.join(prefix);
        fs::create_dir_all(&dir)?;
        fs::write(dir.join(format!("{}.{}", hash, ext)), data)?;

        // Timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        self.manifest.blobs.insert(hash.clone(), BlobMeta {
            hash: hash.clone(),
            mime_type: mime_type.clone(),
            size_bytes: size,
            created_at: now,
            ref_count: 1,
            original_name: name.to_string(),
        });

        Ok((hash, mime_type))
    }

    /// Read blob data by hash.
    pub fn read(&self, hash: &str) -> io::Result<Vec<u8>> {
        let meta = self.manifest.blobs.get(hash).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("Blob not found: {}", hash))
        })?;
        let prefix = &hash[..2];
        let ext = ext_for_mime(&meta.mime_type);
        fs::read(self.base_path.join(prefix).join(format!("{}.{}", hash, ext)))
    }

    /// Check if blob exists.
    pub fn exists(&self, hash: &str) -> bool {
        self.manifest.blobs.contains_key(hash)
    }

    /// Get metadata.
    pub fn get_meta(&self, hash: &str) -> Option<&BlobMeta> {
        self.manifest.blobs.get(hash)
    }

    /// Decrement ref count.
    pub fn remove_ref(&mut self, hash: &str) {
        if let Some(m) = self.manifest.blobs.get_mut(hash) {
            m.ref_count = m.ref_count.saturating_sub(1);
        }
    }

    /// Garbage collect: remove blobs with ref_count == 0. Returns count removed.
    pub fn gc(&mut self) -> io::Result<usize> {
        let dead: Vec<String> = self.manifest.blobs.iter()
            .filter(|(_, m)| m.ref_count == 0)
            .map(|(h, _)| h.clone())
            .collect();
        let count = dead.len();
        for hash in &dead {
            if let Some(meta) = self.manifest.blobs.get(hash) {
                let prefix = &hash[..2];
                let ext = ext_for_mime(&meta.mime_type);
                let _ = fs::remove_file(
                    self.base_path.join(prefix).join(format!("{}.{}", hash, ext))
                );
                let _ = fs::remove_dir(self.base_path.join(prefix)); // only if empty
            }
            self.manifest.blobs.remove(hash);
        }
        Ok(count)
    }

    /// Number of blobs.
    pub fn blob_count(&self) -> usize { self.manifest.blobs.len() }

    /// Total storage used.
    pub fn total_size(&self) -> u64 {
        self.manifest.blobs.values().map(|m| m.size_bytes).sum()
    }

    /// Flush manifest to disk.
    pub fn flush(&self) -> io::Result<()> {
        self.manifest.save(&self.base_path)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_store() -> (ContentStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = ContentStore::open(dir.path()).unwrap();
        (store, dir)
    }

    // Minimal PNG header
    fn tiny_png() -> Vec<u8> {
        vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,
            0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ]
    }

    #[test]
    fn detect_png() {
        assert_eq!(detect_mime(&tiny_png(), ""), "image/png");
    }

    #[test]
    fn detect_jpeg() {
        assert_eq!(detect_mime(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 0], ""), "image/jpeg");
    }

    #[test]
    fn detect_pdf() {
        assert_eq!(detect_mime(b"%PDF-1.4 stuff", ""), "application/pdf");
    }

    #[test]
    fn detect_from_name_fallback() {
        assert_eq!(detect_mime(b"not magic", "script.py"), "text/x-python");
    }

    #[test]
    fn detect_unknown() {
        assert_eq!(detect_mime(&[1, 2, 3, 4], "mystery"), "application/octet-stream");
    }

    #[test]
    fn store_and_read() {
        let (mut store, _dir) = tmp_store();
        let data = tiny_png();
        let (hash, mime) = store.store(&data, "test.png").unwrap();
        assert_eq!(hash.len(), 64);
        assert_eq!(mime, "image/png");
        assert_eq!(store.read(&hash).unwrap(), data);
    }

    #[test]
    fn dedup() {
        let (mut store, _dir) = tmp_store();
        let data = tiny_png();
        let (h1, _) = store.store(&data, "a.png").unwrap();
        let (h2, _) = store.store(&data, "b.png").unwrap();
        assert_eq!(h1, h2);
        assert_eq!(store.get_meta(&h1).unwrap().ref_count, 2);
        assert_eq!(store.blob_count(), 1);
    }

    #[test]
    fn ref_count_and_gc() {
        let (mut store, _dir) = tmp_store();
        let (hash, _) = store.store(&tiny_png(), "img.png").unwrap();
        assert_eq!(store.get_meta(&hash).unwrap().ref_count, 1);
        store.remove_ref(&hash);
        assert_eq!(store.get_meta(&hash).unwrap().ref_count, 0);
        assert_eq!(store.gc().unwrap(), 1);
        assert!(!store.exists(&hash));
    }

    #[test]
    fn max_size_enforced() {
        let (mut store, _dir) = tmp_store();
        store.max_blob_size = 10;
        assert!(store.store(&tiny_png(), "big.png").is_err());
    }

    #[test]
    fn manifest_persistence() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut store = ContentStore::open(dir.path()).unwrap();
            store.store(&tiny_png(), "persisted.png").unwrap();
            store.flush().unwrap();
        }
        {
            let store = ContentStore::open(dir.path()).unwrap();
            assert_eq!(store.blob_count(), 1);
        }
    }
}
