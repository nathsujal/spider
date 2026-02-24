//! # Record File
//!
//! Memory-mapped file for fixed-size records.
//!
//! ## File Layout
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │ HEADER (64 bytes)                            │
//! │ ├─ magic: [u8; 4] = "SPDR"                   │
//! │ ├─ version: u8                               │
//! │ ├─ record_size: u16                          │
//! │ └─ capacity: u32                             │
//! ├──────────────────────────────────────────────┤
//! │ RECORDS                                      │
//! │ offset = 64 + (id - 1) * record_size         │
//! └──────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **Auto-grow**: File expands when capacity exceeded
//! - **Periodic flush**: Every 1000 writes
//! - **Header validation**: Magic bytes and version check

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use memmap2::MmapMut;

use super::{Record, Result, StoreError};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// File header magic bytes.
const MAGIC: [u8; 4] = *b"SPDR";

/// File format version.
const VERSION: u8 = 1;

/// Header size in bytes.
const HEADER_SIZE: usize = 64;

/// Default initial capacity.
const DEFAULT_CAPACITY: u32 = 1024;

/// Flush interval (writes between flushes).
const FLUSH_INTERVAL: u32 = 1000;

// ─────────────────────────────────────────────────────────────────────────────
// RecordFile
// ─────────────────────────────────────────────────────────────────────────────

/// Memory-mapped file for fixed-size records.
///
/// ID 0 is reserved (deleted/empty). Valid IDs start at 1.
pub struct RecordFile<R: Record> {
    mmap: MmapMut,
    file: File,
    path: PathBuf,
    capacity: u32,
    write_count: u32,
    _phantom: PhantomData<R>,
}

impl<R: Record> RecordFile<R> {
    /// Open or create a record file.
    pub fn open(path: &Path) -> Result<Self> {
        if path.exists() {
            Self::open_existing(path)
        } else {
            Self::create_new(path, DEFAULT_CAPACITY)
        }
    }

    /// Open existing file with header validation.
    fn open_existing(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // Validate magic
        if mmap.len() < HEADER_SIZE || &mmap[0..4] != &MAGIC {
            return Err(StoreError::Corrupted {
                path: path.to_path_buf(),
                reason: "Invalid magic bytes".into(),
            });
        }

        // Validate version
        let version = mmap[4];
        if version > VERSION {
            return Err(StoreError::Corrupted {
                path: path.to_path_buf(),
                reason: format!("Unsupported version {} (max: {})", version, VERSION),
            });
        }

        // Validate record size
        let stored_size = u16::from_le_bytes([mmap[5], mmap[6]]) as usize;
        if stored_size != R::SIZE {
            return Err(StoreError::Corrupted {
                path: path.to_path_buf(),
                reason: format!("Record size mismatch: {} vs {}", stored_size, R::SIZE),
            });
        }

        let capacity = u32::from_le_bytes([mmap[7], mmap[8], mmap[9], mmap[10]]);

        Ok(Self {
            mmap,
            file,
            path: path.to_path_buf(),
            capacity,
            write_count: 0,
            _phantom: PhantomData,
        })
    }

    /// Create new file with given capacity.
    fn create_new(path: &Path, capacity: u32) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file_size = HEADER_SIZE + (capacity as usize * R::SIZE);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        file.set_len(file_size as u64)?;

        // Write header
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4] = VERSION;
        header[5..7].copy_from_slice(&(R::SIZE as u16).to_le_bytes());
        header[7..11].copy_from_slice(&capacity.to_le_bytes());

        file.write_all(&header)?;
        file.flush()?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(Self {
            mmap,
            file,
            path: path.to_path_buf(),
            capacity,
            write_count: 0,
            _phantom: PhantomData,
        })
    }

    /// Read record by ID. Returns `None` if ID is 0 or out of bounds.
    pub fn read(&self, id: u32) -> Option<R> {
        if id == 0 || id > self.capacity {
            return None;
        }

        let offset = self.offset_for(id);
        if offset + R::SIZE > self.mmap.len() {
            return None;
        }

        Some(R::from_bytes(&self.mmap[offset..offset + R::SIZE]))
    }

    /// Write record at ID. Auto-grows if needed.
    pub fn write(&mut self, id: u32, record: &R) -> Result<()> {
        if id == 0 {
            return Err(StoreError::OutOfBounds { id, capacity: self.capacity });
        }

        if id > self.capacity {
            self.grow(id.max(self.capacity * 2))?;
        }

        let offset = self.offset_for(id);
        self.mmap[offset..offset + R::SIZE].copy_from_slice(&record.to_bytes());

        self.write_count += 1;
        if self.write_count % FLUSH_INTERVAL == 0 {
            self.mmap.flush()?;
        }

        Ok(())
    }

    /// Grow file to new capacity.
    pub fn grow(&mut self, new_capacity: u32) -> Result<()> {
        if new_capacity <= self.capacity {
            return Ok(());
        }

        self.mmap.flush()?;

        let new_size = HEADER_SIZE + (new_capacity as usize * R::SIZE);
        self.file.set_len(new_size as u64)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };

        self.capacity = new_capacity;
        self.mmap[7..11].copy_from_slice(&new_capacity.to_le_bytes());

        Ok(())
    }

    /// Flush all pending writes to disk.
    pub fn sync(&mut self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Current capacity.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// File path.
    #[inline]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Calculate byte offset for record ID.
    #[inline]
    fn offset_for(&self, id: u32) -> usize {
        HEADER_SIZE + ((id - 1) as usize * R::SIZE)
    }
}

impl<R: Record> Drop for RecordFile<R> {
    fn drop(&mut self) {
        let _ = self.mmap.flush();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct TestRecord {
        id: u32,
        value: u32,
    }

    impl Record for TestRecord {
        const SIZE: usize = 8;

        fn to_bytes(&self) -> Vec<u8> {
            let mut bytes = Vec::with_capacity(8);
            bytes.extend_from_slice(&self.id.to_le_bytes());
            bytes.extend_from_slice(&self.value.to_le_bytes());
            bytes
        }

        fn from_bytes(bytes: &[u8]) -> Self {
            Self {
                id: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                value: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            }
        }

        fn is_deleted(&self) -> bool {
            self.id == 0
        }
    }

    #[test]
    fn create_and_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        {
            let mut store = RecordFile::<TestRecord>::open(&path).unwrap();
            store.write(1, &TestRecord { id: 1, value: 100 }).unwrap();
            store.write(2, &TestRecord { id: 2, value: 200 }).unwrap();
            store.sync().unwrap();
        }

        {
            let store = RecordFile::<TestRecord>::open(&path).unwrap();
            assert_eq!(store.read(1).unwrap().value, 100);
            assert_eq!(store.read(2).unwrap().value, 200);
        }
    }

    #[test]
    fn bounds_checking() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let store = RecordFile::<TestRecord>::open(&path).unwrap();

        assert!(store.read(0).is_none());
        assert!(store.read(1).unwrap().is_deleted());
    }

    #[test]
    fn auto_grow() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let mut store = RecordFile::<TestRecord>::open(&path).unwrap();
        let initial = store.capacity();

        let big_id = initial + 100;
        store.write(big_id, &TestRecord { id: big_id, value: 999 }).unwrap();

        assert!(store.capacity() > initial);
        assert_eq!(store.read(big_id).unwrap().value, 999);
    }
}
