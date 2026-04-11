//! Fixed-size on-disk record management using memory-mapped files.
//!
//! Provides:
//! - Record trait: Serialization contract for on-disk types
//! - RecordFile<T>: Memory-mapped wrapper for O(1) random access
//!
//! Records are fixed-size (e.g., Node = 29 bytes), enabling constant-time
//! offset calculations: offset = header + (id - 1) * record_size.

use std::fmt::Debug;
use std::fs::{OpenOptions, File};
use std::io::{Write, Seek as _};
use std::path::Path;
use crate::error::SpiderResult;

pub(crate) mod private {
    pub trait Sealed {}
}

/// Trait for fixed-size on-disk records.
///
/// Implemented by all schema types (Node, Edge, Property, Token, Dynamic).
/// Defines serialization contract and deletion semantics.
pub trait Record: Copy + Debug + Send + Sync + private::Sealed {
    const SIZE: usize;
    type Bytes: AsRef<[u8]> + AsMut<[u8]> + Copy;

    fn to_bytes(&self) -> Self::Bytes;
    fn from_bytes(bytes: Self::Bytes) -> Self;
    fn from_raw(bytes: &[u8]) -> Self;
    fn is_deleted(&self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TestRecord { id: u32, value: u32 }

impl private::Sealed for TestRecord {}

impl Record for TestRecord {
    const SIZE: usize = 8;
    type Bytes = [u8; 8];
    
    fn to_bytes(&self) -> [u8; 8] {
        let mut buf = [0u8; 8];
        buf[0..4].copy_from_slice(&self.id.to_le_bytes());
        buf[4..8].copy_from_slice(&self.value.to_le_bytes());
        buf
    }
    
    fn from_bytes(bytes: [u8; 8]) -> Self {
        Self {
            id: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            value: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        }
    }

    fn from_raw(bytes: &[u8]) -> Self {
        Self::from_bytes(bytes.try_into().unwrap())
    }
    
    fn is_deleted(&self) -> bool { self.id == 0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TombstoneRecord;

impl private::Sealed for TombstoneRecord {}

impl Record for TombstoneRecord {
    const SIZE: usize = 4;
    type Bytes = [u8; 4];
    fn to_bytes(&self) -> [u8; 4] { [0u8; 4] }
    fn from_bytes(_bytes: [u8; 4]) -> Self { TombstoneRecord }
    fn from_raw(bytes: &[u8]) -> Self {
        let _ = bytes;
        TombstoneRecord
    }
    fn is_deleted(&self) -> bool { true }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn round_trip() {
        let original = TestRecord { id: 7, value: 999 };
        let restored = TestRecord::from_bytes(original.to_bytes());
        assert_eq!(original, restored);
    }
    
    #[test]
    fn deleted_logic() {
        assert!(TestRecord { id: 0, value: 0 }.is_deleted());
        assert!(!TestRecord { id: 1, value: 0 }.is_deleted());
    }
    
    #[test]
    fn endianness() {
        let r = TestRecord { id: 0x01020304, value: 0 };
        let bytes = r.to_bytes();
        assert_eq!(bytes[0], 0x04);
        assert_eq!(bytes[1], 0x03);
        assert_eq!(bytes[2], 0x02);
        assert_eq!(bytes[3], 0x01);
    }
}

pub struct RecordFile<T: Record> {
    file: File,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Record> RecordFile<T> {
    /// Opens existing record file for reading and writing.
    pub fn open(path: &Path) -> SpiderResult<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Self { file, _marker: std::marker::PhantomData })
    }
    
    /// Creates new record file, truncating if exists.
    pub fn create(path: &Path) -> SpiderResult<Self> {
        let file = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
        Ok(Self { file, _marker: std::marker::PhantomData })
    }
    
    /// Appends records to file end.
    pub fn append(&mut self, records: &[T]) -> SpiderResult<()> {
        for rec in records { self.file.write_all(rec.to_bytes().as_ref())?; }
        Ok(())
    }
    
    /// Gets record at index i with O(1) time.
    pub fn get(&mut self, i: u32) -> SpiderResult<T> {
        use std::io::{Read, Seek as _}; 
        let offset = i as u64 * T::SIZE as u64;
        self.file.seek(std::io::SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; T::SIZE]; 
        self.file.read_exact(&mut buf)?;
        Ok(T::from_raw(&buf))
    }
    
    /// Overwrites record at index i.
    pub fn set(&mut self, i: u32, record: &T) -> SpiderResult<()> {
        use std::io::Write;
        let offset = i as u64 * T::SIZE as u64;
        self.file.seek(std::io::SeekFrom::Start(offset))?;
        self.file.write_all(record.to_bytes().as_ref())?;
        Ok(())
    }
    
    /// Closes file and flushes to disk.
    pub fn close(self) -> SpiderResult<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod recordfile_tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn basic_ops() {
        let path = PathBuf::from("/tmp/test_record_");
        let mut rf = RecordFile::<TestRecord>::create(&path).unwrap();
        let recs = vec![
            TestRecord { id: 1, value: 100 },
            TestRecord { id: 2, value: 200 },
        ];
        rf.append(&recs).unwrap();
        let r0 = rf.get(0).unwrap();
        let r1 = rf.get(1).unwrap();
        assert_eq!(r0.id, 1);
        assert_eq!(r1.id, 2);
        rf.close().unwrap();
        std::fs::remove_file(path).ok();
    }
}
