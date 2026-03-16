//! Sealed `Record` trait for fixed-size on-disk types.

use std::fmt::Debug;

/// Prevents external crates from implementing [`Record`].
pub(crate) mod private {
    pub trait Sealed {}
}

/// Fixed-size on-disk record, implemented by all schema types.
///
/// Sealed — only types inside `spider-core` can implement this.
pub trait Record: private::Sealed + Copy + Debug + Send + Sync {
    /// Serialized byte size. Used by the store layer for offset computation.
    const SIZE: usize;

    /// Fixed-size byte array for zero-alloc serialization.
    type Bytes: AsRef<[u8]> + AsMut<[u8]> + Copy;

    /// Serialize to exactly [`SIZE`](Self::SIZE) bytes.
    fn to_bytes(&self) -> Self::Bytes;

    /// Deserialize from exactly [`SIZE`](Self::SIZE) bytes.
    fn from_bytes(bytes: Self::Bytes) -> Self;

    /// `true` if this slot is deleted / tombstoned.
    fn is_deleted(&self) -> bool;
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
 
    // --- Minimal test record ---
    //
    // We define a minimal in-test record type to test the trait contract in
    // isolation, without depending on Node or any other schema type.
 
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestRecord {
        id: u32,
        value: u32,
    }
 
    // Seal it so it can implement Record.
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
                id:    u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                value: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            }
        }
 
        fn is_deleted(&self) -> bool {
            self.id == 0
        }
    }
 
    // --- Tombstone record (the zero value) ---
 
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TombstoneRecord;
 
    impl private::Sealed for TombstoneRecord {}
 
    impl Record for TombstoneRecord {
        const SIZE: usize = 4;
        type Bytes = [u8; 4];
 
        fn to_bytes(&self) -> [u8; 4] {
            [0u8; 4]
        }
 
        fn from_bytes(_bytes: [u8; 4]) -> Self {
            TombstoneRecord
        }
 
        fn is_deleted(&self) -> bool {
            true // always a tombstone
        }
    }
 
    // --- SIZE contract ---
 
    #[test]
    fn to_bytes_length_equals_size_constant() {
        let r = TestRecord { id: 1, value: 42 };
        assert_eq!(r.to_bytes().as_ref().len(), TestRecord::SIZE);
    }
 
    #[test]
    fn tombstone_to_bytes_length_equals_size_constant() {
        assert_eq!(TombstoneRecord.to_bytes().as_ref().len(), TombstoneRecord::SIZE);
    }
 
    // --- Round-trip ---
 
    #[test]
    fn round_trip_live_record() {
        let original = TestRecord { id: 7, value: 999 };
        let restored = TestRecord::from_bytes(original.to_bytes());
        assert_eq!(original, restored);
    }
 
    #[test]
    fn round_trip_zero_value() {
        // id=0 is the tombstone, but round-trip should still be exact
        let original = TestRecord { id: 0, value: 0 };
        let restored = TestRecord::from_bytes(original.to_bytes());
        assert_eq!(original, restored);
    }
 
    #[test]
    fn round_trip_max_values() {
        let original = TestRecord { id: u32::MAX, value: u32::MAX };
        let restored = TestRecord::from_bytes(original.to_bytes());
        assert_eq!(original, restored);
    }
 
    // --- is_deleted() ---
 
    #[test]
    fn is_deleted_true_for_tombstone_id() {
        let deleted = TestRecord { id: 0, value: 0 };
        assert!(deleted.is_deleted());
    }
 
    #[test]
    fn is_deleted_false_for_live_record() {
        let live = TestRecord { id: 1, value: 0 };
        assert!(!live.is_deleted());
    }
 
    #[test]
    fn is_deleted_false_for_large_id() {
        let live = TestRecord { id: u32::MAX, value: 0 };
        assert!(!live.is_deleted());
    }
 
    // --- Sealed trait ---
    //
    // We cannot write a compile-fail test inline here without a separate
    // trybuild or compile_fail harness. The seal is verified by the fact that
    // TestRecord above requires `impl private::Sealed for TestRecord {}` to
    // compile — omitting it would produce a compiler error.
    //
    // If you have trybuild in your workspace, add:
    //
    //   #[test]
    //   fn external_impl_is_rejected() {
    //       let t = trybuild::TestCases::new();
    //       t.compile_fail("tests/ui/external_record_impl.rs");
    //   }
 
    // --- Copy bound ---
 
    #[test]
    fn record_is_copy() {
        let r = TestRecord { id: 1, value: 42 };
        let _copy = r; // move
        let _also = r; // copy — would fail if Record weren't Copy
    }
 
    // --- Send + Sync assertions ---
    //
    // These are compile-time checks expressed as a regular test function.
    //
    // The `const` block pattern (used at module level in node.rs) does NOT
    // work inside `#[cfg(test)]` modules because Rust does not allow calling
    // generic non-const functions inside `const` contexts at that scope.
    //
    // A dead `#[test]` fn achieves the same guarantee: if `TestRecord` ever
    // gains a non-Send/Sync field the function body fails to compile, catching
    // the regression at build time with a clear error.
 
    #[test]
    fn record_is_send_and_sync() {
        // These calls are never actually executed — the compiler checks them
        // at monomorphisation time. If T is not Send or Sync, the build fails.
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<TestRecord>();
        assert_sync::<TestRecord>();
    }
 
    // --- Little-endian serialization ---
 
    #[test]
    fn serialization_is_little_endian() {
        let r = TestRecord { id: 0x01020304, value: 0 };
        let bytes = r.to_bytes();
        assert_eq!(bytes[0], 0x04); // LSB first
        assert_eq!(bytes[1], 0x03);
        assert_eq!(bytes[2], 0x02);
        assert_eq!(bytes[3], 0x01);
    }
 
    // --- Debug bound ---
 
    #[test]
    fn record_is_debug_formattable() {
        let r = TestRecord { id: 1, value: 42 };
        // If Debug weren't derived / implemented, this would not compile.
        let s = format!("{:?}", r);
        assert!(s.contains("TestRecord"));
        assert!(s.contains("42"));
    }
}