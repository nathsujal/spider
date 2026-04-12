//! String interning for labels, edge types, and property keys.
//!
//! Each unique string is assigned a [`TokenId`] (`1–255`). ID `0` is reserved
//! as the empty sentinel and can never be constructed through the public API.
//!
//! ## Wire format
//!
//! ```text
//! [count: u8] ([name_len: u8] [name: u8 × name_len]) × count
//! ```
//!
//! Token names are limited to 255 UTF-8 bytes each (so the length prefix fits
//! in one byte).
//!
//! ## Memory layout
//!
//! Each name is heap-allocated **once** as [`Arc<str>`] and shared between the
//! forward vec and the reverse map — no string is ever cloned:
//!
//! ```text
//! tokens:     [Arc(""), Arc("Person"), Arc("Document")]
//!                 ^0         ^1              ^2
//! name_to_id: { Arc("Person")→TokenId(1), Arc("Document")→TokenId(2) }
//!                      ↑ same heap block as tokens[1], ref-count = 2
//! ```

use std::sync::Arc;
use rustc_hash::FxHashMap;

// Compile-time Send + Sync check — zero runtime cost.
// Fails to compile if TokenStore ever gains a non-Send/Sync field.
fn _assert_token_store_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<TokenStore>();
    assert_sync::<TokenStore>();
}

// --- TokenId ---

/// A non-zero interned token ID.
///
/// `0` is the empty sentinel and cannot be constructed through the public API.
/// [`TokenId::new_unchecked`] bypasses the check for the storage layer, which
/// only reads bytes it previously validated on write.
///
/// ## Relationship to domain newtypes
///
/// [`LabelId`], [`EdgeTypeId`], and [`PropKeyId`] all wrap `TokenId`. They
/// represent the same numeric ID but scoped to a specific [`TokenStore`]
/// instance, preventing accidental cross-store misuse.
///
/// [`LabelId`]: crate::schema::node::LabelId
/// [`EdgeTypeId`]: crate::schema::edge::EdgeTypeId
/// [`PropKeyId`]: crate::schema::property::PropKeyId
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TokenId(u8);

impl TokenId {
    /// Returns `Err` if `id == 0`.
    #[inline]
    pub fn new(id: u8) -> Result<Self, TokenError> {
        if id == 0 {
            return Err(TokenError::InvalidId);
        }
        Ok(Self(id))
    }

    /// Construct without checking for zero.
    ///
    /// Caller must guarantee `id != 0`.
    #[inline]
    pub(crate) const fn new_unchecked(id: u8) -> Self {
        Self(id)
    }

    /// The underlying raw value.
    #[inline]
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl std::fmt::Display for TokenId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TokenId({})", self.0)
    }
}

// --- Errors ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenError {
    /// A raw ID value of `0` was given to [`TokenId::new`].
    InvalidId,
    /// Empty string is not a valid token name.
    EmptyName,
    /// Token name exceeds the 255-byte wire-format limit.
    NameTooLong { len: usize, max: usize },
    /// Store is full — all 255 token slots are occupied.
    StoreFull,
}

impl std::fmt::Display for TokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidId =>
                write!(f, "token id 0 is invalid (reserved as the empty sentinel)"),
            Self::EmptyName =>
                write!(f, "token name must not be empty"),
            Self::NameTooLong { len, max } =>
                write!(f, "token name is {len} bytes, maximum is {max}"),
            Self::StoreFull =>
                write!(f, "token store is full (maximum {} tokens)", TokenStore::CAPACITY),
        }
    }
}

impl std::error::Error for TokenError {}

// --- TokenStore ---

/// Maximum UTF-8 byte length of a single token name.
/// Enforced so the wire-format length prefix (`u8`) never overflows.
const MAX_NAME_BYTES: usize = 255;

/// Bidirectional string-intern map: name ↔ [`TokenId`].
///
/// Uses [`FxHashMap`] (non-cryptographic hasher) which is significantly faster
/// than `std::collections::HashMap` for short string keys — exactly the case
/// for label and edge-type names (e.g. `"Person"`, `"KNOWS"`).
#[derive(Debug, Clone)]
pub struct TokenStore {
    /// ID → name. Index 0 holds the reserved empty sentinel `""`.
    tokens: Vec<Arc<str>>,
    /// Name → ID. Shares [`Arc`] heap blocks with `tokens` — no extra allocation.
    name_to_id: FxHashMap<Arc<str>, TokenId>,
}

impl TokenStore {
    /// Maximum number of tokens this store can hold.
    pub const CAPACITY: usize = 255;

    // --- Constructors ---

    /// Create a new empty token store. ID `0` is pre-reserved.
    pub fn new() -> Self {
        let mut tokens = Vec::with_capacity(16);
        tokens.push(Arc::from("")); // slot 0 = reserved sentinel

        Self {
            tokens,
            name_to_id: FxHashMap::default(),
        }
    }

    // --- Core operations ---

    /// Return the existing [`TokenId`] for `name`, or intern it and return a
    /// new one.
    ///
    /// - **Fast path** (already interned): one map lookup, zero allocations.
    /// - **Slow path** (new token): one `Arc<str>` allocation, two pointer clones,
    ///   one map insert.
    ///
    /// # Errors
    /// - [`TokenError::EmptyName`]   — `name` is empty
    /// - [`TokenError::NameTooLong`] — `name` exceeds 255 UTF-8 bytes
    /// - [`TokenError::StoreFull`]   — all 255 slots are occupied
    pub fn get_or_create(&mut self, name: &str) -> Result<TokenId, TokenError> {
        if name.is_empty() {
            return Err(TokenError::EmptyName);
        }
        if name.len() > MAX_NAME_BYTES {
            return Err(TokenError::NameTooLong { len: name.len(), max: MAX_NAME_BYTES });
        }

        // Fast path — already interned
        if let Some(&id) = self.name_to_id.get(name) {
            return Ok(id);
        }

        if self.len() >= Self::CAPACITY {
            return Err(TokenError::StoreFull);
        }

        Ok(self.insert_trusted(name))
    }

    /// Return the [`TokenId`] for `name` without creating a new token.
    ///
    /// Returns `None` if `name` has not been interned. O(1), zero allocations.
    #[inline]
    pub fn get_id(&self, name: &str) -> Option<TokenId> {
        self.name_to_id.get(name).copied()
    }

    /// Return the name for `id`.
    ///
    /// Returns `None` if `id == 0` (reserved) or out of range.
    #[inline]
    pub fn get_name(&self, id: TokenId) -> Option<&str> {
        self.tokens.get(id.get() as usize).map(|s| s.as_ref())
    }

    /// `true` if `name` has been interned in this store.
    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        self.name_to_id.contains_key(name)
    }

    // --- Metadata ---

    /// Number of interned tokens (excluding reserved ID 0).
    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len() - 1
    }

    /// `true` if no tokens have been interned yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tokens.len() == 1
    }

    /// `true` if the store has reached [`Self::CAPACITY`] and cannot accept more tokens.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() >= Self::CAPACITY
    }

    // --- Iteration ---

    /// Iterate over all `(id, name)` pairs in insertion order.
    ///
    /// ID `0` (the reserved sentinel) is never yielded.
    pub fn iter(&self) -> impl Iterator<Item = (TokenId, &str)> {
        self.tokens
            .iter()
            .enumerate()
            .skip(1)
            .map(|(id, arc)| (TokenId::new_unchecked(id as u8), arc.as_ref()))
    }

    // --- Serialization ---

    /// Serialize to wire format: `[count: u8] ([len: u8] [name bytes])…`
    ///
    /// Buffer is pre-allocated to the exact required size.
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload: usize = self.tokens.iter().skip(1).map(|n| 1 + n.len()).sum();
        let mut buf = Vec::with_capacity(1 + payload);

        // len() <= CAPACITY (255) is enforced at insertion — safe cast
        debug_assert!(self.len() <= Self::CAPACITY);
        buf.push(self.len() as u8);

        for arc in self.tokens.iter().skip(1) {
            // name len <= MAX_NAME_BYTES (255) is enforced at insertion — safe cast
            debug_assert!(arc.len() <= MAX_NAME_BYTES);
            buf.push(arc.len() as u8);
            buf.extend_from_slice(arc.as_bytes());
        }

        buf
    }

    /// Deserialize from wire format.
    ///
    /// A single `[0x00]` byte is a valid empty store.
    ///
    /// # Panics
    /// Panics if the byte stream is truncated or contains invalid UTF-8.
    /// This indicates on-disk corruption — the store only reads files it
    /// previously wrote via [`to_bytes`](Self::to_bytes).
    /// Run `spider gc --verify` to assess integrity.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(
            !bytes.is_empty(),
            "Storage corruption: token store file is empty. \
             Run `spider gc --verify` to assess integrity."
        );

        let count = bytes[0] as usize;
        let mut store = Self::new();
        let mut offset = 1;

        for i in 0..count {
            assert!(
                offset < bytes.len(),
                "Storage corruption: token store truncated at token {i}. \
                 Run `spider gc --verify` to assess integrity."
            );

            let name_len = bytes[offset] as usize;
            offset += 1;

            assert!(
                offset + name_len <= bytes.len(),
                "Storage corruption: token store truncated while reading \
                 token {i} (expected {name_len} bytes). \
                 Run `spider gc --verify` to assess integrity."
            );

            let name_bytes = &bytes[offset..offset + name_len];
            offset += name_len;

            let name = std::str::from_utf8(name_bytes).unwrap_or_else(|_| {
                panic!(
                    "Storage corruption: token {i} contains invalid UTF-8. \
                     Run `spider gc --verify` to assess integrity."
                )
            });

            // Skip validation — data came from to_bytes(), already validated on write.
            store.insert_trusted(name);
        }

        store
    }

    // --- Private ---

    /// Insert a name that is already known to be valid:
    /// non-empty, ≤ 255 bytes, store not full.
    ///
    /// Allocates the name once as `Arc<str>` then shares it between `tokens`
    /// (forward) and `name_to_id` (reverse) via a cheap pointer clone.
    fn insert_trusted(&mut self, name: &str) -> TokenId {
        let arc: Arc<str> = Arc::from(name);
        // len() < CAPACITY is guaranteed by caller — safe cast
        let id = TokenId::new_unchecked(self.tokens.len() as u8);
        self.tokens.push(Arc::clone(&arc));   // refcount → 2, no allocation
        self.name_to_id.insert(arc, id);       // refcount stays at 2
        id
    }
}

impl Default for TokenStore {
    fn default() -> Self {
        Self::new()
    }
}

// Two stores are equal when they contain the same tokens in the same insertion
// order. `name_to_id` is fully derived from `tokens`, so comparing it would
// be redundant.
impl PartialEq for TokenStore {
    fn eq(&self, other: &Self) -> bool {
        self.tokens == other.tokens
    }
}

impl Eq for TokenStore {}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    // --- TokenId ---

    #[test]
    fn token_id_rejects_zero() {
        assert_eq!(TokenId::new(0), Err(TokenError::InvalidId));
    }

    #[test]
    fn token_id_accepts_nonzero() {
        assert_eq!(TokenId::new(5).unwrap().get(), 5);
    }

    #[test]
    fn token_id_unchecked_preserves_value() {
        assert_eq!(TokenId::new_unchecked(7).get(), 7);
    }

    #[test]
    fn token_id_max_value() {
        assert_eq!(TokenId::new(255).unwrap().get(), 255);
    }

    // --- get_or_create ---

    #[test]
    fn assigns_sequential_ids() {
        let mut s = TokenStore::new();
        assert_eq!(s.get_or_create("Person").unwrap(),   TokenId::new_unchecked(1));
        assert_eq!(s.get_or_create("Document").unwrap(), TokenId::new_unchecked(2));
        assert_eq!(s.get_or_create("Entity").unwrap(),   TokenId::new_unchecked(3));
    }

    #[test]
    fn returns_existing_id_for_duplicate() {
        let mut s = TokenStore::new();
        let a = s.get_or_create("Person").unwrap();
        let b = s.get_or_create("Person").unwrap();
        assert_eq!(a, b);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn rejects_empty_name() {
        assert_eq!(TokenStore::new().get_or_create("").unwrap_err(), TokenError::EmptyName);
    }

    #[test]
    fn rejects_name_too_long() {
        let mut s = TokenStore::new();
        let long = "x".repeat(256);
        assert_eq!(
            s.get_or_create(&long).unwrap_err(),
            TokenError::NameTooLong { len: 256, max: 255 }
        );
    }

    #[test]
    fn accepts_name_at_max_length() {
        let mut s = TokenStore::new();
        let name = "a".repeat(255);
        let id = s.get_or_create(&name).unwrap();
        assert_eq!(s.get_name(id), Some(name.as_str()));
    }

    #[test]
    fn rejects_when_full() {
        let mut s = TokenStore::new();
        for i in 0..TokenStore::CAPACITY {
            s.get_or_create(&format!("t{i}")).unwrap();
        }
        assert!(s.is_full());
        assert_eq!(s.get_or_create("overflow").unwrap_err(), TokenError::StoreFull);
    }

    #[test]
    fn existing_token_accessible_when_full() {
        let mut s = TokenStore::new();
        for i in 0..TokenStore::CAPACITY {
            s.get_or_create(&format!("t{i}")).unwrap();
        }
        assert_eq!(s.get_or_create("t0").unwrap(), TokenId::new_unchecked(1));
    }

    // --- get_id / get_name / contains ---

    #[test]
    fn get_id_known_name() {
        let mut s = TokenStore::new();
        s.get_or_create("Test").unwrap();
        assert_eq!(s.get_id("Test"), Some(TokenId::new_unchecked(1)));
    }

    #[test]
    fn get_id_unknown_name() {
        assert_eq!(TokenStore::new().get_id("X"), None);
    }

    #[test]
    fn get_name_known_id() {
        let mut s = TokenStore::new();
        s.get_or_create("Hello").unwrap();
        assert_eq!(s.get_name(TokenId::new_unchecked(1)), Some("Hello"));
    }

    #[test]
    fn get_name_out_of_range_is_none() {
        assert_eq!(TokenStore::new().get_name(TokenId::new_unchecked(99)), None);
    }

    #[test]
    fn contains_true_for_interned() {
        let mut s = TokenStore::new();
        s.get_or_create("X").unwrap();
        assert!(s.contains("X"));
    }

    #[test]
    fn contains_false_for_unknown() {
        assert!(!TokenStore::new().contains("X"));
    }

    // --- Metadata ---

    #[test]
    fn new_store_is_empty_not_full() {
        let s = TokenStore::new();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert!(!s.is_full());
    }

    #[test]
    fn len_tracks_insertions() {
        let mut s = TokenStore::new();
        s.get_or_create("A").unwrap();
        assert_eq!(s.len(), 1);
        assert!(!s.is_empty());
        s.get_or_create("B").unwrap();
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn len_unchanged_for_duplicate() {
        let mut s = TokenStore::new();
        s.get_or_create("A").unwrap();
        s.get_or_create("A").unwrap();
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn is_full_at_capacity() {
        let mut s = TokenStore::new();
        for i in 0..TokenStore::CAPACITY {
            s.get_or_create(&format!("t{i}")).unwrap();
        }
        assert!(s.is_full());
        assert_eq!(s.len(), TokenStore::CAPACITY);
    }

    // --- iter ---

    #[test]
    fn iter_insertion_order() {
        let mut s = TokenStore::new();
        s.get_or_create("A").unwrap();
        s.get_or_create("B").unwrap();
        let pairs: Vec<_> = s.iter().collect();
        assert_eq!(pairs, vec![
            (TokenId::new_unchecked(1), "A"),
            (TokenId::new_unchecked(2), "B"),
        ]);
    }

    #[test]
    fn iter_never_yields_id_zero() {
        let mut s = TokenStore::new();
        s.get_or_create("X").unwrap();
        assert!(s.iter().all(|(id, _)| id != TokenId::new_unchecked(0)));
    }

    #[test]
    fn iter_empty_yields_nothing() {
        assert_eq!(TokenStore::new().iter().count(), 0);
    }

    // --- PartialEq ---

    #[test]
    fn equal_stores_are_equal() {
        let mut a = TokenStore::new();
        let mut b = TokenStore::new();
        a.get_or_create("X").unwrap();
        b.get_or_create("X").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_tokens_are_not_equal() {
        let mut a = TokenStore::new();
        let mut b = TokenStore::new();
        a.get_or_create("X").unwrap();
        b.get_or_create("Y").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn same_tokens_different_order_are_not_equal() {
        let mut a = TokenStore::new();
        let mut b = TokenStore::new();
        a.get_or_create("X").unwrap(); a.get_or_create("Y").unwrap();
        b.get_or_create("Y").unwrap(); b.get_or_create("X").unwrap();
        // same names, different insertion order → different IDs → not equal
        assert_ne!(a, b);
    }

    #[test]
    fn empty_stores_are_equal() {
        assert_eq!(TokenStore::new(), TokenStore::new());
    }

    // --- Serialization ---

    #[test]
    fn round_trip_multiple_tokens() {
        let mut s = TokenStore::new();
        s.get_or_create("Person").unwrap();
        s.get_or_create("Document").unwrap();
        s.get_or_create("Entity").unwrap();
        assert_eq!(TokenStore::from_bytes(&s.to_bytes()), s);
    }

    #[test]
    fn round_trip_empty_store() {
        let s = TokenStore::new();
        assert_eq!(s.to_bytes(), vec![0x00]);
        assert_eq!(TokenStore::from_bytes(&s.to_bytes()), s);
    }

    #[test]
    fn round_trip_unicode() {
        let mut s = TokenStore::new();
        s.get_or_create("日本語").unwrap();
        s.get_or_create("Ünïcödé").unwrap();
        let r = TokenStore::from_bytes(&s.to_bytes());
        assert_eq!(r.get_name(TokenId::new_unchecked(1)), Some("日本語"));
        assert_eq!(r.get_name(TokenId::new_unchecked(2)), Some("Ünïcödé"));
    }

    #[test]
    fn round_trip_ids_are_stable() {
        let mut s = TokenStore::new();
        let id_a = s.get_or_create("A").unwrap();
        let id_b = s.get_or_create("B").unwrap();
        let r = TokenStore::from_bytes(&s.to_bytes());
        assert_eq!(r.get_id("A"), Some(id_a));
        assert_eq!(r.get_id("B"), Some(id_b));
    }

    #[test]
    #[should_panic(expected = "Storage corruption: token store file is empty")]
    fn from_bytes_panics_on_empty_input() {
        TokenStore::from_bytes(&[]);
    }

    #[test]
    #[should_panic(expected = "Storage corruption: token store truncated at token 0")]
    fn from_bytes_panics_on_missing_name_length() {
        TokenStore::from_bytes(&[0x01]); // count=1 but no name follows
    }

    #[test]
    #[should_panic(expected = "Storage corruption: token store truncated while reading token 0")]
    fn from_bytes_panics_on_truncated_name() {
        let bytes = vec![0x01, 0x05, b'A', b'B']; // count=1, len=5, only 2 bytes
        TokenStore::from_bytes(&bytes);
    }

    #[test]
    #[should_panic(expected = "Storage corruption: token 0 contains invalid UTF-8")]
    fn from_bytes_panics_on_invalid_utf8() {
        let bytes = vec![0x01, 0x02, 0xFF, 0xFE];
        TokenStore::from_bytes(&bytes);
    }

    // --- Arc sharing ---

    #[test]
    fn vec_and_map_share_same_arc_allocation() {
        let mut s = TokenStore::new();
        s.get_or_create("Person").unwrap();

        let from_vec = &s.tokens[1];
        let from_map = s.name_to_id.get("Person")
            .map(|id| &s.tokens[id.get() as usize])
            .unwrap();

        // Arc::ptr_eq confirms both sides point to the same heap block
        assert!(Arc::ptr_eq(from_vec, from_map));
    }

    // --- Send + Sync ---

    #[test]
    fn token_store_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<TokenStore>();
        assert_sync::<TokenStore>();
    }

    // --- Error display ---

    #[test]
    fn error_messages_are_readable() {
        assert!(TokenError::InvalidId.to_string().contains("0"));
        assert!(TokenError::EmptyName.to_string().contains("empty"));
        assert!(TokenError::NameTooLong { len: 300, max: 255 }.to_string().contains("300"));
        assert!(TokenError::StoreFull.to_string().contains("255"));
    }
}