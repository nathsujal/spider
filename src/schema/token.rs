//! # Token Storage
//!
//! [`TokenStore`] provides string interning for labels, relationship types,
//! and property keys. Each unique string is assigned a u8 ID (1-255).
//!
//! ## Design
//!
//! - ID 0 is reserved (means empty/deleted)
//! - Maximum 255 unique tokens per store
//! - O(1) lookup in both directions (name ↔ ID)
//! - Serializable for persistence
//!
//! ## Usage
//!
//! ```rust
//! use spider::schema::TokenStore;
//!
//! let mut labels = TokenStore::new();
//!
//! let person_id = labels.get_or_create("Person").unwrap();
//! let doc_id = labels.get_or_create("Document").unwrap();
//!
//! assert_eq!(person_id, 1);
//! assert_eq!(doc_id, 2);
//! assert_eq!(labels.get_name(1), Some("Person"));
//! ```

use std::collections::HashMap;

/// Bidirectional mapping between string names and u8 IDs.
///
/// Used for:
/// - Labels: "Person" → 1, "Document" → 2, etc.
/// - Relationship types: "KNOWS" → 1, "MENTIONS" → 2, etc.
/// - Property keys: "name" → 1, "age" → 2, etc.
#[derive(Debug, Clone)]
pub struct TokenStore {
    /// ID → name mapping (index = ID).
    tokens: Vec<String>,
    /// Name → ID mapping for O(1) reverse lookup.
    name_to_id: HashMap<String, u8>,
}

impl TokenStore {
    /// Create a new empty token store. ID 0 is reserved.
    pub fn new() -> Self {
        Self {
            tokens: vec![String::new()], // ID 0 = reserved
            name_to_id: HashMap::new(),
        }
    }

    /// Get or create a token ID for a name.
    ///
    /// Returns `None` if:
    /// - The name is empty
    /// - The store is full (255 tokens)
    pub fn get_or_create(&mut self, name: &str) -> Option<u8> {
        if name.is_empty() {
            return None;
        }

        // Return existing ID if found
        if let Some(&id) = self.name_to_id.get(name) {
            return Some(id);
        }

        // Check capacity (ID 0 is reserved, so max 255 tokens)
        if self.tokens.len() >= 256 {
            return None;
        }

        // Create new token
        let id = self.tokens.len() as u8;
        self.tokens.push(name.to_string());
        self.name_to_id.insert(name.to_string(), id);

        Some(id)
    }

    /// Get the ID for a name, or `None` if not found.
    #[inline]
    pub fn get_id(&self, name: &str) -> Option<u8> {
        self.name_to_id.get(name).copied()
    }

    /// Get the name for an ID, or `None` if ID is 0 or invalid.
    #[inline]
    pub fn get_name(&self, id: u8) -> Option<&str> {
        if id == 0 {
            return None;
        }
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// Number of tokens (excluding reserved ID 0).
    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len().saturating_sub(1)
    }

    /// Returns true if no tokens are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all (id, name) pairs.
    pub fn all_tokens(&self) -> impl Iterator<Item = (u8, &str)> {
        self.tokens
            .iter()
            .enumerate()
            .skip(1) // Skip reserved ID 0
            .map(|(id, name)| (id as u8, name.as_str()))
    }

    /// Serialize to bytes.
    ///
    /// Format: `[count: u8] [len: u8, name: [u8]]...`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.len() as u8);

        for name in self.tokens.iter().skip(1) {
            let name_bytes = name.as_bytes();
            bytes.push(name_bytes.len() as u8);
            bytes.extend_from_slice(name_bytes);
        }

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            return None;
        }

        let count = bytes[0] as usize;
        let mut store = Self::new();
        let mut offset = 1;

        for _ in 0..count {
            if offset >= bytes.len() {
                return None;
            }

            let len = bytes[offset] as usize;
            offset += 1;

            if offset + len > bytes.len() {
                return None;
            }

            let name = std::str::from_utf8(&bytes[offset..offset + len]).ok()?;
            store.get_or_create(name)?;
            offset += len;
        }

        Some(store)
    }
}

impl Default for TokenStore {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_usage() {
        let mut store = TokenStore::new();

        let id1 = store.get_or_create("Person").unwrap();
        let id2 = store.get_or_create("Document").unwrap();
        let id3 = store.get_or_create("Person").unwrap(); // Duplicate

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 1); // Same as first

        assert_eq!(store.get_name(1), Some("Person"));
        assert_eq!(store.get_name(2), Some("Document"));
        assert_eq!(store.get_name(0), None); // Reserved
        assert_eq!(store.get_name(3), None); // Not found
    }

    #[test]
    fn lookup() {
        let mut store = TokenStore::new();
        store.get_or_create("Test").unwrap();

        assert_eq!(store.get_id("Test"), Some(1));
        assert_eq!(store.get_id("NotFound"), None);
    }

    #[test]
    fn serialization() {
        let mut store = TokenStore::new();
        store.get_or_create("Person").unwrap();
        store.get_or_create("Document").unwrap();
        store.get_or_create("Entity").unwrap();

        let bytes = store.to_bytes();
        let restored = TokenStore::from_bytes(&bytes).unwrap();

        assert_eq!(store.len(), restored.len());
        assert_eq!(restored.get_name(1), Some("Person"));
        assert_eq!(restored.get_name(2), Some("Document"));
        assert_eq!(restored.get_name(3), Some("Entity"));
    }

    #[test]
    fn empty_name_rejected() {
        let mut store = TokenStore::new();
        assert_eq!(store.get_or_create(""), None);
    }

    #[test]
    fn all_tokens_iterator() {
        let mut store = TokenStore::new();
        store.get_or_create("A").unwrap();
        store.get_or_create("B").unwrap();

        let tokens: Vec<_> = store.all_tokens().collect();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], (1, "A"));
        assert_eq!(tokens[1], (2, "B"));
    }
}
