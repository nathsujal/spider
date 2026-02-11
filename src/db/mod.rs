//! # SpiderDB
//!
//! Main database struct combining all storage components.
//!
//! ## Example
//!
//! ```rust,ignore
//! use spider::db::SpiderDB;
//!
//! // Open or create database
//! let mut db = SpiderDB::open("./data/mydb")?;
//!
//! // Create a node
//! let node_id = db.create_node(&["Person"])?;
//!
//! // Create a relationship
//! let rel_id = db.create_rel(node_id, other_id, "KNOWS")?;
//!
//! // Close (auto-flushes)
//! db.close()?;
//! ```

mod error;
pub mod property;
pub mod query;

pub use error::{DbError, Result};
pub use property::PropertyValue;
pub use query::{Direction, RelInfo};

use std::path::{Path, PathBuf};

use crate::schema::{
    DynamicArrayRecord, DynamicStringRecord, NodeRecord, PropertyRecord, RelRecord, TokenStore,
};
use crate::store::{FreeList, Metadata, RecordFile};

// ─────────────────────────────────────────────────────────────────────────────
// SpiderDB
// ─────────────────────────────────────────────────────────────────────────────

/// Main Spider graph database.
pub struct SpiderDB {
    path: PathBuf,

    // Record files
    nodes: RecordFile<NodeRecord>,
    rels: RecordFile<RelRecord>,
    props: RecordFile<PropertyRecord>,
    strings: RecordFile<DynamicStringRecord>,
    arrays: RecordFile<DynamicArrayRecord>,

    // Free lists for ID reuse
    node_free: FreeList,
    rel_free: FreeList,
    prop_free: FreeList,
    string_free: FreeList,
    array_free: FreeList,

    // Metadata (next IDs)
    meta: Metadata,

    // Token stores
    labels: TokenStore,
    rel_types: TokenStore,
    prop_keys: TokenStore,
}

impl SpiderDB {
    /// Open or create a database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;

        // Open record files
        let nodes = RecordFile::open(&path.join("nodes.db"))?;
        let rels = RecordFile::open(&path.join("rels.db"))?;
        let props = RecordFile::open(&path.join("props.db"))?;
        let strings = RecordFile::open(&path.join("strings.db"))?;
        let arrays = RecordFile::open(&path.join("arrays.db"))?;

        // Load metadata
        let meta = Metadata::load(&path.join("meta.db"))?;

        // Load token stores
        let labels = TokenStore::load(&path.join("labels.tok")).unwrap_or_default();
        let rel_types = TokenStore::load(&path.join("rel_types.tok")).unwrap_or_default();
        let prop_keys = TokenStore::load(&path.join("prop_keys.tok")).unwrap_or_default();

        // Load free lists (or create empty)
        let node_free = Self::load_free_list(&path.join("node_free.bin"));
        let rel_free = Self::load_free_list(&path.join("rel_free.bin"));
        let prop_free = Self::load_free_list(&path.join("prop_free.bin"));
        let string_free = Self::load_free_list(&path.join("string_free.bin"));
        let array_free = Self::load_free_list(&path.join("array_free.bin"));

        Ok(Self {
            path,
            nodes,
            rels,
            props,
            strings,
            arrays,
            node_free,
            rel_free,
            prop_free,
            string_free,
            array_free,
            meta,
            labels,
            rel_types,
            prop_keys,
        })
    }

    /// Close the database, flushing all data to disk.
    pub fn close(&mut self) -> Result<()> {
        // Sync record files
        self.nodes.sync()?;
        self.rels.sync()?;
        self.props.sync()?;
        self.strings.sync()?;
        self.arrays.sync()?;

        // Save metadata
        self.meta.save(&self.path.join("meta.db"))?;

        // Save token stores
        self.labels.save(&self.path.join("labels.tok"))?;
        self.rel_types.save(&self.path.join("rel_types.tok"))?;
        self.prop_keys.save(&self.path.join("prop_keys.tok"))?;

        // Save free lists
        Self::save_free_list(&self.node_free, &self.path.join("node_free.bin"))?;
        Self::save_free_list(&self.rel_free, &self.path.join("rel_free.bin"))?;
        Self::save_free_list(&self.prop_free, &self.path.join("prop_free.bin"))?;
        Self::save_free_list(&self.string_free, &self.path.join("string_free.bin"))?;
        Self::save_free_list(&self.array_free, &self.path.join("array_free.bin"))?;

        Ok(())
    }

    // ─── Node Operations ─────────────────────────────────────────────────────

    /// Create a new node with the given labels.
    ///
    /// Returns the new node's ID.
    pub fn create_node(&mut self, labels: &[&str]) -> Result<u32> {
        // Validate label count
        if labels.len() > 4 {
            return Err(DbError::TooManyLabels { max: 4 });
        }

        // Convert label names to IDs
        let mut label_ids = [0u8; 4];
        for (i, label) in labels.iter().take(4).enumerate() {
            label_ids[i] = self.labels.get_or_create(label).ok_or(DbError::TokenStoreExhausted {
                store: "labels",
            })?;
        }

        // Allocate ID
        let id = self.node_free.allocate(&mut self.meta.next_node_id);

        // Create and write record
        let node = NodeRecord::new(id, &label_ids[..labels.len()]);
        self.nodes.write(id, &node)?;

        Ok(id)
    }

    /// Get a node by ID. Returns `None` if not found or deleted.
    pub fn get_node(&self, id: u32) -> Option<NodeRecord> {
        let node = self.nodes.read(id)?;
        if node.is_deleted() {
            None
        } else {
            Some(node)
        }
    }

    /// Delete a node by ID.
    ///
    /// Also deletes all relationships connected to this node.
    pub fn delete_node(&mut self, id: u32) -> Result<()> {
        let node = self.nodes.read(id).ok_or(DbError::NodeNotFound(id))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(id));
        }

        // Delete all relationships from this node
        let mut rel_id = node.first_rel_id;
        while rel_id != 0 {
            if let Some(rel) = self.rels.read(rel_id) {
                let next = if rel.source_id == id {
                    rel.next_rel_source
                } else {
                    rel.next_rel_target
                };
                self.delete_rel_internal(rel_id)?;
                rel_id = next;
            } else {
                break;
            }
        }

        // Mark node as deleted
        let empty = NodeRecord::empty();
        self.nodes.write(id, &empty)?;
        self.node_free.free(id);

        Ok(())
    }

    // ─── Relationship Operations ─────────────────────────────────────────────

    /// Create a relationship between two nodes.
    ///
    /// Returns the new relationship's ID.
    pub fn create_rel(&mut self, source_id: u32, target_id: u32, rel_type: &str) -> Result<u32> {
        // Validate nodes exist
        let mut source = self.nodes.read(source_id).ok_or(DbError::SourceNodeNotFound(source_id))?;
        if source.is_deleted() {
            return Err(DbError::SourceNodeNotFound(source_id));
        }

        let mut target = self.nodes.read(target_id).ok_or(DbError::TargetNodeNotFound(target_id))?;
        if target.is_deleted() {
            return Err(DbError::TargetNodeNotFound(target_id));
        }

        // Get rel type ID
        let rel_type_id = self.rel_types.get_or_create(rel_type).ok_or(DbError::TokenStoreExhausted {
            store: "rel_types",
        })?;

        // Allocate ID
        let id = self.rel_free.allocate(&mut self.meta.next_rel_id);

        // Create relationship
        let mut rel = RelRecord::new(id, source_id, target_id, rel_type_id);

        // Link into source's chain
        if source.first_rel_id != 0 {
            if let Some(mut old_first) = self.rels.read(source.first_rel_id) {
                if old_first.source_id == source_id {
                    old_first.prev_rel_source = id;
                } else {
                    old_first.prev_rel_target = id;
                }
                self.rels.write(source.first_rel_id, &old_first)?;
            }
            rel.next_rel_source = source.first_rel_id;
        }
        source.first_rel_id = id;
        self.nodes.write(source_id, &source)?;

        // Link into target's chain (if different from source)
        if target_id != source_id {
            if target.first_rel_id != 0 {
                if let Some(mut old_first) = self.rels.read(target.first_rel_id) {
                    if old_first.source_id == target_id {
                        old_first.prev_rel_source = id;
                    } else {
                        old_first.prev_rel_target = id;
                    }
                    self.rels.write(target.first_rel_id, &old_first)?;
                }
                rel.next_rel_target = target.first_rel_id;
            }
            target.first_rel_id = id;
            self.nodes.write(target_id, &target)?;
        }

        // Write relationship
        self.rels.write(id, &rel)?;

        Ok(id)
    }

    /// Get a relationship by ID.
    pub fn get_rel(&self, id: u32) -> Option<RelRecord> {
        let rel = self.rels.read(id)?;
        if rel.is_deleted() {
            None
        } else {
            Some(rel)
        }
    }

    /// Delete a relationship by ID.
    pub fn delete_rel(&mut self, id: u32) -> Result<()> {
        self.delete_rel_internal(id)
    }

    /// Internal relationship deletion with chain unlinking.
    fn delete_rel_internal(&mut self, id: u32) -> Result<()> {
        let rel = self.rels.read(id).ok_or(DbError::RelNotFound(id))?;
        if rel.is_deleted() {
            return Err(DbError::RelNotFound(id));
        }

        // Unlink from source chain
        self.unlink_rel_from_node(rel.source_id, id, &rel, true)?;

        // Unlink from target chain (if different)
        if rel.target_id != rel.source_id {
            self.unlink_rel_from_node(rel.target_id, id, &rel, false)?;
        }

        // Mark as deleted
        let empty = RelRecord::empty();
        self.rels.write(id, &empty)?;
        self.rel_free.free(id);

        Ok(())
    }

    /// Unlink relationship from a node's chain.
    fn unlink_rel_from_node(
        &mut self,
        node_id: u32,
        _rel_id: u32,
        rel: &RelRecord,
        is_source: bool,
    ) -> Result<()> {
        let (prev_id, next_id) = if is_source {
            (rel.prev_rel_source, rel.next_rel_source)
        } else {
            (rel.prev_rel_target, rel.next_rel_target)
        };

        // Update previous
        if prev_id != 0 {
            if let Some(mut prev) = self.rels.read(prev_id) {
                if prev.source_id == node_id {
                    prev.next_rel_source = next_id;
                } else {
                    prev.next_rel_target = next_id;
                }
                self.rels.write(prev_id, &prev)?;
            }
        } else {
            // This was the head - update node
            if let Some(mut node) = self.nodes.read(node_id) {
                node.first_rel_id = next_id;
                self.nodes.write(node_id, &node)?;
            }
        }

        // Update next
        if next_id != 0 {
            if let Some(mut next) = self.rels.read(next_id) {
                if next.source_id == node_id {
                    next.prev_rel_source = prev_id;
                } else {
                    next.prev_rel_target = prev_id;
                }
                self.rels.write(next_id, &next)?;
            }
        }

        Ok(())
    }

    // ─── Query Operations ────────────────────────────────────────────────────

    /// Get all node IDs connected to the given node.
    pub fn get_neighbors(&self, node_id: u32) -> Vec<u32> {
        let mut neighbors = Vec::new();

        let node = match self.nodes.read(node_id) {
            Some(n) if !n.is_deleted() => n,
            _ => return neighbors,
        };

        let mut rel_id = node.first_rel_id;
        while rel_id != 0 {
            if let Some(rel) = self.rels.read(rel_id) {
                if rel.is_deleted() {
                    break;
                }

                // Add the other node
                if rel.source_id == node_id {
                    neighbors.push(rel.target_id);
                    rel_id = rel.next_rel_source;
                } else {
                    neighbors.push(rel.source_id);
                    rel_id = rel.next_rel_target;
                }
            } else {
                break;
            }
        }

        neighbors
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    fn load_free_list(path: &Path) -> FreeList {
        if let Ok(bytes) = std::fs::read(path) {
            FreeList::from_bytes(&bytes).unwrap_or_default()
        } else {
            FreeList::new()
        }
    }

    fn save_free_list(list: &FreeList, path: &Path) -> Result<()> {
        std::fs::write(path, list.to_bytes())?;
        Ok(())
    }
}

impl Drop for SpiderDB {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn create_and_get_node() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let id = db.create_node(&["Person"]).unwrap();
        assert_eq!(id, 1);

        let node = db.get_node(id).unwrap();
        assert_eq!(node.id, 1);
        assert!(node.has_label(1)); // "Person" = 1
    }

    #[test]
    fn create_and_get_rel() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        let rel_id = db.create_rel(a, b, "KNOWS").unwrap();

        let rel = db.get_rel(rel_id).unwrap();
        assert_eq!(rel.source_id, a);
        assert_eq!(rel.target_id, b);
    }

    #[test]
    fn get_neighbors() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        let c = db.create_node(&["Person"]).unwrap();

        db.create_rel(a, b, "KNOWS").unwrap();
        db.create_rel(a, c, "KNOWS").unwrap();

        let neighbors = db.get_neighbors(a);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&b));
        assert!(neighbors.contains(&c));
    }

    #[test]
    fn delete_node() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let id = db.create_node(&["Person"]).unwrap();
        assert!(db.get_node(id).is_some());

        db.delete_node(id).unwrap();
        assert!(db.get_node(id).is_none());
    }

    #[test]
    fn persist_and_reopen() {
        let dir = tempdir().unwrap();

        // Create and close
        {
            let mut db = SpiderDB::open(dir.path()).unwrap();
            db.create_node(&["Person"]).unwrap();
            db.create_node(&["Document"]).unwrap();
            db.close().unwrap();
        }

        // Reopen and verify
        {
            let db = SpiderDB::open(dir.path()).unwrap();
            assert!(db.get_node(1).is_some());
            assert!(db.get_node(2).is_some());
        }
    }

    #[test]
    fn too_many_labels_error() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let result = db.create_node(&["a", "b", "c", "d", "e"]);
        assert!(matches!(result, Err(DbError::TooManyLabels { .. })));
    }

    #[test]
    fn invalid_node_errors() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        // Create rel with non-existent source
        let result = db.create_rel(999, 1, "KNOWS");
        assert!(matches!(result, Err(DbError::SourceNodeNotFound(999))));
    }
}
