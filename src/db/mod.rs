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
pub mod bio;

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

    /// Open or create a database, optionally overriding bio scoring parameters.
    ///
    /// If `None` is passed for any param, the persisted value (or default 1.0) is kept.
    pub fn open_with_bio_params<P: AsRef<Path>>(
        path: P,
        w_sig: Option<f64>,
        w_freq: Option<f64>,
        gravity: Option<f64>,
    ) -> Result<Self> {
        let mut db = Self::open(path)?;
        if let Some(v) = w_sig { db.meta.bio_w_sig = v; }
        if let Some(v) = w_freq { db.meta.bio_w_freq = v; }
        if let Some(v) = gravity { db.meta.bio_gravity = v; }
        Ok(db)
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

        // Get current timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        // Create and write record (with bio metrics initialized)
        let node = NodeRecord::new(id, &label_ids[..labels.len()], now);
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
    /// Cascade deletes: frees all node properties, then all relationships
    /// (and their properties), then marks the node as deleted.
    pub fn delete_node(&mut self, id: u32) -> Result<()> {
        let node = self.nodes.read(id).ok_or(DbError::NodeNotFound(id))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(id));
        }

        // Free node's property chain (props + DynStrings)
        if node.first_prop_id != 0 {
            self.free_all_properties(node.first_prop_id)?;
        }

        // Delete all relationships from this node
        // (delete_rel_internal also frees rel properties)
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

    // ─── Bio Operations ──────────────────────────────────────────────────────

    /// "Touch" a node — strengthens the memory.
    ///
    /// Increments `access_count` and updates `last_accessed_at` to now.
    pub fn touch_node(&mut self, id: u32) -> Result<()> {
        let mut node = self.nodes.read(id).ok_or(DbError::NodeNotFound(id))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(id));
        }

        node.access_count += 1;
        node.last_accessed_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        self.nodes.write(id, &node)?;
        Ok(())
    }

    /// Set the significance (importance) of a node.
    ///
    /// * `significance`: 0-255 (maps to 0.0-1.0 in the bio formula).
    pub fn set_significance(&mut self, id: u32, significance: u8) -> Result<()> {
        let mut node = self.nodes.read(id).ok_or(DbError::NodeNotFound(id))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(id));
        }

        node.significance = significance;
        self.nodes.write(id, &node)?;
        Ok(())
    }

    /// Calculate the current bio score (life force) of a node.
    ///
    /// Uses the database-level BioParams (stored in `meta.db`).
    /// Returns 0.0 if the node doesn't exist or is deleted.
    pub fn get_bio_score(&self, id: u32) -> f64 {
        let node = match self.nodes.read(id) {
            Some(n) if !n.is_deleted() => n,
            _ => return 0.0,
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        let params = bio::BioParams {
            w_sig: self.meta.bio_w_sig,
            w_freq: self.meta.bio_w_freq,
            gravity: self.meta.bio_gravity,
        };

        bio::calculate_bio_score_with_params(
            node.access_count,
            node.significance,
            node.last_accessed_at,
            now,
            &params,
        )
    }

    /// Set the database-level bio scoring parameters.
    ///
    /// These are persisted in `meta.db` and used for all nodes.
    pub fn set_bio_params(&mut self, w_sig: f64, w_freq: f64, gravity: f64) {
        self.meta.bio_w_sig = w_sig;
        self.meta.bio_w_freq = w_freq;
        self.meta.bio_gravity = gravity;
    }

    /// Get the current bio scoring parameters (w_sig, w_freq, gravity).
    pub fn get_bio_params(&self) -> (f64, f64, f64) {
        (self.meta.bio_w_sig, self.meta.bio_w_freq, self.meta.bio_gravity)
    }

    /// Get all live (non-deleted) node IDs.
    pub fn get_all_node_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        for id in 1..self.meta.next_node_id {
            if let Some(node) = self.nodes.read(id) {
                if !node.is_deleted() {
                    ids.push(id);
                }
            }
        }
        ids
    }

    /// Number of live (non-deleted) nodes.
    pub fn node_count(&self) -> usize {
        self.get_all_node_ids().len()
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
    ///
    /// Frees all relationship properties before unlinking and marking deleted.
    fn delete_rel_internal(&mut self, id: u32) -> Result<()> {
        let rel = self.rels.read(id).ok_or(DbError::RelNotFound(id))?;
        if rel.is_deleted() {
            return Err(DbError::RelNotFound(id));
        }

        // Free relationship's property chain (props + DynStrings)
        if rel.first_prop_id != 0 {
            self.free_all_properties(rel.first_prop_id)?;
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

    // ─── Cascade Delete Tests ────────────────────────────────────────────────

    #[test]
    fn cascade_delete_node_frees_properties() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let id = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id, "name", PropertyValue::Int(42)).unwrap();
        db.set_node_property(id, "age", PropertyValue::Int(30)).unwrap();
        db.set_node_property(id, "active", PropertyValue::Bool(true)).unwrap();

        // Delete the node — properties should be freed
        db.delete_node(id).unwrap();

        // Node is gone
        assert!(db.get_node(id).is_none());

        // Create a new node — it should reuse the freed ID
        let id2 = db.create_node(&["Doc"]).unwrap();
        assert_eq!(id2, id, "Freed node ID should be reused");

        // The new node should have NO properties (old ones were freed)
        assert_eq!(db.get_node_property(id2, "name").unwrap(), None);
        assert_eq!(db.get_node_property(id2, "age").unwrap(), None);
    }

    #[test]
    fn cascade_delete_node_frees_dyn_strings() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let id = db.create_node(&["Person"]).unwrap();
        // Dynamic string (>6 bytes, stored in strings.db)
        let long_name = "Alice Wonderland of the Long Name";
        db.set_node_property(id, "name", PropertyValue::String(long_name.into())).unwrap();

        // Delete — should free the DynString chain too
        db.delete_node(id).unwrap();
        assert!(db.get_node(id).is_none());

        // Create new node and set a new DynString — should reuse freed string IDs
        let id2 = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(id2, "title", PropertyValue::String("Reused".into())).unwrap();
        assert_eq!(
            db.get_node_property(id2, "title").unwrap(),
            Some(PropertyValue::String("Reused".into()))
        );
    }

    #[test]
    fn cascade_delete_node_frees_rel_properties() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();

        let r = db.create_rel(a, b, "KNOWS").unwrap();
        db.set_rel_property(r, "since", PropertyValue::Int(2020)).unwrap();
        db.set_rel_property(r, "weight", PropertyValue::Float(0.9)).unwrap();

        // Delete node A — should cascade: free node props, free rel, free rel props
        db.delete_node(a).unwrap();

        // Node A gone
        assert!(db.get_node(a).is_none());
        // Relationship gone
        assert!(db.get_rel(r).is_none());
        // Node B still alive
        assert!(db.get_node(b).is_some());
        // B should have no relationships left
        assert_eq!(db.get_neighbors(b).len(), 0);
    }

    #[test]
    fn cascade_delete_rel_frees_properties() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();

        let r = db.create_rel(a, b, "KNOWS").unwrap();
        db.set_rel_property(r, "since", PropertyValue::Int(2020)).unwrap();
        db.set_rel_property(r, "note", PropertyValue::String("Best friends forever".into())).unwrap();

        // Delete just the relationship (nodes stay)
        db.delete_rel(r).unwrap();

        // Rel is gone
        assert!(db.get_rel(r).is_none());
        // Nodes still exist
        assert!(db.get_node(a).is_some());
        assert!(db.get_node(b).is_some());
        // No more neighbors
        assert_eq!(db.get_neighbors(a).len(), 0);
    }

    #[test]
    fn cascade_delete_id_reuse() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let a = db.create_node(&["Person"]).unwrap();
        db.set_node_property(a, "x", PropertyValue::Int(1)).unwrap();
        let prop_id_before = db.get_node(a).unwrap().first_prop_id;
        assert_ne!(prop_id_before, 0);

        // Delete — should free node + property IDs
        db.delete_node(a).unwrap();

        // Create new node — should reuse node ID
        let b = db.create_node(&["Doc"]).unwrap();
        assert_eq!(b, a, "Node ID should be reused");

        // Set a property — should reuse property record ID
        db.set_node_property(b, "y", PropertyValue::Int(2)).unwrap();
        let prop_id_after = db.get_node(b).unwrap().first_prop_id;
        assert_eq!(prop_id_after, prop_id_before, "Property record ID should be reused");

        // Verify the new property, NOT the old one
        assert_eq!(db.get_node_property(b, "y").unwrap(), Some(PropertyValue::Int(2)));
        assert_eq!(db.get_node_property(b, "x").unwrap(), None);
    }

    // ─── Bio Scoring Tests ───────────────────────────────────────────────────

    #[test]
    fn bio_touch_increments_access() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let id = db.create_node(&["Memory"]).unwrap();
        let node = db.get_node(id).unwrap();
        assert_eq!(node.access_count, 1);
        assert_eq!(node.significance, 128);

        db.touch_node(id).unwrap();
        let node = db.get_node(id).unwrap();
        assert_eq!(node.access_count, 2);

        db.touch_node(id).unwrap();
        db.touch_node(id).unwrap();
        let node = db.get_node(id).unwrap();
        assert_eq!(node.access_count, 4);
    }

    #[test]
    fn bio_significance() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let id = db.create_node(&["Fact"]).unwrap();
        assert_eq!(db.get_node(id).unwrap().significance, 128);

        db.set_significance(id, 255).unwrap();
        assert_eq!(db.get_node(id).unwrap().significance, 255);

        db.set_significance(id, 0).unwrap();
        assert_eq!(db.get_node(id).unwrap().significance, 0);
    }

    #[test]
    fn bio_score_positive_and_increases_with_touch() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let id = db.create_node(&["Memory"]).unwrap();
        let score1 = db.get_bio_score(id);
        assert!(score1 > 0.0, "Fresh node should have positive score");

        // Touch increases frequency → higher score
        db.touch_node(id).unwrap();
        let score2 = db.get_bio_score(id);
        assert!(score2 > score1, "Touching should increase score");

        // Higher significance → higher score
        db.set_significance(id, 255).unwrap();
        let score3 = db.get_bio_score(id);
        assert!(score3 > score2, "Max significance should boost score");
    }

    #[test]
    fn bio_score_deleted_node_is_zero() {
        let dir = tempdir().unwrap();
        let mut db = SpiderDB::open(dir.path()).unwrap();

        let id = db.create_node(&["Temp"]).unwrap();
        assert!(db.get_bio_score(id) > 0.0);

        db.delete_node(id).unwrap();
        assert_eq!(db.get_bio_score(id), 0.0);
    }
}

