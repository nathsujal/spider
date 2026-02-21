//! # Spider
//!
//! Main database struct combining all storage components.
//!
//! ## Example
//!
//! ```rust,ignore
//! use spider::db::Spider;
//!
//! // Open or create database
//! let mut db = Spider::open("./data/mydb")?;
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
mod lifecycle;
mod node;
mod rel;
pub mod property;
pub mod query;
pub mod bio;
pub mod content;

pub use error::{DbError, Result};
pub use property::PropertyValue;
pub use query::{Direction, RelInfo};

use std::path::PathBuf;

use crate::schema::{
    DynamicArrayRecord, DynamicStringRecord, NodeRecord, PropertyRecord, RelRecord, TokenStore,
};
use crate::store::{FreeList, Metadata, RecordFile};
use content::ContentStore;

/// Main Spider graph database.
pub struct Spider {
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

    // Token Stores (String ↔ u8 ID)
    pub labels: TokenStore,
    pub rel_types: TokenStore,
    pub(crate) prop_keys: TokenStore,

    // Content-addressed blob store
    content: ContentStore,
}


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn create_and_get_node() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let id = db.create_node(&["Person"]).unwrap();
        assert_eq!(id, 1);

        let node = db.get_node(id).unwrap();
        assert_eq!(node.id, 1);
        assert!(node.has_label(1)); // "Person" = 1
    }

    #[test]
    fn create_and_get_rel() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

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
        let mut db = Spider::open(dir.path()).unwrap();

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
        let mut db = Spider::open(dir.path()).unwrap();

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
            let mut db = Spider::open(dir.path()).unwrap();
            db.create_node(&["Person"]).unwrap();
            db.create_node(&["Document"]).unwrap();
            db.close().unwrap();
        }

        // Reopen and verify
        {
            let db = Spider::open(dir.path()).unwrap();
            assert!(db.get_node(1).is_some());
            assert!(db.get_node(2).is_some());
        }
    }

    #[test]
    fn too_many_labels_error() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let result = db.create_node(&["a", "b", "c", "d", "e"]);
        assert!(matches!(result, Err(DbError::TooManyLabels { .. })));
    }

    #[test]
    fn invalid_node_errors() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        // Create rel with non-existent source
        let result = db.create_rel(999, 1, "KNOWS");
        assert!(matches!(result, Err(DbError::SourceNodeNotFound(999))));
    }


    #[test]
    fn cascade_delete_node_frees_properties() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

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
        let mut db = Spider::open(dir.path()).unwrap();

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
        let mut db = Spider::open(dir.path()).unwrap();

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
        let mut db = Spider::open(dir.path()).unwrap();

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
        let mut db = Spider::open(dir.path()).unwrap();

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


    #[test]
    fn bio_touch_increments_access() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

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
        let mut db = Spider::open(dir.path()).unwrap();

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
        let mut db = Spider::open(dir.path()).unwrap();

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
        let mut db = Spider::open(dir.path()).unwrap();

        let id = db.create_node(&["Temp"]).unwrap();
        assert!(db.get_bio_score(id) > 0.0);

        db.delete_node(id).unwrap();
        assert_eq!(db.get_bio_score(id), 0.0);
    }
}

