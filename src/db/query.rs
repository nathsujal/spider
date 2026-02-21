//! Query & retrieval operations for Spider.
//!
//! Retrieval:
//! - `get_all_node_properties` / `get_all_rel_properties` — bulk property retrieval
//! - `get_relationships` — direction + type filtered traversal
//! - `find_nodes_by_label` — label scan
//! - `find_nodes_by_property` — property scan

use std::collections::HashMap;

use crate::db::{DbError, Result, Spider};
use crate::db::property::PropertyValue;

// Direction enum

/// Relationship traversal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Relationships where the node is the source.
    Outgoing,
    /// Relationships where the node is the target.
    Incoming,
    /// All relationships regardless of direction.
    Both,
}

// RelInfo — rich relationship descriptor

/// A relationship with its metadata resolved to human-readable form.
#[derive(Debug, Clone, PartialEq)]
pub struct RelInfo {
    pub rel_id: u32,
    pub source_id: u32,
    pub target_id: u32,
    pub rel_type: String,
}

// Spider Query Methods

impl Spider {
    // Property Collection

    /// Get all properties of a node as a key-value map.
    ///
    /// Walks the entire property chain once and resolves key IDs to names.
    /// Returns an empty map if the node has no properties.
    pub fn get_all_node_properties(
        &self,
        node_id: u32,
    ) -> Result<HashMap<String, PropertyValue>> {
        let node = self
            .nodes
            .read(node_id)
            .ok_or(DbError::NodeNotFound(node_id))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(node_id));
        }
        self.collect_all_properties(node.first_prop_id)
    }

    /// Get all properties of a relationship as a key-value map.
    pub fn get_all_rel_properties(
        &self,
        rel_id: u32,
    ) -> Result<HashMap<String, PropertyValue>> {
        let rel = self
            .rels
            .read(rel_id)
            .ok_or(DbError::RelNotFound(rel_id))?;
        if rel.is_deleted() {
            return Err(DbError::RelNotFound(rel_id));
        }
        self.collect_all_properties(rel.first_prop_id)
    }

    /// Internal: walk a property chain and return all key-value pairs.
    fn collect_all_properties(
        &self,
        first_prop_id: u32,
    ) -> Result<HashMap<String, PropertyValue>> {
        let mut map = HashMap::new();
        let mut prop_id = first_prop_id;

        while prop_id != 0 {
            let record = self.props.read(prop_id).ok_or(DbError::Corrupted(
                format!("Property record {} missing", prop_id),
            ))?;

            for block in &record.blocks {
                if !block.is_empty() {
                    // Resolve key_id -> key name
                    if let Some(key_name) = self.prop_keys.get_name(block.key_id()) {
                        if let Some(value) = self.decode_block(block) {
                            map.insert(key_name.to_string(), value);
                        }
                    }
                }
            }

            prop_id = record.next_prop_id;
        }

        Ok(map)
    }

    // Relationship Traversal

    /// Get relationships for a node, filtered by direction and optional type.
    ///
    /// # Arguments
    /// - `node_id` — the node to query
    /// - `direction` — Outgoing, Incoming, or Both
    /// - `rel_type` — optional type name to filter by (e.g., "KNOWS")
    ///
    /// Returns a list of `RelInfo` structs with resolved type names.
    pub fn get_relationships(
        &self,
        node_id: u32,
        direction: Direction,
        rel_type: Option<&str>,
    ) -> Result<Vec<RelInfo>> {
        let node = self
            .nodes
            .read(node_id)
            .ok_or(DbError::NodeNotFound(node_id))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(node_id));
        }

        // Resolve type filter to ID (if provided)
        let type_id_filter: Option<u8> = rel_type.and_then(|t| self.rel_types.get_id(t));

        // If a type filter was given but the type doesn't exist, no results
        if rel_type.is_some() && type_id_filter.is_none() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();
        let mut rel_id = node.first_rel_id;

        while rel_id != 0 {
            let rel = match self.rels.read(rel_id) {
                Some(r) if !r.is_deleted() => r,
                _ => break,
            };

            let is_source = rel.source_id == node_id;
            let is_target = rel.target_id == node_id;

            // Direction filter
            let include = match direction {
                Direction::Outgoing => is_source,
                Direction::Incoming => is_target && !is_source,
                Direction::Both => true,
            };

            // Type filter
            let type_ok = match type_id_filter {
                Some(tid) => rel.rel_type_id == tid,
                None => true,
            };

            if include && type_ok {
                let type_name = self
                    .rel_types
                    .get_name(rel.rel_type_id)
                    .unwrap_or("UNKNOWN")
                    .to_string();

                results.push(RelInfo {
                    rel_id: rel.id,
                    source_id: rel.source_id,
                    target_id: rel.target_id,
                    rel_type: type_name,
                });
            }

            // Advance along the correct chain
            rel_id = if is_source {
                rel.next_rel_source
            } else {
                rel.next_rel_target
            };
        }

        Ok(results)
    }

    /// Get the type name of a relationship.
    pub fn get_rel_type_name(&self, rel_id: u32) -> Result<Option<String>> {
        let rel = self
            .rels
            .read(rel_id)
            .ok_or(DbError::RelNotFound(rel_id))?;
        if rel.is_deleted() {
            return Err(DbError::RelNotFound(rel_id));
        }
        Ok(self
            .rel_types
            .get_name(rel.rel_type_id)
            .map(|s| s.to_string()))
    }

    // Label Scan

    /// Find all node IDs that have the given label.
    ///
    /// This is a full scan of `nodes.db` — O(n).
    /// For production use, a label index would replace this.
    pub fn find_nodes_by_label(&self, label: &str) -> Result<Vec<u32>> {
        let label_id = match self.labels.get_id(label) {
            Some(id) => id,
            None => return Ok(Vec::new()),
        };

        let mut results = Vec::new();
        for id in 1..self.meta.next_node_id {
            if let Some(node) = self.nodes.read(id) {
                if !node.is_deleted() && node.has_label(label_id) {
                    results.push(id);
                }
            }
        }
        Ok(results)
    }

    /// Find all node IDs where a given property key equals a value.
    ///
    /// Full scan — O(n * k). For production use, a hash index would replace this.
    pub fn find_nodes_by_property(
        &self,
        key: &str,
        value: &PropertyValue,
    ) -> Result<Vec<u32>> {
        let key_id = match self.prop_keys.get_id(key) {
            Some(id) => id,
            None => return Ok(Vec::new()),
        };

        let mut results = Vec::new();
        for id in 1..self.meta.next_node_id {
            if let Some(node) = self.nodes.read(id) {
                if !node.is_deleted() && node.first_prop_id != 0 {
                    // Check if this node has the property with matching value
                    if let Ok(Some(v)) = self.find_property(node.first_prop_id, key_id) {
                        if v == *value {
                            results.push(id);
                        }
                    }
                }
            }
        }
        Ok(results)
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // Property Collection

    #[test]
    fn get_all_props_empty_node() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();
        let props = db.get_all_node_properties(id).unwrap();
        assert!(props.is_empty());
    }

    #[test]
    fn get_all_props_mixed() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();

        db.set_node_property(id, "name", PropertyValue::String("Alice".into())).unwrap();
        db.set_node_property(id, "age", PropertyValue::Int(30)).unwrap();
        db.set_node_property(id, "active", PropertyValue::Bool(true)).unwrap();

        let props = db.get_all_node_properties(id).unwrap();
        assert_eq!(props.len(), 3);
        assert_eq!(props["name"], PropertyValue::String("Alice".into()));
        assert_eq!(props["age"], PropertyValue::Int(30));
        assert_eq!(props["active"], PropertyValue::Bool(true));
    }

    #[test]
    fn get_all_props_with_dynamic_string() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Doc"]).unwrap();

        db.set_node_property(id, "title", PropertyValue::String("A Long Title Here".into())).unwrap();
        db.set_node_property(id, "tag", PropertyValue::String("hi".into())).unwrap();

        let props = db.get_all_node_properties(id).unwrap();
        assert_eq!(props["title"], PropertyValue::String("A Long Title Here".into()));
        assert_eq!(props["tag"], PropertyValue::String("hi".into()));
    }

    #[test]
    fn get_all_rel_props() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        let r = db.create_rel(a, b, "KNOWS").unwrap();

        db.set_rel_property(r, "since", PropertyValue::Int(2020)).unwrap();
        db.set_rel_property(r, "weight", PropertyValue::Float(0.9)).unwrap();

        let props = db.get_all_rel_properties(r).unwrap();
        assert_eq!(props.len(), 2);
        assert_eq!(props["since"], PropertyValue::Int(2020));
        assert_eq!(props["weight"], PropertyValue::Float(0.9));
    }

    // Relationship Traversal

    #[test]
    fn get_rels_outgoing() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        let c = db.create_node(&["Person"]).unwrap();
        db.create_rel(a, b, "KNOWS").unwrap();
        db.create_rel(a, c, "LIKES").unwrap();
        db.create_rel(c, a, "FOLLOWS").unwrap(); // incoming to a

        let out = db.get_relationships(a, Direction::Outgoing, None).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|r| r.source_id == a));
    }

    #[test]
    fn get_rels_incoming() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        db.create_rel(b, a, "FOLLOWS").unwrap();
        db.create_rel(a, b, "KNOWS").unwrap();

        let inc = db.get_relationships(a, Direction::Incoming, None).unwrap();
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].target_id, a);
        assert_eq!(inc[0].rel_type, "FOLLOWS");
    }

    #[test]
    fn get_rels_both() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        db.create_rel(a, b, "KNOWS").unwrap();
        db.create_rel(b, a, "FOLLOWS").unwrap();

        let all = db.get_relationships(a, Direction::Both, None).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn get_rels_type_filter() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        let c = db.create_node(&["Person"]).unwrap();
        db.create_rel(a, b, "KNOWS").unwrap();
        db.create_rel(a, c, "LIKES").unwrap();

        let knows = db.get_relationships(a, Direction::Both, Some("KNOWS")).unwrap();
        assert_eq!(knows.len(), 1);
        assert_eq!(knows[0].target_id, b);
        assert_eq!(knows[0].rel_type, "KNOWS");
    }

    #[test]
    fn get_rels_nonexistent_type() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        db.create_rel(a, b, "KNOWS").unwrap();

        let none = db.get_relationships(a, Direction::Both, Some("HATES")).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn rel_type_name() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        let r = db.create_rel(a, b, "KNOWS").unwrap();

        assert_eq!(db.get_rel_type_name(r).unwrap(), Some("KNOWS".to_string()));
    }

    // Label Scan

    #[test]
    fn find_by_label() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let p1 = db.create_node(&["Person"]).unwrap();
        let p2 = db.create_node(&["Person"]).unwrap();
        let _d = db.create_node(&["Document"]).unwrap();

        let people = db.find_nodes_by_label("Person").unwrap();
        assert_eq!(people.len(), 2);
        assert!(people.contains(&p1));
        assert!(people.contains(&p2));
    }

    #[test]
    fn find_by_label_empty() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        db.create_node(&["Person"]).unwrap();

        let found = db.find_nodes_by_label("NonExistent").unwrap();
        assert!(found.is_empty());
    }

    // Property Scan

    #[test]
    fn find_by_property() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        let c = db.create_node(&["Person"]).unwrap();

        db.set_node_property(a, "name", PropertyValue::String("Alice".into())).unwrap();
        db.set_node_property(b, "name", PropertyValue::String("Bob".into())).unwrap();
        db.set_node_property(c, "name", PropertyValue::String("Alice".into())).unwrap();

        let alices = db.find_nodes_by_property("name", &PropertyValue::String("Alice".into())).unwrap();
        assert_eq!(alices.len(), 2);
        assert!(alices.contains(&a));
        assert!(alices.contains(&c));
    }

    #[test]
    fn find_by_property_no_match() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let a = db.create_node(&["Person"]).unwrap();
        db.set_node_property(a, "age", PropertyValue::Int(30)).unwrap();

        let found = db.find_nodes_by_property("age", &PropertyValue::Int(99)).unwrap();
        assert!(found.is_empty());
    }

    // Complex Traversal

    #[test]
    fn social_graph_traversal() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        // Build: Alice -KNOWS-> Bob -KNOWS-> Carol
        //        Alice -LIKES-> Carol
        let alice = db.create_node(&["Person"]).unwrap();
        let bob = db.create_node(&["Person"]).unwrap();
        let carol = db.create_node(&["Person"]).unwrap();

        db.set_node_property(alice, "name", PropertyValue::String("Alice".into())).unwrap();
        db.set_node_property(bob, "name", PropertyValue::String("Bob".into())).unwrap();
        db.set_node_property(carol, "name", PropertyValue::String("Carol".into())).unwrap();

        db.create_rel(alice, bob, "KNOWS").unwrap();
        db.create_rel(bob, carol, "KNOWS").unwrap();
        db.create_rel(alice, carol, "LIKES").unwrap();

        // Alice's outgoing KNOWS → only Bob
        let alice_knows = db.get_relationships(alice, Direction::Outgoing, Some("KNOWS")).unwrap();
        assert_eq!(alice_knows.len(), 1);
        assert_eq!(alice_knows[0].target_id, bob);

        // Alice's all outgoing → Bob (KNOWS) + Carol (LIKES)
        let alice_out = db.get_relationships(alice, Direction::Outgoing, None).unwrap();
        assert_eq!(alice_out.len(), 2);

        // Bob's incoming → Alice
        let bob_in = db.get_relationships(bob, Direction::Incoming, None).unwrap();
        assert_eq!(bob_in.len(), 1);
        assert_eq!(bob_in[0].source_id, alice);

        // Carol's incoming → Bob (KNOWS) + Alice (LIKES)
        let carol_in = db.get_relationships(carol, Direction::Incoming, None).unwrap();
        assert_eq!(carol_in.len(), 2);

        // Get all of Alice's properties
        let alice_props = db.get_all_node_properties(alice).unwrap();
        assert_eq!(alice_props["name"], PropertyValue::String("Alice".into()));

        // Find all "Person" nodes
        let people = db.find_nodes_by_label("Person").unwrap();
        assert_eq!(people.len(), 3);

        // Find node named "Bob"
        let bobs = db.find_nodes_by_property("name", &PropertyValue::String("Bob".into())).unwrap();
        assert_eq!(bobs, vec![bob]);
    }
}
