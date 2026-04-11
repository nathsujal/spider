//! Query-by-label and query-by-property operations.
//!
//! Provides sequential scan queries for finding nodes by their labels or
//! property values. These are O(n) scans — fine for ingestion-scale graphs.
//! Proper indexes are a future roadmap item.
//!
//! ## API
//!
//! - [`find_by_label()`] — all node IDs with a given label
//! - [`find_by_property()`] — all node IDs with a given property key and value
//! - [`find_one_by_property()`] — first node ID matching a property (short-circuits)

use crate::db::lifecycle::Spider;
use crate::db::nodes::NodeId;
use crate::error::{DbError, SpiderResult};
use crate::schema::node::Node;
use crate::schema::token::TokenId;
use crate::store::record::Record;

/// Finds all live node IDs that have the given label.
///
/// Scans all nodes sequentially and checks their label token IDs against
/// the label name in the label token store.
///
/// # Errors
/// - [`DbError::Io`] — if node records cannot be read
pub fn find_by_label(spider: &mut Spider, label: &str) -> SpiderResult<Vec<NodeId>> {
    let label_tid = match spider.label_tokens.get_id(label) {
        Some(tid) => tid,
        None => return Ok(Vec::new()), // Label never assigned, no nodes match.
    };

    let max_id = spider.metadata.next_node_id;
    let mut results = Vec::new();

    for nid in 1..max_id {
        let node = match spider.nodes.get(nid - 1) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if node.is_deleted() {
            continue;
        }

        if node_has_label(&node, label_tid) {
            results.push(NodeId::new(nid)?);
        }
    }

    Ok(results)
}

/// Finds all live node IDs that have a property with the given key and
/// short string value.
///
/// Scans all nodes and checks the property chain for a block whose key
/// matches `key` and whose inline short string value matches `value`.
/// Dynamic string (long string) properties are NOT followed — only inline
/// short strings (≤6 bytes) are compared.
///
/// # Errors
/// - [`DbError::Io`] — if node or property records cannot be read
/// - [`DbError::TraversalDepthExceeded`] — if a property chain exceeds 10,000 steps
pub fn find_by_property(spider: &mut Spider, key: &str, value: &str) -> SpiderResult<Vec<NodeId>> {
    let key_tid = match spider.prop_key_tokens.get_id(key) {
        Some(tid) => tid,
        None => return Ok(Vec::new()), // Key never set, no nodes match.
    };

    let max_id = spider.metadata.next_node_id;
    let mut results = Vec::new();

    for nid in 1..max_id {
        let node = match spider.nodes.get(nid - 1) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if node.is_deleted() {
            continue;
        }

        if node.first_prop_id != 0 && node_has_property_value(spider, &node, key_tid, value)? {
            results.push(NodeId::new(nid)?);
        }
    }

    Ok(results)
}

/// Finds the first live node ID that has a property with the given key and
/// short string value.
///
/// Short-circuits on the first match — more efficient than [`find_by_property()`]
/// when you only need one result.
///
/// # Errors
/// - [`DbError::Io`] — if node or property records cannot be read
/// - [`DbError::TraversalDepthExceeded`] — if a property chain exceeds 10,000 steps
pub fn find_one_by_property(spider: &mut Spider, key: &str, value: &str) -> SpiderResult<Option<NodeId>> {
    let key_tid = match spider.prop_key_tokens.get_id(key) {
        Some(tid) => tid,
        None => return Ok(None),
    };

    let max_id = spider.metadata.next_node_id;

    for nid in 1..max_id {
        let node = match spider.nodes.get(nid - 1) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if node.is_deleted() {
            continue;
        }

        if node.first_prop_id != 0 && node_has_property_value(spider, &node, key_tid, value)? {
            return Ok(Some(NodeId::new(nid)?));
        }
    }

    Ok(None)
}

/// Checks if a node has a specific label token.
fn node_has_label(node: &Node, label_tid: TokenId) -> bool {
    node.labels()
        .iter()
        .any(|opt| opt.is_some_and(|lid| lid.get() == label_tid.get()))
}

/// Checks if a node has a property with the given key token and string value.
///
/// Only matches inline short strings (≤6 bytes). Dynamic strings are not
/// dereferenced.
fn node_has_property_value(
    spider: &mut Spider,
    node: &Node,
    key_tid: TokenId,
    value: &str,
) -> SpiderResult<bool> {
    let mut cursor = node.first_prop_id;
    let max_steps = 10_000;
    let mut steps = 0;

    while cursor != 0 {
        steps += 1;
        if steps > max_steps {
            return Err(DbError::TraversalDepthExceeded { limit: max_steps });
        }

        let prop = spider.properties.get(cursor - 1)?;
        if prop.is_deleted() {
            break;
        }

        let next = prop.next_prop_id;

        for block in &prop.blocks {
            if block.is_empty() {
                continue;
            }
            if block.key_id().map_or(true, |k| k.get() != key_tid.get()) {
                continue;
            }

            // Match inline short string.
            if let Some(s) = block.as_short_string() {
                if s == value {
                    return Ok(true);
                }
            }
        }

        cursor = next;
    }

    Ok(false)
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ingest::{Entity, IngestRequest, Proposition};
    use crate::db::nodes::NodeId;
    use crate::schema::node::LabelId;
    use crate::schema::node::Node;

    fn setup() -> (tempfile::TempDir, Spider) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_find_db");
        let db = Spider::open(&db_path).unwrap();
        (dir, db)
    }

    fn create_test_node(spider: &mut Spider, labels: &[&str]) -> SpiderResult<NodeId> {
        let label_ids: Vec<LabelId> = labels
            .iter()
            .map(|&name| {
                let tid = spider.label_tokens.get_or_create(name).unwrap();
                LabelId::new(tid.get()).unwrap()
            })
            .collect();

        let node_id = spider.metadata.next_node_id;
        spider.metadata.next_node_id += 1;

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;

        let node = Node::new(node_id, &label_ids, ts, None).unwrap();
        spider.nodes.append(&[node])?;
        NodeId::new(node_id)
    }

    // --- find_by_label ---

    #[test]
    fn find_by_label_empty_database() {
        let (_dir, mut db) = setup();
        let results = find_by_label(&mut db, "DOCUMENT").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn find_by_label_no_matching_nodes() {
        let (_dir, mut db) = setup();
        create_test_node(&mut db, &["PERSON"]).unwrap();
        let results = find_by_label(&mut db, "DOCUMENT").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn find_by_label_single_node() {
        let (_dir, mut db) = setup();
        let id = create_test_node(&mut db, &["DOCUMENT"]).unwrap();
        let results = find_by_label(&mut db, "DOCUMENT").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get(), id.get());
    }

    #[test]
    fn find_by_label_multiple_nodes() {
        let (_dir, mut db) = setup();
        let id1 = create_test_node(&mut db, &["PERSON"]).unwrap();
        create_test_node(&mut db, &["DOCUMENT"]).unwrap();
        let id3 = create_test_node(&mut db, &["PERSON"]).unwrap();

        let results = find_by_label(&mut db, "PERSON").unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].get(), id1.get());
        assert_eq!(results[1].get(), id3.get());
    }

    #[test]
    fn find_by_label_skips_deleted_nodes() {
        let (_dir, mut db) = setup();
        let id1 = create_test_node(&mut db, &["PERSON"]).unwrap();
        let id2 = create_test_node(&mut db, &["PERSON"]).unwrap();

        // Mark node id2 as deleted by writing a tombstone.
        let idx = id2.get() - 1;
        let tombstone = Node::default(); // id=0, all zeros
        db.nodes.set(idx, &tombstone).unwrap();

        let results = find_by_label(&mut db, "PERSON").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get(), id1.get());
    }

    // --- find_by_property (via ingestion) ---

    #[test]
    fn find_by_property_finds_by_name() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test Doc",
            propositions: vec![
                Proposition {
                    text: "Mumbai is a city",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                        Entity { name: "Paris", entity_type: "LOCATION" },
                    ],
                },
            ],
        };
        crate::db::ingest::index(&mut db, &req).unwrap();

        let results = find_by_property(&mut db, "name", "Mumbai").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn find_by_property_no_match() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test Doc",
            propositions: vec![
                Proposition {
                    text: "Hello world",
                    entities: vec![],
                },
            ],
        };
        crate::db::ingest::index(&mut db, &req).unwrap();

        let results = find_by_property(&mut db, "name", "nonexistent").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn find_by_property_unknown_key() {
        let (_dir, mut db) = setup();
        let results = find_by_property(&mut db, "unknown_key", "value").unwrap();
        assert!(results.is_empty());
    }

    // --- find_one_by_property ---

    #[test]
    fn find_one_by_property_returns_first_match() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test Doc",
            propositions: vec![
                Proposition {
                    text: "Facts",
                    entities: vec![
                        Entity { name: "Alice", entity_type: "PERSON" },
                        Entity { name: "Bob", entity_type: "PERSON" },
                    ],
                },
            ],
        };
        crate::db::ingest::index(&mut db, &req).unwrap();

        let result = find_one_by_property(&mut db, "name", "Alice").unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn find_one_by_property_returns_none() {
        let (_dir, mut db) = setup();
        let result = find_one_by_property(&mut db, "name", "nobody").unwrap();
        assert!(result.is_none());
    }
}
