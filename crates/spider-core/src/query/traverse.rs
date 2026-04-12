//! Graph traversal queries — neighbor and relationship lookups.
//!
//! Walks the doubly-linked edge chains anchored at a node to find connected
//! neighbors and relationships. All operations are O(degree) — proportional
//! to the number of edges connected to the queried node.
//!
//! ## API
//!
//! - [`get_neighbors()`] — node IDs connected to a given node
//! - [`get_relationships()`] — full edge records for a given node
//! - [`count_relationships()`] — edge count without allocating a Vec

use crate::db::lifecycle::Spider;
use crate::db::nodes::NodeId;
use crate::db::rels::{Direction, EdgeId};
use crate::error::{DbError, SpiderResult};
use crate::schema::edge::Edge;
use crate::schema::node::Node;

/// Maximum chain walk length before assuming corruption.
const MAX_CHAIN_WALK: usize = 10_000_000;

/// A neighbor node reached via a specific edge.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    /// The neighbor node ID.
    pub node_id: NodeId,
    /// The edge connecting the queried node to this neighbor.
    pub edge_id: EdgeId,
}

/// Finds all neighbor node IDs connected to the given node.
///
/// Walks the edge chain and returns the node on the other end of each edge.
/// For directed graphs, `Direction::Outgoing` returns targets,
/// `Direction::Incoming` returns sources, `Direction::Both` returns all.
///
/// # Errors
/// - [`DbError::NodeNotFound`] — if the queried node doesn't exist or is deleted
/// - [`DbError::TraversalDepthExceeded`] — if chain walk exceeds 10M steps
pub fn get_neighbors(
    spider: &mut Spider,
    node_id: NodeId,
    direction: Direction,
) -> SpiderResult<Vec<Neighbor>> {
    let node = spider.nodes.get(node_id.get() - 1)?;
    if node.is_deleted() {
        return Err(DbError::NodeNotFound(node_id.get()));
    }

    walk_edge_chain(spider, &node, node_id.get(), direction)
        .map(|edges| {
            edges
                .into_iter()
                .filter_map(|edge| {
                    let neighbor_id = if edge.source_id == node_id.get() {
                        edge.target_id
                    } else {
                        edge.source_id
                    };
                    // These should always succeed since edge IDs and node IDs
                    // come from valid records.
                    NodeId::new(neighbor_id)
                        .ok()
                        .and_then(|nid| EdgeId::new(edge.id).ok().map(|eid| Neighbor { node_id: nid, edge_id: eid }))
                })
                .collect()
        })
}

/// Returns all edges connected to the given node.
///
/// Equivalent to [`get_neighbors()`] but returns full [`Edge`] records
/// instead of just neighbor IDs.
///
/// # Errors
/// - [`DbError::NodeNotFound`] — if the queried node doesn't exist or is deleted
/// - [`DbError::TraversalDepthExceeded`] — if chain walk exceeds 10M steps
pub fn get_relationships(
    spider: &mut Spider,
    node_id: NodeId,
    direction: Direction,
) -> SpiderResult<Vec<Edge>> {
    let node = spider.nodes.get(node_id.get() - 1)?;
    if node.is_deleted() {
        return Err(DbError::NodeNotFound(node_id.get()));
    }

    walk_edge_chain(spider, &node, node_id.get(), direction)
}

/// Counts edges connected to the given node without allocating a Vec.
///
/// More efficient than [`get_relationships()`] when you only need the count.
///
/// # Errors
/// - [`DbError::NodeNotFound`] — if the queried node doesn't exist or is deleted
/// - [`DbError::TraversalDepthExceeded`] — if chain walk exceeds 10M steps
pub fn count_relationships(
    spider: &mut Spider,
    node_id: NodeId,
    direction: Direction,
) -> SpiderResult<usize> {
    let node = spider.nodes.get(node_id.get() - 1)?;
    if node.is_deleted() {
        return Err(DbError::NodeNotFound(node_id.get()));
    }

    count_edge_chain(spider, &node, node_id.get(), direction)
}

/// Walks the edge chain from a node, returning matching edges.
fn walk_edge_chain(
    spider: &mut Spider,
    node: &Node,
    node_id: u32,
    direction: Direction,
) -> SpiderResult<Vec<Edge>> {
    let mut result = Vec::new();
    let mut cursor = node.first_edge_id;
    let mut steps = 0usize;

    while cursor != 0 {
        steps += 1;
        if steps > MAX_CHAIN_WALK {
            return Err(DbError::TraversalDepthExceeded { limit: MAX_CHAIN_WALK });
        }

        let edge = spider.edges.get(cursor - 1)?;
        if edge.is_deleted() {
            break;
        }

        let next = match direction {
            Direction::Outgoing => {
                if edge.source_id == node_id {
                    edge.next_edge_source
                } else {
                    0
                }
            }
            Direction::Incoming => {
                if edge.target_id == node_id {
                    edge.next_edge_target
                } else {
                    0
                }
            }
            Direction::Both => chain_next(&edge, node_id),
        };

        let include = match direction {
            Direction::Outgoing => edge.source_id == node_id,
            Direction::Incoming => edge.target_id == node_id,
            Direction::Both => true,
        };
        if include {
            result.push(edge);
        }

        cursor = next;
    }

    Ok(result)
}

/// Counts edges in the chain without allocating a Vec.
fn count_edge_chain(
    spider: &mut Spider,
    node: &Node,
    node_id: u32,
    direction: Direction,
) -> SpiderResult<usize> {
    let mut count = 0usize;
    let mut cursor = node.first_edge_id;
    let mut steps = 0usize;

    while cursor != 0 {
        steps += 1;
        if steps > MAX_CHAIN_WALK {
            return Err(DbError::TraversalDepthExceeded { limit: MAX_CHAIN_WALK });
        }

        let edge = spider.edges.get(cursor - 1)?;
        if edge.is_deleted() {
            break;
        }

        let next = match direction {
            Direction::Outgoing => {
                if edge.source_id == node_id {
                    edge.next_edge_source
                } else {
                    0
                }
            }
            Direction::Incoming => {
                if edge.target_id == node_id {
                    edge.next_edge_target
                } else {
                    0
                }
            }
            Direction::Both => chain_next(&edge, node_id),
        };

        let include = match direction {
            Direction::Outgoing => edge.source_id == node_id,
            Direction::Incoming => edge.target_id == node_id,
            Direction::Both => true,
        };
        if include {
            count += 1;
        }

        cursor = next;
    }

    Ok(count)
}

/// Returns the next edge ID in `node_id`'s chain.
#[inline]
fn chain_next(edge: &Edge, node_id: u32) -> u32 {
    if edge.source_id == node_id {
        edge.next_edge_source
    } else {
        edge.next_edge_target
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ingest::{Entity, IngestRequest, Proposition};

    fn setup() -> (tempfile::TempDir, Spider) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_traverse_db");
        let db = Spider::open(&db_path).unwrap();
        (dir, db)
    }

    // --- get_neighbors ---

    #[test]
    fn get_neighbors_empty() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Hello world",
                    entities: vec![],
                },
            ],
        };
        let result = crate::db::ingest::index(&mut db, &req).unwrap();

        // Proposition node has no outgoing edges except MENTIONS (no entities).
        // Document node has 1 outgoing CONTAINS edge.
        let neighbors = get_neighbors(&mut db, result.document_id, Direction::Outgoing).unwrap();
        assert_eq!(neighbors.len(), 1);
    }

    #[test]
    fn get_neighbors_with_entities() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Mumbai is in India",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                        Entity { name: "India", entity_type: "LOCATION" },
                    ],
                },
            ],
        };
        let result = crate::db::ingest::index(&mut db, &req).unwrap();

        // Proposition has 2 MENTIONS edges (outgoing to entities).
        // The proposition node is the one connected to Document via CONTAINS.
        let doc_neighbors = get_neighbors(&mut db, result.document_id, Direction::Outgoing).unwrap();
        assert_eq!(doc_neighbors.len(), 1); // 1 proposition

        // Get the proposition node ID from the neighbor.
        let prop_id = doc_neighbors[0].node_id;

        let prop_neighbors = get_neighbors(&mut db, prop_id, Direction::Outgoing).unwrap();
        assert_eq!(prop_neighbors.len(), 2); // 2 entities

        // Check that both entity neighbors are returned.
        let entity_names: Vec<u32> = prop_neighbors.iter().map(|n| n.node_id.get()).collect();
        assert_eq!(entity_names.len(), 2);
    }

    #[test]
    fn get_neighbors_incoming() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Facts",
                    entities: vec![
                        Entity { name: "X", entity_type: "CONCEPT" },
                    ],
                },
            ],
        };
        let result = crate::db::ingest::index(&mut db, &req).unwrap();

        // Entity node has 1 incoming MENTIONS edge from the proposition.
        let prop_neighbors = get_neighbors(&mut db, result.document_id, Direction::Outgoing).unwrap();
        let prop_id = prop_neighbors[0].node_id;

        let prop_out = get_neighbors(&mut db, prop_id, Direction::Outgoing).unwrap();
        assert_eq!(prop_out.len(), 1); // 1 entity

        let entity_id = prop_out[0].node_id;
        let entity_in = get_neighbors(&mut db, entity_id, Direction::Incoming).unwrap();
        assert_eq!(entity_in.len(), 1); // 1 incoming edge (MENTIONS from proposition)
    }

    #[test]
    fn get_neighbors_both_directions() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Facts",
                    entities: vec![
                        Entity { name: "X", entity_type: "CONCEPT" },
                    ],
                },
            ],
        };
        let result = crate::db::ingest::index(&mut db, &req).unwrap();

        // Proposition has 1 outgoing (MENTIONS to entity).
        let prop_id = get_neighbors(&mut db, result.document_id, Direction::Outgoing)
            .unwrap()[0].node_id;

        let both = get_neighbors(&mut db, prop_id, Direction::Both).unwrap();
        // 1 incoming (CONTAINS from document) + 1 outgoing (MENTIONS to entity)
        assert_eq!(both.len(), 2);
    }

    #[test]
    fn get_neighbors_nonexistent_node() {
        let (_dir, mut db) = setup();
        let result = get_neighbors(&mut db, NodeId::new(999).unwrap(), Direction::Both);
        assert!(result.is_err());
    }

    // --- get_relationships ---

    #[test]
    fn get_relationships_returns_edges() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Hello",
                    entities: vec![
                        Entity { name: "A", entity_type: "X" },
                    ],
                },
            ],
        };
        let result = crate::db::ingest::index(&mut db, &req).unwrap();

        let rels = get_relationships(&mut db, result.document_id, Direction::Outgoing).unwrap();
        assert_eq!(rels.len(), 1);

        // The edge should be a CONTAINS edge.
        let edge = &rels[0];
        assert_eq!(edge.source_id, result.document_id.get());
    }

    // --- count_relationships ---

    #[test]
    fn count_relationship_matches_get_relationships_len() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Hello",
                    entities: vec![
                        Entity { name: "A", entity_type: "X" },
                        Entity { name: "B", entity_type: "Y" },
                    ],
                },
            ],
        };
        let result = crate::db::ingest::index(&mut db, &req).unwrap();

        let prop_id = get_neighbors(&mut db, result.document_id, Direction::Outgoing)
            .unwrap()[0].node_id;

        let edges = get_relationships(&mut db, prop_id, Direction::Outgoing).unwrap();
        let count = count_relationships(&mut db, prop_id, Direction::Outgoing).unwrap();
        assert_eq!(count, edges.len());
    }

    #[test]
    fn count_relationships_zero_for_leaf() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Hello",
                    entities: vec![],
                },
            ],
        };
        let result = crate::db::ingest::index(&mut db, &req).unwrap();

        let prop_id = get_neighbors(&mut db, result.document_id, Direction::Outgoing)
            .unwrap()[0].node_id;

        let count = count_relationships(&mut db, prop_id, Direction::Outgoing).unwrap();
        assert_eq!(count, 0);
    }
}
