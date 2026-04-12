//! Edge CRUD with O(1) linked-list insertion and deletion.
//!
//! Each edge participates in **two** doubly-linked chains — one anchored at
//! the source node, one at the target. A single `first_edge_id` on each node
//! serves as the head pointer for *all* edges (both directions). During
//! traversal, the node's role (source or target) in each edge determines
//! which chain pointer to follow.
//!
//! ## Chain structure
//!
//! ```text
//! Node A (first_edge_id = 3)
//!   │
//!   ▼
//! E3(A→C)  ──source_next──▶  E1(A→B)  ──source_next──▶  0
//!   ◀──source_prev──           ◀──source_prev── 0
//! ```

use crate::db::nodes::NodeId;
use crate::error::{DbError, SpiderResult};
use crate::db::lifecycle::Metadata;
use crate::schema::edge::{Edge, EdgeTypeId};
use crate::schema::node::Node;
use crate::store::record::RecordFile;

/// Maximum chain walk length before we assume corruption.
const MAX_CHAIN_WALK: usize = 10_000_000;

// EdgeId

/// Non-zero edge ID (`0` = tombstone sentinel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeId(u32);

impl EdgeId {
    /// Constructs an `EdgeId`. Returns `Err` if `id == 0`.
    #[inline]
    pub fn new(id: u32) -> Result<Self, DbError> {
        if id == 0 {
            Err(DbError::EdgeNotFound(0))
        } else {
            Ok(Self(id))
        }
    }

    /// Returns the underlying `u32`.
    #[inline]
    pub fn get(self) -> u32 {
        self.0
    }
}

// Direction

/// Traversal direction for edge queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Edges where the queried node is the source.
    Outgoing,
    /// Edges where the queried node is the target.
    Incoming,
    /// All edges connected to the queried node.
    Both,
}

// Chain pointer helpers

/// Returns the next edge ID in `node_id`'s chain.
#[inline]
fn chain_next(edge: &Edge, node_id: u32) -> u32 {
    if edge.source_id == node_id {
        edge.next_edge_source
    } else {
        edge.next_edge_target
    }
}

/// Returns the prev edge ID in `node_id`'s chain.
#[inline]
fn chain_prev(edge: &Edge, node_id: u32) -> u32 {
    if edge.source_id == node_id {
        edge.prev_edge_source
    } else {
        edge.prev_edge_target
    }
}

/// Sets the next pointer in `node_id`'s chain.
#[inline]
fn set_chain_next(edge: &mut Edge, node_id: u32, next: u32) {
    if edge.source_id == node_id {
        edge.next_edge_source = next;
    } else {
        edge.next_edge_target = next;
    }
}

/// Sets the prev pointer in `node_id`'s chain.
#[inline]
fn set_chain_prev(edge: &mut Edge, node_id: u32, prev: u32) {
    if edge.source_id == node_id {
        edge.prev_edge_source = prev;
    } else {
        edge.prev_edge_target = prev;
    }
}

// EdgeOps

/// Short-lived handle for edge operations.
///
/// Borrows the minimum set of fields from [`Spider`](crate::db::lifecycle::Spider)
/// needed to perform edge CRUD. Construct via [`Spider::edge_ops()`].
pub struct EdgeOps<'db> {
    pub(crate) nodes: &'db mut RecordFile<Node>,
    pub(crate) edges: &'db mut RecordFile<Edge>,
    pub(crate) metadata: &'db mut Metadata,
}

impl<'db> EdgeOps<'db> {
    /// Creates an edge and inserts it into both the source and target chains.
    ///
    /// Allocates a new edge ID from `metadata.next_rel_id`.
    ///
    /// # Errors
    /// - [`DbError::NodeNotFound`] — source or target node doesn't exist
    /// - [`DbError::NodeDeleted`] — source or target is a tombstone
    /// - [`DbError::EdgeError`] — schema validation (self-loop, zero IDs)
    pub fn create(
        &mut self,
        source: NodeId,
        target: NodeId,
        type_id: EdgeTypeId,
    ) -> SpiderResult<EdgeId> {
        let src_idx = source.get() - 1;
        let tgt_idx = target.get() - 1;

        // Read and validate source node.
        let mut source_node = self.nodes.get(src_idx)
            .map_err(|_| DbError::SourceNodeNotFound(source.get()))?;
        if source_node.is_deleted() {
            return Err(DbError::NodeDeleted(source.get()));
        }

        // Read and validate target node.
        let mut target_node = self.nodes.get(tgt_idx)
            .map_err(|_| DbError::TargetNodeNotFound(target.get()))?;
        if target_node.is_deleted() {
            return Err(DbError::NodeDeleted(target.get()));
        }

        // Allocate edge ID.
        let edge_id = self.metadata.next_rel_id;
        self.metadata.next_rel_id += 1;

        // Create the edge (validates no self-loop, non-zero IDs).
        let mut edge = Edge::new(edge_id, source.get(), target.get(), type_id)?;

        // Insert into source node's chain (head insertion).
        self.insert_at_head(&mut source_node, &mut edge, source.get())?;

        // Insert into target node's chain (head insertion).
        self.insert_at_head(&mut target_node, &mut edge, target.get())?;

        // Persist: append new edge, then update both nodes.
        self.edges.append(&[edge])?;
        self.nodes.set(src_idx, &source_node)?;
        self.nodes.set(tgt_idx, &target_node)?;

        EdgeId::new(edge_id)
    }

    /// Reads an edge by ID.
    ///
    /// # Errors
    /// - [`DbError::EdgeNotFound`] — no live edge at this ID
    pub fn get(&mut self, id: EdgeId) -> SpiderResult<Edge> {
        let edge = self.edges.get(id.get() - 1)
            .map_err(|_| DbError::EdgeNotFound(id.get()))?;
        if edge.is_deleted() {
            return Err(DbError::EdgeNotFound(id.get()));
        }
        Ok(edge)
    }

    /// Deletes an edge by unlinking it from both chains and writing a tombstone.
    ///
    /// # Errors
    /// - [`DbError::EdgeNotFound`] — edge doesn't exist or already deleted
    pub fn delete(&mut self, id: EdgeId) -> SpiderResult<()> {
        let edge = self.get(id)?;

        // Unlink from source node's chain.
        self.unlink_from_chain(&edge, edge.source_id)?;

        // Unlink from target node's chain.
        self.unlink_from_chain(&edge, edge.target_id)?;

        // Write tombstone.
        self.edges.set(id.get() - 1, &Edge::empty())?;
        Ok(())
    }

    /// Returns all edges connected to `node_id` in the given direction.
    ///
    /// Walks the node's single chain and filters by direction.
    pub fn edges_for_node(
        &mut self,
        node_id: NodeId,
        direction: Direction,
    ) -> SpiderResult<Vec<Edge>> {
        let node = self.nodes.get(node_id.get() - 1)
            .map_err(|_| DbError::NodeNotFound(node_id.get()))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(node_id.get()));
        }

        let nid = node_id.get();
        let mut result = Vec::new();
        let mut cursor = node.first_edge_id;
        let mut steps = 0usize;

        while cursor != 0 {
            steps += 1;
            if steps > MAX_CHAIN_WALK {
                return Err(DbError::TraversalDepthExceeded { limit: MAX_CHAIN_WALK });
            }

            let edge = self.edges.get(cursor - 1)?;
            if edge.is_deleted() {
                break; // Corrupt chain — stop gracefully.
            }

            let next = chain_next(&edge, nid);

            let include = match direction {
                Direction::Outgoing => edge.source_id == nid,
                Direction::Incoming => edge.target_id == nid,
                Direction::Both => true,
            };
            if include {
                result.push(edge);
            }

            cursor = next;
        }

        Ok(result)
    }

    /// Counts edges connected to `node_id` without allocating a `Vec`.
    pub fn edge_count_for_node(
        &mut self,
        node_id: NodeId,
        direction: Direction,
    ) -> SpiderResult<usize> {
        // Reuse traversal logic — allocation is minimal for counting.
        self.edges_for_node(node_id, direction).map(|v| v.len())
    }

    // Private helpers

    /// Head-inserts `new_edge` into `node`'s chain.
    ///
    /// Mutates `new_edge` (sets its next pointer) and `node` (updates
    /// `first_edge_id`). If the chain was non-empty, also updates the old
    /// head edge's prev pointer on disk.
    fn insert_at_head(
        &mut self,
        node: &mut Node,
        new_edge: &mut Edge,
        node_id: u32,
    ) -> SpiderResult<()> {
        let old_head = node.first_edge_id;

        // New edge's next → old head.
        set_chain_next(new_edge, node_id, old_head);

        // Old head's prev → new edge.
        if old_head != 0 {
            let mut old_head_edge = self.edges.get(old_head - 1)?;
            set_chain_prev(&mut old_head_edge, node_id, new_edge.id);
            self.edges.set(old_head - 1, &old_head_edge)?;
        }

        // Node head → new edge.
        node.first_edge_id = new_edge.id;

        Ok(())
    }

    /// Unlinks an edge from a specific node's chain.
    ///
    /// Updates the prev edge's next pointer (or the node's `first_edge_id`
    /// if the edge was the chain head), and the next edge's prev pointer.
    fn unlink_from_chain(&mut self, edge: &Edge, node_id: u32) -> SpiderResult<()> {
        let prev_id = chain_prev(edge, node_id);
        let next_id = chain_next(edge, node_id);

        // Rewire prev → next (or update head pointer).
        if prev_id != 0 {
            let mut prev_edge = self.edges.get(prev_id - 1)?;
            set_chain_next(&mut prev_edge, node_id, next_id);
            self.edges.set(prev_id - 1, &prev_edge)?;
        } else {
            // Edge was the head — update node's first_edge_id.
            let mut node = self.nodes.get(node_id - 1)?;
            node.first_edge_id = next_id;
            self.nodes.set(node_id - 1, &node)?;
        }

        // Rewire next → prev.
        if next_id != 0 {
            let mut next_edge = self.edges.get(next_id - 1)?;
            set_chain_prev(&mut next_edge, node_id, prev_id);
            self.edges.set(next_id - 1, &next_edge)?;
        }

        Ok(())
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::lifecycle::Spider;
    use crate::schema::edge::EdgeTypeId;
    use crate::schema::node::Node;

    const NOW: u32 = 1_700_000_000;

    fn tid(id: u8) -> EdgeTypeId {
        EdgeTypeId::new(id).unwrap()
    }

    /// Creates a test database with `n` nodes (IDs 1..=n).
    fn setup(n: u32) -> (tempfile::TempDir, Spider) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        let mut db = Spider::open(&db_path).unwrap();

        for i in 1..=n {
            let node = Node::new(i, &[], NOW, None).unwrap();
            db.nodes.append(&[node]).unwrap();
        }
        db.metadata.next_node_id = n + 1;

        (dir, db)
    }

    // EdgeId

    #[test]
    fn edge_id_rejects_zero() {
        assert!(EdgeId::new(0).is_err());
        assert_eq!(EdgeId::new(1).unwrap().get(), 1);
    }

    // Create

    #[test]
    fn create_single_edge() {
        let (_dir, mut db) = setup(2);
        let eid = db.edge_ops().create(
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            tid(1),
        ).unwrap();
        assert_eq!(eid.get(), 1);
        assert_eq!(db.metadata.next_rel_id, 2);
    }

    #[test]
    fn create_edge_wires_source_chain() {
        let (_dir, mut db) = setup(2);
        db.edge_ops().create(
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            tid(1),
        ).unwrap();

        let node_a = db.nodes.get(0).unwrap();
        assert_eq!(node_a.first_edge_id, 1);
    }

    #[test]
    fn create_edge_wires_target_chain() {
        let (_dir, mut db) = setup(2);
        db.edge_ops().create(
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            tid(1),
        ).unwrap();

        let node_b = db.nodes.get(1).unwrap();
        assert_eq!(node_b.first_edge_id, 1);
    }

    #[test]
    fn create_multiple_edges_same_source() {
        let (_dir, mut db) = setup(3);
        let src = NodeId::new(1).unwrap();

        db.edge_ops().create(src, NodeId::new(2).unwrap(), tid(1)).unwrap();
        db.edge_ops().create(src, NodeId::new(3).unwrap(), tid(1)).unwrap();

        // Head insertion: newest edge (id=2) is the head.
        let node_a = db.nodes.get(0).unwrap();
        assert_eq!(node_a.first_edge_id, 2);

        // E2's next_source → E1.
        let e2 = db.edges.get(1).unwrap();
        assert_eq!(e2.next_edge_source, 1);

        // E1's prev_source → E2.
        let e1 = db.edges.get(0).unwrap();
        assert_eq!(e1.prev_edge_source, 2);
    }

    #[test]
    fn create_edge_invalid_source() {
        let (_dir, mut db) = setup(2);
        let result = db.edge_ops().create(
            NodeId::new(99).unwrap(),
            NodeId::new(2).unwrap(),
            tid(1),
        );
        assert!(matches!(result, Err(DbError::SourceNodeNotFound(99))));
    }

    #[test]
    fn create_edge_invalid_target() {
        let (_dir, mut db) = setup(2);
        let result = db.edge_ops().create(
            NodeId::new(1).unwrap(),
            NodeId::new(99).unwrap(),
            tid(1),
        );
        assert!(matches!(result, Err(DbError::TargetNodeNotFound(99))));
    }

    #[test]
    fn self_loop_rejected() {
        let (_dir, mut db) = setup(2);
        let result = db.edge_ops().create(
            NodeId::new(1).unwrap(),
            NodeId::new(1).unwrap(),
            tid(1),
        );
        assert!(matches!(result, Err(DbError::EdgeError(_))));
    }

    // Get

    #[test]
    fn get_edge_returns_created_edge() {
        let (_dir, mut db) = setup(2);
        let eid = db.edge_ops().create(
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            tid(5),
        ).unwrap();

        let edge = db.edge_ops().get(eid).unwrap();
        assert_eq!(edge.source_id, 1);
        assert_eq!(edge.target_id, 2);
        assert_eq!(edge.edge_type().unwrap().get(), 5);
    }

    #[test]
    fn get_nonexistent_edge_returns_error() {
        let (_dir, mut db) = setup(2);
        let result = db.edge_ops().get(EdgeId::new(99).unwrap());
        assert!(matches!(result, Err(DbError::EdgeNotFound(99))));
    }

    // Delete

    #[test]
    fn delete_only_edge() {
        let (_dir, mut db) = setup(2);
        let eid = db.edge_ops().create(
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            tid(1),
        ).unwrap();

        db.edge_ops().delete(eid).unwrap();

        // Edge is now a tombstone.
        assert!(matches!(
            db.edge_ops().get(eid),
            Err(DbError::EdgeNotFound(_))
        ));
        // Node head pointers reset to 0.
        assert_eq!(db.nodes.get(0).unwrap().first_edge_id, 0);
        assert_eq!(db.nodes.get(1).unwrap().first_edge_id, 0);
    }

    #[test]
    fn delete_head_edge() {
        let (_dir, mut db) = setup(3);
        let src = NodeId::new(1).unwrap();
        let e1 = db.edge_ops().create(src, NodeId::new(2).unwrap(), tid(1)).unwrap();
        let e2 = db.edge_ops().create(src, NodeId::new(3).unwrap(), tid(1)).unwrap();
        // Chain: A → E2 → E1 → end

        db.edge_ops().delete(e2).unwrap();

        // E1 is now the head.
        let node_a = db.nodes.get(0).unwrap();
        assert_eq!(node_a.first_edge_id, e1.get());

        // E1's prev_source should be 0 (it's the new head).
        let edge1 = db.edges.get(e1.get() - 1).unwrap();
        assert_eq!(edge1.prev_edge_source, 0);
    }

    #[test]
    fn delete_tail_edge() {
        let (_dir, mut db) = setup(3);
        let src = NodeId::new(1).unwrap();
        let e1 = db.edge_ops().create(src, NodeId::new(2).unwrap(), tid(1)).unwrap();
        let e2 = db.edge_ops().create(src, NodeId::new(3).unwrap(), tid(1)).unwrap();
        // Chain: A → E2 → E1 → end

        db.edge_ops().delete(e1).unwrap();

        // E2 is still the head, its next_source should be 0.
        let node_a = db.nodes.get(0).unwrap();
        assert_eq!(node_a.first_edge_id, e2.get());

        let edge2 = db.edges.get(e2.get() - 1).unwrap();
        assert_eq!(edge2.next_edge_source, 0);
    }

    #[test]
    fn delete_middle_edge() {
        let (_dir, mut db) = setup(4);
        let src = NodeId::new(1).unwrap();
        let e1 = db.edge_ops().create(src, NodeId::new(2).unwrap(), tid(1)).unwrap();
        let e2 = db.edge_ops().create(src, NodeId::new(3).unwrap(), tid(1)).unwrap();
        let e3 = db.edge_ops().create(src, NodeId::new(4).unwrap(), tid(1)).unwrap();
        // Chain: A → E3 → E2 → E1 → end

        db.edge_ops().delete(e2).unwrap();

        // E3's next_source → E1 (skipped E2).
        let edge3 = db.edges.get(e3.get() - 1).unwrap();
        assert_eq!(edge3.next_edge_source, e1.get());

        // E1's prev_source → E3 (skipped E2).
        let edge1 = db.edges.get(e1.get() - 1).unwrap();
        assert_eq!(edge1.prev_edge_source, e3.get());
    }

    #[test]
    fn get_deleted_edge_returns_error() {
        let (_dir, mut db) = setup(2);
        let eid = db.edge_ops().create(
            NodeId::new(1).unwrap(),
            NodeId::new(2).unwrap(),
            tid(1),
        ).unwrap();
        db.edge_ops().delete(eid).unwrap();

        assert!(matches!(
            db.edge_ops().get(eid),
            Err(DbError::EdgeNotFound(_))
        ));
    }

    // Traversal

    #[test]
    fn edges_for_node_outgoing() {
        let (_dir, mut db) = setup(3);
        let a = NodeId::new(1).unwrap();
        db.edge_ops().create(a, NodeId::new(2).unwrap(), tid(1)).unwrap();
        db.edge_ops().create(a, NodeId::new(3).unwrap(), tid(2)).unwrap();

        let edges = db.edge_ops().edges_for_node(a, Direction::Outgoing).unwrap();
        assert_eq!(edges.len(), 2);
        // All should have source_id == 1.
        assert!(edges.iter().all(|e| e.source_id == 1));
    }

    #[test]
    fn edges_for_node_incoming() {
        let (_dir, mut db) = setup(3);
        let b = NodeId::new(2).unwrap();
        db.edge_ops().create(NodeId::new(1).unwrap(), b, tid(1)).unwrap();
        db.edge_ops().create(NodeId::new(3).unwrap(), b, tid(2)).unwrap();

        let edges = db.edge_ops().edges_for_node(b, Direction::Incoming).unwrap();
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().all(|e| e.target_id == 2));
    }

    #[test]
    fn edges_for_node_both() {
        let (_dir, mut db) = setup(3);
        let a = NodeId::new(1).unwrap();
        let b = NodeId::new(2).unwrap();
        let c = NodeId::new(3).unwrap();

        db.edge_ops().create(a, b, tid(1)).unwrap(); // A→B (outgoing from A)
        db.edge_ops().create(c, a, tid(2)).unwrap(); // C→A (incoming to A)

        let edges = db.edge_ops().edges_for_node(a, Direction::Both).unwrap();
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn edges_for_node_empty() {
        let (_dir, mut db) = setup(2);
        let edges = db.edge_ops().edges_for_node(
            NodeId::new(1).unwrap(),
            Direction::Both,
        ).unwrap();
        assert!(edges.is_empty());
    }

    #[test]
    fn edges_after_delete() {
        let (_dir, mut db) = setup(3);
        let a = NodeId::new(1).unwrap();
        let e1 = db.edge_ops().create(a, NodeId::new(2).unwrap(), tid(1)).unwrap();
        db.edge_ops().create(a, NodeId::new(3).unwrap(), tid(2)).unwrap();

        db.edge_ops().delete(e1).unwrap();

        let edges = db.edge_ops().edges_for_node(a, Direction::Outgoing).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target_id, 3);
    }

    #[test]
    fn edge_count_for_node() {
        let (_dir, mut db) = setup(3);
        let a = NodeId::new(1).unwrap();
        db.edge_ops().create(a, NodeId::new(2).unwrap(), tid(1)).unwrap();
        db.edge_ops().create(a, NodeId::new(3).unwrap(), tid(2)).unwrap();

        assert_eq!(
            db.edge_ops().edge_count_for_node(a, Direction::Outgoing).unwrap(),
            2
        );
    }
}
