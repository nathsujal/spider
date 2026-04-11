//! Node CRUD operations with O(1) random access using memory-mapped files.
//!
//! Provides [`NodeAccess`] for managing nodes on disk with constant-time
//! operations via [`RecordFile<Node>`]. Nodes are graph entities with
//! labels, access tracking, and significance scores.

use crate::schema::node::Node;
use crate::store::record::RecordFile;
use crate::error::SpiderResult;

/// Node access manager wrapping nodes.db file.
pub struct NodeAccess {
    file: RecordFile<Node>,
}

impl NodeAccess {
    /// Creates new node with given ID and appends to nodes.db.
    ///
    /// # Errors
    /// - `DbError::Io` if file write fails
    pub fn create(&mut self, id: NodeId) -> SpiderResult<NodeId> {
        let node = Node {
            id: id.get(),
            first_edge_id: 0,
            first_prop_id: 0,
            labels: [0; 4],
            access_count: 0,
            created_at: now_unix_secs(),
            last_accessed_at: 0,
            significance: 128,
        };
        self.file.append(&[node])?;
        Ok(id)
    }

    /// Retrieves node by ID (1‑based). Returns Err if node not found.
    pub fn get(&mut self, id: NodeId) -> SpiderResult<Node> {
        let id_val = id.get().saturating_sub(1);
        self.file.get(id_val)
    }

    /// Touches node: increments access_count and updates last_accessed_at.
    ///
    /// # Returns
    /// New access_count value
    pub fn touch(&mut self, id: NodeId) -> SpiderResult<u32> {
        let idx = id.get().saturating_sub(1);
        let mut node = self.file.get(idx)?;
        node.access_count = node.access_count.saturating_add(1);
        node.last_accessed_at = now_unix_secs();
        self.file.set(idx, &node)?;
        Ok(node.access_count)
    }

    /// Updates node significance (0‑255). Higher values increase bio score.
    pub fn set_significance(&mut self, id: NodeId, sig: u8) -> SpiderResult<()> {
        let idx = id.get().saturating_sub(1);
        let mut node = self.file.get(idx)?;
        node.significance = sig;
        self.file.set(idx, &node)?;
        Ok(())
    }

    /// Marks node as deleted (tombstone) at given ID.
    pub fn delete(&mut self, id: NodeId) -> SpiderResult<bool> {
        let idx = id.get().saturating_sub(1);
        let node = Node {
            id: 0,
            first_edge_id: 0,
            first_prop_id: 0,
            labels: [0; 4],
            access_count: 0,
            created_at: 0,
            last_accessed_at: 0,
            significance: 0,
        };
        self.file.set(idx, &node)?;
        Ok(true)
    }
}

/// Non‑zero node ID (0 = tombstone sentinel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(u32);

impl NodeId {
    /// Constructs NodeId. Returns Err if id == 0 (reserved for deleted slots).
    pub fn new(id: u32) -> Result<Self, crate::error::DbError> {
        if id == 0 {
            Err(crate::error::DbError::NodeNotFound(0))
        } else {
            Ok(Self(id))
        }
    }

    /// Returns underlying u32 value.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// Returns current Unix timestamp in seconds.
fn now_unix_secs() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_rejects_zero() {
        assert!(NodeId::new(0).is_err());
        assert!(NodeId::new(1).is_ok());
    }
}
