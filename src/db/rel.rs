//! Relationship operations: CRUD + graph traversal.

use super::*;

impl Spider {
    /// Create a new relationship between two nodes with the given type.
    pub fn create_rel(&mut self, source_id: u32, target_id: u32, rel_type: &str) -> Result<u32> {
        // Validate node existence
        let mut source = self.get_node(source_id).ok_or(DbError::SourceNodeNotFound(source_id))?;
        let mut target = self.get_node(target_id).ok_or(DbError::TargetNodeNotFound(target_id))?;

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
        let rel = self.rels.read(id).ok_or(DbError::RelNotFound(id))?;
        if rel.is_deleted() {
            return Ok(());
        }

        // Free relationship's property chain (props + DynStrings)
        if rel.first_prop_id != 0 {
            self.free_all_properties(rel.first_prop_id)?;
        }

        // Unlink from source chain
        self.unlink_rel_from_node(rel.source_id, id, &rel, true)?;

        // Unlink from target chain
        if rel.target_id != rel.source_id {
            self.unlink_rel_from_node(rel.target_id, id, &rel, false)?;
        }

        // Mark as deleted
        let empty = RelRecord::empty();
        self.rels.write(id, &empty)?;

        // Return ID to free list
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
}