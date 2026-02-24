//! Node operations: CRUD + bio scoring

use super::*;

impl Spider {
    /// Create a new node with the given labels. Returns the new node's ID.
    pub fn create_node(&mut self, labels: &[&str]) -> Result<u32> {
        if labels.len() > 4 {
            return Err(DbError::TooManyLabels {max: 4 });
        }

        let mut label_ids = [0u8; 4];
        for (i, label) in labels.iter().take(4).enumerate() {
            label_ids[i] = self.labels.get_or_create(label).ok_or(DbError::TokenStoreExhausted {
                store: "labels",
            })?;
        }

        let id = self.node_free.allocate(&mut self.meta.next_node_id);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        let node = NodeRecord::new(id, &label_ids[..labels.len()], now);
        self.nodes.write(id, &node)?;

        Ok(id)
    }

    /// Get a node by ID. Retuns `None` if not found or deleted.
    pub fn get_node(&self, id:u32) -> Option<NodeRecord> {
        let node = self.nodes.read(id)?;
        if node.is_deleted() { None } else { Some(node) }
    }

    /// Delete a node by ID, cascade-deleting all its properties and relationships.
    pub fn delete_node(&mut self, id: u32) -> Result<()> {
        let node = self.nodes.read(id).ok_or(DbError::NodeNotFound(id))?;
        if node.is_deleted() { 
            return Err(DbError::NodeNotFound(id));
        }

        if node.first_prop_id != 0 {
            self.free_all_properties(node.first_prop_id)?;
        }

        let mut rel_id = node.first_rel_id;
        while rel_id != 0 {
            if let Some(rel) = self.rels.read(rel_id) {
                let next = if rel.source_id == id {
                    rel.next_rel_source
                } else {
                    rel.next_rel_target
                };
                self.delete_rel(rel_id)?;
                rel_id = next;
            } else {
                break;
            }
        }

        let empty = NodeRecord::empty();
        self.nodes.write(id, &empty)?;
        self.node_free.free(id);

        Ok(())
    }

    /// Get all live (non-deleted) node IDs.
    pub fn get_all_node_ids(&self) -> Vec<u32> {
        (1..self.meta.next_node_id)
            .filter_map(|id| self.nodes.read(id))
            .filter(|n| !n.is_deleted())
            .map(|n| n.id)
            .collect()
    }

    /// Number of live nodes.
    pub fn node_count(&self) -> usize {
        self.get_all_node_ids().len()
    }

    // Bio

    /// Reinforce a node - increments access_count and update last_accessed_at
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
    pub fn set_significance(&mut self, id: u32, significance: u8) -> Result<()> {
        let mut node = self.nodes.read(id).ok_or(DbError::NodeNotFound(id))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(id));
        }

        node.significance = significance;
        self.nodes.write(id, &node)?;
        Ok(())
    }

    /// Calculate the current bio-score for a node. Returns 0.0 if deleted or missing.
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

        bio::calculate_bio_score(
            node.access_count,
            node.significance,
            node.last_accessed_at,
            now,
            &params,
        )
    }

    /// Set the database-level bio scoring parameters (persisted in `meta.db`).
    pub fn set_bio_params(&mut self, w_sig: f64, w_freq: f64, gravity: f64) -> Result<()> {
        self.meta.bio_w_sig = w_sig;
        self.meta.bio_w_freq = w_freq;
        self.meta.bio_gravity = gravity;
        Ok(())
    }

    /// Get the database-level bio scoring parameters.
    pub fn get_bio_params(&self) -> Result<(f64, f64, f64)> {
        Ok((
            self.meta.bio_w_sig,
            self.meta.bio_w_freq,
            self.meta.bio_gravity
        ))
    }
}