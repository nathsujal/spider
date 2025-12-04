use crate::bio;
use crate::search;
use crate::storage::NodeHeader;
use pyo3::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

/// The main database struct holding all data arenas.
#[pyclass]
pub struct SpiderDB {
    /// Fixed-size metadata headers.
    headers: Vec<NodeHeader>,
    /// Variable-length content storage.
    data_heap: Vec<u8>,
    /// Adjacency list (Vector of Vectors) for easy updates
    edge_list: Vec<Vec<u64>>,
    /// Vector embeddings for nodes.
    embeddings: Vec<Vec<f32>>,
    /// HNSW Index for fast approximate nearest neighbor search.
    index: search::VectorIndex,
}

#[pymethods]
impl SpiderDB {
    // The #[new] macro handles Python arguments
    #[new]
    pub fn new(
        max_capacity: Option<usize>,
        m: Option<usize>,
        ef_construction: Option<usize>
    ) -> Self {
        // We don't even need to unwrap here if we pass Options down to search.rs
        // But for vectors, we usually want to reserve capacity immediately.
        let cap = max_capacity.unwrap_or(1_000_000);

        SpiderDB {
            headers: Vec::with_capacity(cap),
            data_heap: Vec::with_capacity(cap * 100),
            edge_list: Vec::with_capacity(cap),
            embeddings: Vec::with_capacity(cap),
            index: search::VectorIndex::new(m, max_capacity, ef_construction),
        }
    }

    /// Adds a new node to the database.
    ///
    /// # Arguments
    ///
    /// * `content` - The string content of the node.
    /// * `embedding` - The vector embedding of the node.
    /// * `significance` - The significance score (0-255).
    ///
    /// # Returns
    ///
    /// * `u64` - The ID of the newly created node.
    pub fn add_node(&mut self, content: String, embedding: Vec<f32>, significance: u8) -> u64 {
        let id = self.headers.len() as u64;
        let data_bytes = content.as_bytes();
        let data_offset = self.data_heap.len() as u64;
        let data_len = data_bytes.len() as u32;

        self.data_heap.extend_from_slice(data_bytes);
        
        // Add to HNSW Index
        self.index.add(id, &embedding);
        
        // Keep raw embeddings for now (optional, but good for debugging)
        self.embeddings.push(embedding);

        // Initialize empty edge list for this new node
        self.edge_list.push(Vec::new());

        let header = NodeHeader {
            id,
            data_offset,
            data_len,
            edge_start: 0,
            edge_count: 0,
            last_access_ts: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            access_count: 0,
            significance,
        };

        self.headers.push(header);
        id
    }

    /// Adds a Bi-directional edge from source to target.
    ///
    /// # Arguments
    ///
    /// * `source_id` - ID of the source node.
    /// * `target_id` - ID of the target node.
    ///
    /// A -> B AND B -> A === A <-> B
    pub fn add_edge(&mut self, source_id: u64, target_id: u64) {
        if source_id as usize >= self.headers.len() || target_id as usize >= self.headers.len() {
            return;
        }
        
        // 1. Add Forward Link (Source -> Target) [Child]
        self.edge_list[source_id as usize].push(target_id);
        self.headers[source_id as usize].edge_count += 1;

        // 2. Add Backward Link (Target -> Source) [Parent]
        self.edge_list[target_id as usize].push(source_id);
        self.headers[target_id as usize].edge_count += 1;
    }

    /// Retrieves a node's content by ID.
    pub fn get_node(&mut self, id: u64) -> Option<String> {
        if id as usize >= self.headers.len() {
            return None;
        }

        let header = &mut self.headers[id as usize];
        
        // Update Bio-Metrics
        header.last_access_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        header.access_count += 1;

        let start = header.data_offset as usize;
        let end = start + header.data_len as usize;
        
        if end > self.data_heap.len() {
            return None;
        }

        let bytes = &self.data_heap[start..end];
        String::from_utf8(bytes.to_vec()).ok()
    }

    /// Performs a hybrid search combining vector similarity and biological score.
    ///
    /// # Arguments
    ///
    /// * `query_embedding` - The query vector.
    /// * `k` - Number of results to return.
    ///
    /// # Returns
    ///
    /// * `Vec<u64>` - List of top-k node IDs.
    pub fn hybrid_search(&self, query_embedding: Vec<f32>, k: usize, ef_search: Option<usize>) -> Vec<u64> {
        // Step 1: Vector Search (The GPS)
        let similar = self.index.search(&query_embedding, k, ef_search);
        
        // Step 2: Graph Traversal (The Expansion)
        // We collect the matches AND their direct neighbors
        let mut context_pool = Vec::new();
        
        for (id, _score) in similar {
            context_pool.push(id); 
            
            // Pull NEIGHBORS, PARENTS (incoming), and CHILDREN (outgoing)
            if let Some(neighbors) = self.edge_list.get(id as usize) {
                for &neighbor_id in neighbors {
                    context_pool.push(neighbor_id);
                }
            }
        }

        // Step 3: Deduplicate & Rank by Life Score
        context_pool.sort();
        context_pool.dedup();
        
        // Re-rank the expanded pool purely by Biological Importance
        let mut final_results: Vec<(u64, f32)> = context_pool.into_iter().map(|id| {
            let bio_score = bio::calc_life_score(&self.headers[id as usize]);
            (id, bio_score)
        }).collect();

        // Sort desc by Life Score
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top K from the *expanded* set
        final_results.into_iter().take(k).map(|(id, _)| id).collect()
    }

    /// Identifies nodes that should be removed based on their Life Score.
    pub fn vacuum(&self, threshold: f32) -> Vec<u64> {
        let mut dead_nodes = Vec::new();
        for header in &self.headers {
            let score = bio::calc_life_score(header);
            if score < threshold {
                dead_nodes.push(header.id);
            }
        }
        dead_nodes
    }

    /// Calculates the life score of a node.
    pub fn calculate_life_score(&self, id: u64) -> f32 {
        if id as usize >= self.headers.len() {
            return 0.0;
        }
        bio::calc_life_score(&self.headers[id as usize])
    }
}
