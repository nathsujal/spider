use crate::bio;
use crate::search;
use crate::storage::NodeHeader;
use pyo3::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::path::Path;
use std::io::{BufReader, BufWriter};

// SNAPSHOT: Stores data, BUT NOT the Index (we rebuild it)
#[derive(Serialize, Deserialize)]
struct SpiderSnapshot {
    headers: Vec<NodeHeader>,
    data_heap: Vec<u8>,
    edge_list: Vec<Vec<u64>>,
    embeddings: Vec<Vec<f32>>,
}

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
    /// Path to the database file.
    file_path: Option<String>,
}

#[pymethods]
impl SpiderDB {
    // The #[new] macro handles Python arguments
    #[new]
    pub fn new(
        db_path: Option<String>,
        max_capacity: Option<usize>,
        m: Option<usize>,
        ef_construction: Option<usize>
    ) -> PyResult<Self> {
        let db_path = db_path.unwrap_or("./spider.db".to_string());
        
        // --- 1. TRY LOADING FROM DISK ---
        if Path::new(&db_path).exists() {
            // Load Data
            let file = File::open(&db_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            let reader = BufReader::new(file);
            let snapshot: SpiderSnapshot = bincode::deserialize_from(reader)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load DB: {}", e)))?;

            // REBUILD INDEX (The Fix)
            // We create a fresh index and re-insert all vectors.
            let index = search::VectorIndex::new(m, max_capacity, ef_construction);
            for (i, vec) in snapshot.embeddings.iter().enumerate() {
                index.add(i as u64, vec);
            }

            return Ok(SpiderDB {
                headers: snapshot.headers,
                data_heap: snapshot.data_heap,
                edge_list: snapshot.edge_list,
                embeddings: snapshot.embeddings,
                index, // Rebuilt index
                file_path: Some(db_path),
            });
        }

        // --- 2. START FRESH (RAM or New File) ---
        // We don't even need to unwrap here if we pass Options down to search.rs
        // But for vectors, we usually want to reserve capacity immediately.
        let cap = max_capacity.unwrap_or(1_000_000);
        Ok(SpiderDB {
            headers: Vec::with_capacity(cap),
            data_heap: Vec::with_capacity(cap * 100),
            edge_list: Vec::with_capacity(cap),
            embeddings: Vec::with_capacity(cap),
            index: search::VectorIndex::new(m, max_capacity, ef_construction),
            file_path: Some(db_path), // Remember the path (even if it doesn't exist yet)
        })
    }

    /// Adds a new node and AUTOMATICALLY links (bi-directional) it to relevant existing nodes.
    ///
    /// # Arguments
    /// * `content` - Text content.
    /// * `embedding` - Vector.
    /// * `significance` - Bio-importance.
    /// * `auto_link_threshold` (Option<f32>) - If set (e.g., 0.75), automatically creates edges 
    ///                                          to existing nodes with similarity > threshold.
    pub fn add_node(
        &mut self, 
        content: String, 
        embedding: Vec<f32>, 
        significance: u8, 
        auto_link_threshold: Option<f32>
    ) -> u64 {
        let id = self.headers.len() as u64;
        let data_bytes = content.as_bytes();
        let data_offset = self.data_heap.len() as u64;
        let data_len = data_bytes.len() as u32;

        self.data_heap.extend_from_slice(data_bytes);
        
        // Add to HNSW Index
        self.index.add(id, &embedding);
        
        // Keep raw embeddings for now (optional, but good for debugging)
        self.embeddings.push(embedding.clone());

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

        // --- AUTO-LINKING LOGIC ---
        // Default to 0.8 if not provided.
        let threshold = auto_link_threshold.unwrap_or(0.8);

        // Search existing nodes (k=10 to ensure we find valid candidates)
        let similar_nodes = self.index.search(&embedding, 10, Some(64)); 

        let mut edges_to_add = Vec::new();
        let max_auto_links = 3; // <--- LIMIT TO TOP 3

        for (neighbor_id, similarity) in similar_nodes {
            // Don't link to self, and only link if similarity is strong enough
            if neighbor_id != id && similarity >= threshold {
                edges_to_add.push(neighbor_id);
                
                // Stop if we have enough links
                if edges_to_add.len() >= max_auto_links {
                    break;
                }
            }
        }

        // Apply edges
        for neighbor_id in edges_to_add {
            self.add_edge(id, neighbor_id);
        }

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

    /// --- Get Neighbors ---
    pub fn get_neighbors(&self, id: u64) -> Vec<u64> {
        if id as usize >= self.edge_list.len() {
            return Vec::new();
        }
        // Return a clone of the neighbor list
        self.edge_list[id as usize].clone()
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
    /// * `Vec<(u64, f32)>` - List of top-k (node_id, score).
    pub fn hybrid_search(&self, query_embedding: Vec<f32>, k: usize, ef_search: Option<usize>) -> Vec<(u64, f32)> {
        let min_score = 0.3;
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

        // Step 3: Deduplicate
        context_pool.sort();
        context_pool.dedup();
        
        // Step 4: Rank by Similarity (Cosine)
        // We calculate the exact cosine similarity for every candidate in the pool.
        let mut final_results: Vec<(u64, f32)> = context_pool.into_iter().map(|id| {
            let node_embedding = &self.embeddings[id as usize];
            let score = search::cosine_similarity(&query_embedding, node_embedding);
            (id, score)
        })
        .filter(|&(_, score)| score >= min_score)
        .collect();

        // Sort desc by Similarity
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top K
        final_results.into_iter().take(k).collect()
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

    /// Saves the database.
    ///
    /// # Arguments
    /// * `path` (Optional) - If provided, saves to this new path.
    ///                       If None, saves to the path provided at init.
    pub fn save(&self, path: Option<String>) -> PyResult<()> {
        // Determine target path: Argument > Init Path > Error
        let target_path = match path {
            Some(p) => p,
            None => match &self.file_path {
                Some(p) => p.clone(),
                None => return Err(pyo3::exceptions::PyValueError::new_err("No path specified for save() and no db_path provided at init.")),
            },
        };

        let snapshot = SpiderSnapshot {
            headers: self.headers.clone(),
            data_heap: self.data_heap.clone(),
            edge_list: self.edge_list.clone(),
            embeddings: self.embeddings.clone(),
        };

        let file = File::create(&target_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let writer = BufWriter::new(file);
        
        bincode::serialize_into(writer, &snapshot)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
        Ok(())
    }

    /// Exports the entire graph for visualization.
    /// Returns a tuple: (nodes, edges)
    /// - nodes: List of (id, label, significance)
    /// - edges: List of (source_id, target_id)
    pub fn get_all_graph_data(&self) -> (Vec<(u64, String, u8)>, Vec<(u64, u64)>) {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // 1. Collect Nodes
        for header in &self.headers {
            // Fetch a short snippet of text for the label
            let start = header.data_offset as usize;
            let end = start + header.data_len as usize;
            
            let label = if end <= self.data_heap.len() {
                // Get first 30 chars of content
                let slice = &self.data_heap[start..end];
                let full_text = String::from_utf8_lossy(slice);
                full_text.chars().take(30).collect::<String>()
            } else {
                format!("Node {}", header.id)
            };
            
            nodes.push((header.id, label, header.significance));
        }

        // 2. Collect Edges
        for (source_idx, targets) in self.edge_list.iter().enumerate() {
            let source_id = source_idx as u64;
            for &target_id in targets {
                // Only export unique edges to avoid clutter (optional)
                edges.push((source_id, target_id));
            }
        }

        (nodes, edges)
    }
}
