use crate::bio;
use crate::search;
use crate::storage::NodeHeader;
use crate::cluster::{ClusterEngine, ClusterConfig, Cluster};
use crate::ranking;
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
    clusters: Option<Vec<Cluster>>,
    cluster_config: ClusterConfig,
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
    /// Cached cluster hierarchy
    clusters: Option<Vec<Cluster>>,
    /// Clustering configuration
    cluster_config: ClusterConfig,
    /// Path to the database file.
    file_path: Option<String>,
    
    // ========================================================================
    // AUTO-NOTIFY CONFIGURATION
    // These fields enable automatic notification to the visualization server
    // whenever the database is saved.
    // ========================================================================
    
    /// URL of the visualization server (e.g., "http://localhost:3000")
    /// When set, the database will automatically notify the server on save.
    server_url: Option<String>,
    
    /// Whether to automatically notify the server when save() is called.
    /// Defaults to true. Set to false to disable auto-notify.
    auto_notify: bool,
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
                clusters: snapshot.clusters,
                cluster_config: snapshot.cluster_config,
                server_url: None,           // No server by default
                auto_notify: true,           // Auto-notify enabled by default
            });
        }

        // --- 2. START FRESH (RAM or New File) ---
        let cap = max_capacity.unwrap_or(1_000_000);
        Ok(SpiderDB {
            headers: Vec::with_capacity(cap),
            data_heap: Vec::with_capacity(cap * 100),
            edge_list: Vec::with_capacity(cap),
            embeddings: Vec::with_capacity(cap),
            index: search::VectorIndex::new(m, max_capacity, ef_construction),
            file_path: Some(db_path), // Remember the path (even if it doesn't exist yet)
            clusters: None,
            cluster_config: ClusterConfig::default(),
            server_url: None,           // No server by default
            auto_notify: true,           // Auto-notify enabled by default
        })
    }

    /// Adds a new node and AUTOMATICALLY links (bi-directional) it to relevant existing nodes.
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
        
        // Keep raw embeddings for now
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

    /// Builds or rebuilds the cluster hierarchy
    pub fn build_clusters(&mut self, k_clusters: Option<usize>) -> PyResult<()> {
        let k = k_clusters.unwrap_or(10); // Default: 10 root clusters
        
        let engine = ClusterEngine::new(self.cluster_config.clone());
        
        self.clusters = Some(engine.cluster_graph(
            &self.headers,
            &self.embeddings,
            &self.edge_list,
            k,
        ));
        
        Ok(())
    }

    /// Get all clusters (returns simplified structure for Python)
    pub fn get_clusters(&self) -> Vec<(u64, u64, Vec<u64>, f32)> {
        match &self.clusters {
            Some(clusters) => {
                clusters.iter().map(|c| {
                    (c.id, c.anchor_node_id, c.member_ids.clone(), c.significance)
                }).collect()
            }
            None => Vec::new(),
        }
    }

    /// Find which cluster(s) a node belongs to
    pub fn get_node_clusters(&self, node_id: u64) -> Vec<u64> {
        match &self.clusters {
            Some(clusters) => {
                let engine = ClusterEngine::new(self.cluster_config.clone());
                engine.find_node_clusters(node_id, clusters)
            }
            None => Vec::new(),
        }
    }

    /// Search within a specific cluster
    pub fn search_in_cluster(
        &self,
        cluster_id: u64,
        query_embedding: Vec<f32>,
        k: usize,
    ) -> Vec<(u64, f32)> {
        match &self.clusters {
            Some(clusters) => {
                // Find the cluster
                if let Some(cluster) = clusters.iter().find(|c| c.id == cluster_id) {
                    let engine = ClusterEngine::new(self.cluster_config.clone());
                    return engine.cluster_search(&query_embedding, cluster, &self.embeddings, k);
                }
                Vec::new()
            }
            None => Vec::new(),
        }
    }

    /// Get cluster statistics
    pub fn get_cluster_stats(&self) -> Option<(usize, f32, f32)> {
        self.clusters.as_ref().map(|clusters| {
            let total_clusters = clusters.len();
            let avg_size = clusters.iter()
                .map(|c| c.member_ids.len())
                .sum::<usize>() as f32 / total_clusters as f32;
            let avg_sig = clusters.iter()
                .map(|c| c.significance)
                .sum::<f32>() / total_clusters as f32;
            
            (total_clusters, avg_size, avg_sig)
        })
    }

    /// Export cluster hierarchy as string (for debugging)
    pub fn export_cluster_tree(&self) -> String {
        match &self.clusters {
            Some(clusters) => {
                clusters.iter()
                    .map(|c| crate::cluster::export_cluster_tree(c))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            None => "No clusters built yet. Call build_clusters() first.".to_string(),
        }
    }

    /// Adds a Bi-directional edge from source to target.
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

    /// Performs a hybrid search combining vector similarity, biological score, and cluster awareness.
    /// It identifies and re-ranks candidates from clusters or HNSW based on a combined score.
    ///
    /// # Arguments
    /// * `query_embedding` - A vector of floats representing the query embedding.
    /// * `k` - The desired number of top results to return.
    /// * `ef_search` - An optional parameter for the HNSW search, controlling the size of the
    ///                 dynamic list of neighbors during search. If `None`, a default is used.
    ///
    /// # Returns
    /// A `Vec` of tuples, where each tuple contains a node ID (`u64`) and its
    /// combined similarity score (`f32`), sorted in descending order of score.
    pub fn hybrid_search(
        &mut self, 
        query_embedding: Vec<f32>, 
        k: usize, 
        ef_search: Option<usize>
    ) -> Vec<(u64, f32)> {
        let config = ranking::RankConfig::default();

        // 1. Get Candidates (Cluster-aware or Raw Index)
        let candidates = if self.clusters.is_some() {
            ranking::find_cluster_candidates(self.clusters.as_ref().unwrap(), &query_embedding, k * 3)
        } else {
            self.index.search(&query_embedding, k * 3, ef_search).into_iter().map(|(id, _)| id).collect()
        };

        // 2. Expand Graph
        let pool = ranking::expand_with_neighbors(&candidates, &self.edge_list, 2);
        let mut scored = Vec::new();

        // 3. Score Everything
        for id in pool {
            if id as usize >= self.embeddings.len() { continue; }
            
            let semantic = search::cosine_similarity(&query_embedding, &self.embeddings[id as usize]);
            let graph = ranking::calculate_graph_score(id, &self.edge_list, &self.embeddings, &candidates, &query_embedding);
            let bio = ranking::calculate_bio_score(&self.headers[id as usize]);
            let cluster = ranking::calculate_cluster_score(id, self.clusters.as_ref(), &query_embedding);

            let total = (semantic * config.semantic_weight) + 
                        (graph * config.graph_weight) + 
                        (bio * config.bio_weight) + 
                        (cluster * config.cluster_weight);

            if total >= 0.25 {
                scored.push((id, total));
            }
        }

        // 4. Sort and Update Access
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let final_results: Vec<(u64, f32)> = scored.into_iter().take(k).collect();

        // Update bio-metrics for the winners
        for (id, _) in &final_results {
            self.update_node_access(*id);
        }

        final_results
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

    // ========================================================================
    // AUTO-NOTIFY CONFIGURATION METHODS
    // ========================================================================

    /// Set the visualization server URL.
    /// 
    /// When set, the database will automatically POST to {server_url}/api/notify
    /// whenever save() is called, triggering a realtime update in connected browsers.
    /// 
    /// Example:
    ///     db.set_server_url("http://localhost:3000")
    ///     db.add_node(...)
    ///     db.save()  # Automatically notifies the server
    pub fn set_server_url(&mut self, url: String) {
        self.server_url = Some(url);
    }

    /// Enable or disable automatic server notification on save.
    /// 
    /// When enabled (default), calling save() will also notify the visualization
    /// server if server_url is set.
    pub fn set_auto_notify(&mut self, enabled: bool) {
        self.auto_notify = enabled;
    }

    /// Get the current server URL (if set)
    pub fn get_server_url(&self) -> Option<String> {
        self.server_url.clone()
    }

    /// Manually trigger a server notification.
    /// 
    /// This is useful if you want to notify without saving, or if auto_notify is disabled.
    /// Returns true if notification was successful, false otherwise.
    pub fn notify(&self) -> bool {
        if let Some(ref url) = self.server_url {
            match self.send_notify(url) {
                Ok(_) => true,
                Err(e) => {
                    eprintln!("[SpiderDB] Failed to notify server: {}", e);
                    false
                }
            }
        } else {
            eprintln!("[SpiderDB] No server URL set. Call set_server_url() first.");
            false
        }
    }

    /// Saves the database and optionally notifies the visualization server.
    /// 
    /// If server_url is set and auto_notify is true, this will automatically
    /// POST to the server's /api/notify endpoint after saving.
    pub fn save(&self, path: Option<String>) -> PyResult<()> {
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
            clusters: self.clusters.clone(),
            cluster_config: self.cluster_config.clone(),
        };

        let file = File::create(&target_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let writer = BufWriter::new(file);
        
        bincode::serialize_into(writer, &snapshot)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // ====================================================================
        // AUTO-NOTIFY: If server URL is set and auto_notify is enabled,
        // POST to the server's notify endpoint to trigger realtime updates
        // ====================================================================
        if self.auto_notify {
            if let Some(ref url) = self.server_url {
                if let Err(e) = self.send_notify(url) {
                    // Log but don't fail - saving succeeded, notification is optional
                    eprintln!("[SpiderDB] Auto-notify failed (save succeeded): {}", e);
                }
            }
        }
            
        Ok(())
    }

    /// Exports the entire graph for visualization.
    pub fn get_all_graph_data(&self) -> (Vec<(u64, String, u8, Option<u64>)>, Vec<(u64, u64)>) {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // 0. Build Node -> Cluster Map
        let mut node_cluster_map = std::collections::HashMap::new();
        if let Some(clusters) = &self.clusters {
             // Helper to recursively map
             fn map_clusters(clusters: &[Cluster], map: &mut std::collections::HashMap<u64, u64>) {
                 for c in clusters {
                     for &member in &c.member_ids {
                         map.insert(member, c.id);
                     }
                     map_clusters(&c.sub_clusters, map);
                 }
             }
             map_clusters(clusters, &mut node_cluster_map);
        }

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
            
            let cluster_id = node_cluster_map.get(&header.id).copied();
            nodes.push((header.id, label, header.significance, cluster_id));
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

impl SpiderDB {
    /// Update access metrics when a node is retrieved
    fn update_node_access(&mut self, node_id: u64) {
        if node_id as usize >= self.headers.len() {
            return;
        }

        let header = &mut self.headers[node_id as usize];
        header.access_count += 1;
        header.last_access_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    // ========================================================================
    // INTERNAL HELPERS: Notification to visualization server
    // ========================================================================

    /// Try to send a notification to the server.
    /// 
    /// This is a silent helper that only notifies if:
    /// 1. server_url is set
    /// 2. auto_notify is enabled
    /// 
    /// Failures are logged but don't cause errors.
    fn try_notify(&self) {
        if self.auto_notify {
            if let Some(ref url) = self.server_url {
                if let Err(e) = self.send_notify(url) {
                    eprintln!("[SpiderDB] Auto-notify failed: {}", e);
                }
            }
        }
    }
    
    /// Send an HTTP POST to the server's notify endpoint.
    /// 
    /// This uses the reqwest blocking client to make a synchronous HTTP call.
    /// The endpoint triggers a WebSocket broadcast to all connected clients.
    fn send_notify(&self, server_url: &str) -> Result<(), String> {
        // Construct the notify endpoint URL
        let notify_url = format!("{}/api/notify", server_url.trim_end_matches('/'));
        
        // Use reqwest blocking client (sync)
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        
        // Send POST request
        let response = client.post(&notify_url)
            .send()
            .map_err(|e| format!("Failed to send notify request: {}", e))?;
        
        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!("Server returned error status: {}", response.status()))
        }
    }
}
