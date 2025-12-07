// ============================================================================
// SPIDER GRAPH VISUALIZER SERVER
// ============================================================================
//
// This server provides REST and WebSocket APIs for realtime graph visualization.
// It loads the Spider database and serves graph data to connected clients.
//
// Architecture:
// ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
// │  SpiderDB   │────▶│ Axum Server │────▶│  Browser    │
// │  (.db file) │     │ REST + WS   │     │  Sigma.js   │
// └─────────────┘     └─────────────┘     └─────────────┘
//
// Usage:
//   cargo run --bin spider-server -- --db path/to/spider.db --port 3000
//
// ============================================================================

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
    Router,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
};
use tokio::sync::{broadcast, RwLock};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Represents a node in the graph for JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier for the node
    pub id: u64,
    /// Short label for display
    pub label: String,
    /// Full content of the node
    pub content: String,
    /// Significance score (0-9)
    pub significance: u8,
    /// Cluster ID this node belongs to (if any)
    pub cluster_id: Option<u64>,
    /// X position for layout (optional, can be computed client-side)
    pub x: Option<f32>,
    /// Y position for layout (optional, can be computed client-side)
    pub y: Option<f32>,
    /// Color for this node (hex string like "#FF5733")
    pub color: String,
}

/// Represents an edge in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub source: u64,
    /// Target node ID
    pub target: u64,
}

/// Complete graph data for initial load
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphData {
    /// All nodes in the graph
    pub nodes: Vec<GraphNode>,
    /// All edges in the graph
    pub edges: Vec<GraphEdge>,
    /// Cluster metadata (id -> color mapping)
    pub clusters: HashMap<u64, ClusterInfo>,
}

/// Cluster information for the legend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub id: u64,
    pub color: String,
    pub node_count: usize,
    pub significance: f32,
}

/// Statistics about the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub cluster_count: usize,
}

/// WebSocket event types that can be sent to clients
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsEvent {
    /// Full graph snapshot (sent on initial connection)
    #[serde(rename = "graph_snapshot")]
    GraphSnapshot(GraphData),
    
    /// A new node was added
    #[serde(rename = "node_added")]
    NodeAdded(GraphNode),
    
    /// A new edge was added
    #[serde(rename = "edge_added")]
    EdgeAdded(GraphEdge),
    
    /// Cluster information was updated
    #[serde(rename = "cluster_updated")]
    ClusterUpdated(ClusterInfo),
    
    /// Full graph refresh (when clusters are rebuilt)
    #[serde(rename = "refresh")]
    Refresh,
}

// ============================================================================
// APPLICATION STATE
// ============================================================================

/// Shared application state accessible by all request handlers
pub struct AppState {
    /// Path to the Spider database file
    pub db_path: PathBuf,
    
    /// Broadcast channel for sending updates to all connected WebSocket clients
    /// When we call `tx.send(event)`, all subscribers receive it
    pub broadcast_tx: broadcast::Sender<WsEvent>,
    
    /// Cached graph data (refreshed on notify)
    pub graph_cache: RwLock<Option<GraphData>>,
}

impl AppState {
    /// Create new application state with the given database path
    pub fn new(db_path: PathBuf) -> Arc<Self> {
        // Create a broadcast channel with buffer size 100
        // This means we can have up to 100 pending messages before older ones are dropped
        let (tx, _rx) = broadcast::channel(100);
        
        Arc::new(Self {
            db_path,
            broadcast_tx: tx,
            graph_cache: RwLock::new(None),
        })
    }
    
    /// Load graph data from the database file
    /// This reads the .db file and converts it to our JSON-friendly format
    pub async fn load_graph_data(&self) -> Result<GraphData, String> {
        // Read the database file
        let db_path = self.db_path.clone();
        
        // We need to spawn a blocking task because file I/O is synchronous
        let result = tokio::task::spawn_blocking(move || {
            load_graph_from_db(&db_path)
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?;
        
        result
    }
    
    /// Refresh the cache and notify all clients
    pub async fn notify_update(&self) {
        // Reload data from disk
        match self.load_graph_data().await {
            Ok(data) => {
                // Update cache
                let mut cache = self.graph_cache.write().await;
                *cache = Some(data.clone());
                
                // Broadcast refresh event to all connected clients
                let _ = self.broadcast_tx.send(WsEvent::GraphSnapshot(data));
                println!("📡 Broadcasted graph update to clients");
            }
            Err(e) => {
                eprintln!("❌ Failed to load graph: {}", e);
            }
        }
    }
}

// ============================================================================
// DATABASE LOADING
// ============================================================================

/// Load graph data from a Spider database file
/// This function reads the binary .db file format
fn load_graph_from_db(db_path: &PathBuf) -> Result<GraphData, String> {
    use std::fs::File;
    use std::io::BufReader;
    
    // Check if file exists
    if !db_path.exists() {
        return Err(format!("Database file not found: {:?}", db_path));
    }
    
    // Open and deserialize the database snapshot
    let file = File::open(db_path)
        .map_err(|e| format!("Failed to open database: {}", e))?;
    let reader = BufReader::new(file);
    
    // The database uses bincode serialization
    // We need to match the SpiderSnapshot struct from db.rs
    #[derive(Deserialize)]
    struct SpiderSnapshot {
        headers: Vec<NodeHeader>,
        data_heap: Vec<u8>,
        edge_list: Vec<Vec<u64>>,
        #[allow(dead_code)]
        embeddings: Vec<Vec<f32>>,
        clusters: Option<Vec<Cluster>>,
        #[allow(dead_code)]
        cluster_config: ClusterConfig,
    }
    
    #[derive(Deserialize)]
    struct NodeHeader {
        id: u64,
        data_offset: u64,
        data_len: u32,
        #[allow(dead_code)]
        edge_start: u32,
        #[allow(dead_code)]
        edge_count: u32,
        #[allow(dead_code)]
        last_access_ts: u64,
        #[allow(dead_code)]
        access_count: u32,
        significance: u8,
    }
    
    #[derive(Deserialize)]
    struct Cluster {
        id: u64,
        #[allow(dead_code)]
        anchor_node_id: u64,
        member_ids: Vec<u64>,
        #[allow(dead_code)]
        centroid: Vec<f32>,
        significance: f32,
        #[allow(dead_code)]
        sub_clusters: Vec<Cluster>,
        #[allow(dead_code)]
        depth: usize,
    }
    
    #[derive(Deserialize)]
    struct ClusterConfig {
        #[allow(dead_code)]
        min_cluster_size: usize,
        #[allow(dead_code)]
        max_cluster_size: usize,
        #[allow(dead_code)]
        max_depth: usize,
        #[allow(dead_code)]
        similarity_threshold: f32,
    }
    
    let snapshot: SpiderSnapshot = bincode::deserialize_from(reader)
        .map_err(|e| format!("Failed to deserialize database: {}", e))?;
    
    // Build node -> cluster mapping
    let mut node_cluster_map: HashMap<u64, u64> = HashMap::new();
    let mut cluster_infos: HashMap<u64, ClusterInfo> = HashMap::new();
    
    if let Some(ref clusters) = snapshot.clusters {
        for cluster in clusters {
            // Generate a color for this cluster based on its ID
            let color = generate_cluster_color(cluster.id, clusters.len());
            
            cluster_infos.insert(cluster.id, ClusterInfo {
                id: cluster.id,
                color: color.clone(),
                node_count: cluster.member_ids.len(),
                significance: cluster.significance,
            });
            
            // Map each member to this cluster
            for &member_id in &cluster.member_ids {
                node_cluster_map.insert(member_id, cluster.id);
            }
        }
    }
    
    // Convert headers to GraphNodes
    let mut nodes = Vec::new();
    for header in &snapshot.headers {
        // Extract content from data heap
        let start = header.data_offset as usize;
        let end = start + header.data_len as usize;
        let content = if end <= snapshot.data_heap.len() {
            String::from_utf8_lossy(&snapshot.data_heap[start..end]).to_string()
        } else {
            format!("Node {}", header.id)
        };
        
        // Get cluster info
        let cluster_id = node_cluster_map.get(&header.id).copied();
        let color = if let Some(cid) = cluster_id {
            cluster_infos.get(&cid)
                .map(|c| c.color.clone())
                .unwrap_or_else(|| "#808080".to_string())
        } else {
            "#808080".to_string() // Gray for unclustered nodes
        };
        
        // Create short label (first 30 chars)
        let label = content.chars().take(30).collect::<String>();
        
        nodes.push(GraphNode {
            id: header.id,
            label,
            content,
            significance: header.significance,
            cluster_id,
            x: None, // Let client compute positions
            y: None,
            color,
        });
    }
    
    // Convert edge list to GraphEdges
    // Note: edge_list contains duplicates (bidirectional), so we dedupe
    let mut edges = Vec::new();
    let mut seen_edges: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
    
    for (source_idx, targets) in snapshot.edge_list.iter().enumerate() {
        let source = source_idx as u64;
        for &target in targets {
            // Only add each edge once (smaller id first to ensure uniqueness)
            let key = if source < target { (source, target) } else { (target, source) };
            if !seen_edges.contains(&key) {
                seen_edges.insert(key);
                edges.push(GraphEdge { source, target });
            }
        }
    }
    
    Ok(GraphData {
        nodes,
        edges,
        clusters: cluster_infos,
    })
}

/// Generate a distinct color for a cluster using HSL color space
fn generate_cluster_color(cluster_id: u64, total_clusters: usize) -> String {
    // Distribute hues evenly across the spectrum
    let hue = (cluster_id as f64 / total_clusters.max(1) as f64) * 360.0;
    let saturation = 75.0; // Vibrant colors
    let lightness = 55.0;  // Not too dark, not too light
    
    // Convert HSL to RGB
    let (r, g, b) = hsl_to_rgb(hue, saturation, lightness);
    
    format!("#{:02X}{:02X}{:02X}", r, g, b)
}

/// Convert HSL to RGB color values
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    let s = s / 100.0;
    let l = l / 100.0;
    
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    
    let (r, g, b) = match (h as u32) / 60 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

// ============================================================================
// HTTP HANDLERS
// ============================================================================

/// GET /api/graph
/// Returns the complete graph data (nodes, edges, clusters)
async fn get_graph(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Try to use cached data first
    {
        let cache = state.graph_cache.read().await;
        if let Some(ref data) = *cache {
            return Json(data.clone()).into_response();
        }
    }
    
    // Cache miss - load from disk
    match state.load_graph_data().await {
        Ok(data) => {
            // Update cache
            let mut cache = state.graph_cache.write().await;
            *cache = Some(data.clone());
            Json(data).into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, e).into_response()
        }
    }
}

/// GET /api/stats
/// Returns basic statistics about the graph
async fn get_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.load_graph_data().await {
        Ok(data) => {
            let stats = GraphStats {
                node_count: data.nodes.len(),
                edge_count: data.edges.len(),
                cluster_count: data.clusters.len(),
            };
            Json(stats).into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, e).into_response()
        }
    }
}

/// GET /api/clusters
/// Returns cluster information for the legend
async fn get_clusters(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.load_graph_data().await {
        Ok(data) => {
            Json(data.clusters).into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, e).into_response()
        }
    }
}

/// POST /api/notify
/// Called by client code to trigger a refresh
/// This is the "Option B" mechanism - explicit notification
async fn notify_update(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.notify_update().await;
    Json(serde_json::json!({ "status": "ok", "message": "Update broadcasted" }))
}

// ============================================================================
// WEBSOCKET HANDLER
// ============================================================================

/// GET /ws
/// WebSocket endpoint for realtime updates
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Upgrade the HTTP connection to a WebSocket connection
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

/// Handle an individual WebSocket connection
async fn handle_websocket(socket: WebSocket, state: Arc<AppState>) {
    // Split the socket into sender and receiver halves
    let (mut sender, mut receiver) = socket.split();
    
    // Subscribe to broadcast updates
    let mut broadcast_rx = state.broadcast_tx.subscribe();
    
    println!("🔌 New WebSocket client connected");
    
    // Send initial graph snapshot to the new client
    match state.load_graph_data().await {
        Ok(data) => {
            let event = WsEvent::GraphSnapshot(data);
            let json = serde_json::to_string(&event).unwrap();
            if sender.send(Message::Text(json)).await.is_err() {
                println!("❌ Failed to send initial snapshot");
                return;
            }
            println!("📤 Sent initial graph snapshot");
        }
        Err(e) => {
            eprintln!("❌ Failed to load graph for new client: {}", e);
        }
    }
    
    // Spawn two tasks:
    // 1. Forward broadcast messages to this client
    // 2. Handle incoming messages from this client
    
    // Task 1: Forward broadcasts to client
    let send_task = tokio::spawn(async move {
        while let Ok(event) = broadcast_rx.recv().await {
            let json = serde_json::to_string(&event).unwrap();
            if sender.send(Message::Text(json)).await.is_err() {
                break; // Client disconnected
            }
        }
    });
    
    // Task 2: Handle client messages (just ping/pong and close for now)
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Close(_)) => {
                    println!("🔌 Client requested close");
                    break;
                }
                Ok(Message::Ping(data)) => {
                    // Pong is handled automatically by axum
                    println!("🏓 Received ping: {:?}", data);
                }
                Ok(Message::Text(text)) => {
                    println!("📥 Received message: {}", text);
                    // Could handle commands like "subscribe" here
                }
                Err(e) => {
                    eprintln!("❌ WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });
    
    // Wait for either task to complete (client disconnect)
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
    
    println!("🔌 WebSocket client disconnected");
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

#[tokio::main]
async fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    // Default values
    let mut db_path = PathBuf::from("spider_graph.db");
    let mut port: u16 = 3000;
    
    // Parse --db and --port arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--db" => {
                if i + 1 < args.len() {
                    db_path = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("Error: --db requires a path argument");
                    std::process::exit(1);
                }
            }
            "--port" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or(3000);
                    i += 2;
                } else {
                    eprintln!("Error: --port requires a number");
                    std::process::exit(1);
                }
            }
            _ => {
                i += 1;
            }
        }
    }
    
    println!("🕸️  Spider Graph Visualizer Server");
    println!("   Database: {:?}", db_path);
    println!("   Port: {}", port);
    
    // Create application state
    let state = AppState::new(db_path);
    
    // Pre-load the graph into cache
    if let Err(e) = state.load_graph_data().await {
        eprintln!("⚠️  Warning: Could not load initial graph: {}", e);
        eprintln!("   Server will start anyway - create the database and call /api/notify");
    } else {
        // Cache it
        let data = state.load_graph_data().await.unwrap();
        let mut cache = state.graph_cache.write().await;
        *cache = Some(data);
        println!("✅ Graph loaded into cache");
    }
    
    // Build the router
    let app = Router::new()
        // REST API endpoints
        .route("/api/graph", get(get_graph))
        .route("/api/stats", get(get_stats))
        .route("/api/clusters", get(get_clusters))
        .route("/api/notify", axum::routing::post(notify_update))
        // WebSocket endpoint
        .route("/ws", get(ws_handler))
        // Serve static files from ./frontend directory
        .nest_service("/", ServeDir::new("frontend").append_index_html_on_directories(true))
        // Add CORS support for development
        .layer(CorsLayer::permissive())
        // Attach shared state
        .with_state(state);
    
    // Start the server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("🚀 Server running at http://localhost:{}", port);
    println!("   Open in browser to view graph visualization");
    println!("   WebSocket endpoint: ws://localhost:{}/ws", port);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
