use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Serialize;
use spider_core::{
    bio::{score, tier::BioTier},
    db::lifecycle::Spider,
    schema::node::Node,
    schema::token::TokenId,
};

use crate::server::AppState;

// ── Response types ────────────────────────────────────────────────────────────

/// Serializable node label resolved from a token ID.
#[derive(Serialize, Debug, Clone)]
pub struct NodeLabel {
    pub id: u8,
    pub name: String,
}

/// Response payload for `GET /nodes/:id`.
#[derive(Serialize, Debug)]
pub struct NodeResponse {
    pub id: u32,
    pub labels: Vec<NodeLabel>,
    pub access_count: u32,
    pub created_at: u32,
    pub last_accessed_at: u32,
    pub significance: u8,
    pub bio_score: f64,
    pub bio_tier: String,
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert a `Node` + `TokenStore` into a `NodeResponse`.
///
/// The bio score is computed using the default bio params since we don't have
/// access to the metadata weights through this layer.
fn node_to_response(node: &Node, db: &Spider) -> NodeResponse {
    let bio_score = score::calculate(node);
    let bio_tier = BioTier::from_score(bio_score);

    let labels: Vec<NodeLabel> = node
        .labels()
        .into_iter()
        .filter_map(|label_opt| {
            let label = label_opt?;
            let raw = label.get();
            let tid = TokenId::new(raw).ok()?;
            let name = db.label_tokens.get_name(tid)?.to_string();
            Some(NodeLabel { id: raw, name })
        })
        .collect();

    NodeResponse {
        id: node.id,
        labels,
        access_count: node.access_count,
        created_at: node.created_at,
        last_accessed_at: node.last_accessed_at,
        significance: node.significance,
        bio_score,
        bio_tier: bio_tier.to_string(),
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// `GET /nodes/:id` — fetch a single node by its ID.
///
/// Returns 404 if the node doesn't exist or is a deleted tombstone.
pub async fn get_node(
    State(state): State<AppState>,
    Path(id): Path<u32>,
) -> Result<Json<NodeResponse>, (StatusCode, &'static str)> {
    if id == 0 {
        return Err((StatusCode::NOT_FOUND, "node not found"));
    }

    let mut db = state.db.lock().await;

    // Node IDs are 1-based, record indices are 0-based.
    let node = db.nodes.get(id - 1).map_err(|_| (StatusCode::NOT_FOUND, "node not found"))?;

    if node.is_deleted() {
        return Err((StatusCode::NOT_FOUND, "node not found"));
    }

    Ok(Json(node_to_response(&node, &db)))
}

// ── List nodes ────────────────────────────────────────────────────────────────

/// Query parameters for `GET /nodes`.
#[derive(Debug, serde::Deserialize)]
pub struct NodeListQuery {
    /// Maximum number of nodes to return (default 100, max 1000).
    #[serde(default = "default_limit")]
    pub limit: u32,
    /// Number of nodes to skip (default 0).
    #[serde(default)]
    pub offset: u32,
}

fn default_limit() -> u32 {
    100
}

/// Response payload for `GET /nodes`.
#[derive(Serialize, Debug)]
pub struct NodeListResponse {
    pub nodes: Vec<NodeResponse>,
    /// Total number of live (non-deleted) nodes in the database.
    pub total: u32,
    /// Limit used for this request.
    pub limit: u32,
    /// Offset used for this request.
    pub offset: u32,
}

/// Iterate all node records, collecting live nodes with pagination.
///
/// The RecordFile has no `len()` method, so we read sequentially until EOF.
/// Tombstoned nodes (id == 0) are skipped. This is O(n) in total nodes.
fn list_nodes_inner(
    db: &mut Spider,
    limit: u32,
    offset: u32,
) -> NodeListResponse {
    let mut result = NodeListResponse {
        nodes: Vec::new(),
        total: 0,
        limit,
        offset,
    };

    let mut index: u32 = 0;
    let mut seen_live: u32 = 0;

    loop {
        let node = match db.nodes.get(index) {
            Ok(n) => n,
            Err(_) => break, // EOF or read error — stop iteration
        };

        if !node.is_deleted() {
            result.total += 1;
            if seen_live >= offset && result.nodes.len() < limit as usize {
                result.nodes.push(node_to_response(&node, db));
            }
            seen_live += 1;
        }

        index += 1;
    }

    result
}

/// `GET /nodes?limit=100&offset=0` — list all live nodes with pagination.
///
/// Iterates all records in the node file, skipping tombstones.
/// Returns a paginated subset with the total count.
pub async fn list_nodes(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<NodeListQuery>,
) -> Json<NodeListResponse> {
    let limit = params.limit.min(1000);
    let mut db = state.db.lock().await;
    Json(list_nodes_inner(&mut db, limit, params.offset))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use spider_core::db::lifecycle::Spider;
    use spider_core::schema::node::{LabelId, Node};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::sync::Mutex;

    fn now_secs() -> u32 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32
    }

    fn make_state() -> (AppState, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        // Intern a label so we can test resolution.
        let lid = db.label_tokens.get_or_create("Document").unwrap();

        let now = now_secs();
        let node = Node::new(1, &[LabelId::new(lid.get()).unwrap()], now, None).unwrap();
        db.nodes.append(&[node]).unwrap();

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };
        (state, dir)
    }

    #[tokio::test]
    async fn get_existing_node() {
        let (state, _dir) = make_state();

        let resp = get_node(State(state), Path(1)).await.unwrap();
        assert_eq!(resp.id, 1);
        assert_eq!(resp.labels.len(), 1);
        assert_eq!(resp.labels[0].name, "Document");
        assert!(resp.bio_score > 0.0);
        assert_eq!(resp.significance, 128);
    }

    #[tokio::test]
    async fn get_nonexistent_node_returns_404() {
        let (state, _dir) = make_state();

        let result = get_node(State(state), Path(999)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_zero_id_returns_404() {
        let (state, _dir) = make_state();

        let result = get_node(State(state), Path(0)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn list_nodes_returns_all_live() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let lid = db.label_tokens.get_or_create("Document").unwrap();
        let now = now_secs();
        let label = LabelId::new(lid.get()).unwrap();

        // Create 5 nodes.
        for id in 1..=5 {
            let node = Node::new(id, &[label], now, None).unwrap();
            db.nodes.append(&[node]).unwrap();
        }

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        let mut guard = state.db.lock().await;
        let resp = list_nodes_inner(&mut guard, 100, 0);
        assert_eq!(resp.total, 5);
        assert_eq!(resp.nodes.len(), 5);
    }

    #[tokio::test]
    async fn list_nodes_skips_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let lid = db.label_tokens.get_or_create("Document").unwrap();
        let now = now_secs();
        let label = LabelId::new(lid.get()).unwrap();

        // Create 3 nodes.
        for id in 1..=3 {
            let node = Node::new(id, &[label], now, None).unwrap();
            db.nodes.append(&[node]).unwrap();
        }

        // Delete node 2 by writing a tombstone at index 1.
        let tombstone = Node::empty();
        db.nodes.set(1, &tombstone).unwrap();

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        let mut guard = state.db.lock().await;
        let resp = list_nodes_inner(&mut guard, 100, 0);
        assert_eq!(resp.total, 2);
        assert_eq!(resp.nodes.len(), 2);
        // Node IDs in the result should be 1 and 3, not 2.
        let ids: Vec<u32> = resp.nodes.iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[tokio::test]
    async fn list_nodes_pagination_offset() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let lid = db.label_tokens.get_or_create("Document").unwrap();
        let now = now_secs();
        let label = LabelId::new(lid.get()).unwrap();

        for id in 1..=10 {
            let node = Node::new(id, &[label], now, None).unwrap();
            db.nodes.append(&[node]).unwrap();
        }

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        let mut guard = state.db.lock().await;
        // Offset 5, limit 3 → nodes 6, 7, 8
        let resp = list_nodes_inner(&mut guard, 3, 5);
        assert_eq!(resp.total, 10);
        assert_eq!(resp.nodes.len(), 3);
        let ids: Vec<u32> = resp.nodes.iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![6, 7, 8]);
    }

    #[tokio::test]
    async fn list_nodes_empty_database() {
        let dir = tempfile::tempdir().unwrap();
        let db = Spider::open(dir.path()).unwrap();

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        let mut guard = state.db.lock().await;
        let resp = list_nodes_inner(&mut guard, 100, 0);
        assert_eq!(resp.total, 0);
        assert_eq!(resp.nodes.len(), 0);
    }
}
