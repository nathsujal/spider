use serde::Serialize;

use crate::server::AppState;

/// Query parameters for `GET /bio`.
#[derive(Debug, serde::Deserialize)]
pub struct BioQuery {
    /// Maximum number of nodes to return (default 20, max 500).
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    20
}

/// Response payload for `GET /bio`.
#[derive(Serialize, Debug)]
pub struct BioResponse {
    /// Nodes sorted by bio_score descending.
    pub nodes: Vec<BioEntry>,
    /// Total number of live nodes in the database.
    pub total: u32,
}

/// A single entry in the vitality leaderboard.
#[derive(Serialize, Debug)]
pub struct BioEntry {
    pub id: u32,
    pub labels: Vec<crate::routes::nodes::NodeLabel>,
    pub bio_score: f64,
    pub bio_tier: String,
    pub significance: u8,
    pub access_count: u32,
}

/// `GET /bio?limit=20` — vitality leaderboard sorted by bio score descending.
///
/// Fetches all live nodes, computes bio_score for each, sorts by score,
/// and returns the top `limit` entries.
pub async fn handler(
    axum::extract::Query(params): axum::extract::Query<BioQuery>,
    axum::extract::State(state): axum::extract::State<AppState>,
) -> axum::Json<BioResponse> {
    let limit = params.limit.min(500);
    let mut db = state.db.lock().await;

    // Collect all live nodes.
    let mut entries: Vec<BioEntry> = Vec::new();
    let mut index: u32 = 0;
    let mut total: u32 = 0;

    loop {
        let node = match db.nodes.get(index) {
            Ok(n) => n,
            Err(_) => break,
        };

        if !node.is_deleted() {
            total += 1;
            let score = spider_core::bio::score::calculate(&node);
            let tier = spider_core::bio::tier::BioTier::from_score(score);

            let labels: Vec<crate::routes::nodes::NodeLabel> = node
                .labels()
                .into_iter()
                .filter_map(|label_opt| {
                    let label = label_opt?;
                    let raw = label.get();
                    let tid = spider_core::schema::token::TokenId::new(raw).ok()?;
                    let name = db.label_tokens.get_name(tid)?.to_string();
                    Some(crate::routes::nodes::NodeLabel { id: raw, name })
                })
                .collect();

            entries.push(BioEntry {
                id: node.id,
                labels,
                bio_score: score,
                bio_tier: tier.to_string(),
                significance: node.significance,
                access_count: node.access_count,
            });
        }

        index += 1;
    }

    // Sort by bio_score descending.
    entries.sort_by(|a, b| b.bio_score.partial_cmp(&a.bio_score).unwrap_or(std::cmp::Ordering::Equal));

    // Take top `limit`.
    entries.truncate(limit as usize);

    axum::Json(BioResponse { nodes: entries, total })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use spider_core::db::lifecycle::Spider;
    use spider_core::schema::node::{LabelId, Node};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn now_secs() -> u32 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32
    }

    #[tokio::test]
    async fn bio_returns_sorted_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let lid_doc = db.label_tokens.get_or_create("Document").unwrap();
        let lid_prop = db.label_tokens.get_or_create("Proposition").unwrap();
        let now = now_secs();

        // Node 1: low significance, no extra access.
        let n1 = Node::new(1, &[LabelId::new(lid_doc.get()).unwrap()], now, Some(50)).unwrap();
        db.nodes.append(&[n1]).unwrap();

        // Node 2: high significance.
        let n2 = Node::new(2, &[LabelId::new(lid_doc.get()).unwrap()], now, Some(200)).unwrap();
        db.nodes.append(&[n2]).unwrap();

        // Node 3: medium significance but accessed many times (higher freq boost).
        let mut n3 = Node::new(3, &[LabelId::new(lid_prop.get()).unwrap()], now, Some(100)).unwrap();
        n3.access_count = 100;
        db.nodes.append(&[n3]).unwrap();

        let state = AppState::test(db);

        let resp = handler(
            axum::extract::Query(BioQuery { limit: 10 }),
            axum::extract::State(state.clone()),
        )
        .await;

        assert_eq!(resp.nodes.len(), 3);
        assert_eq!(resp.total, 3);

        // Node 2 (high sig) should be first, node 3 (freq boost) second, node 1 last.
        assert_eq!(resp.nodes[0].id, 2);
        assert_eq!(resp.nodes[1].id, 3);
        assert_eq!(resp.nodes[2].id, 1);

        // Scores should be strictly decreasing.
        assert!(resp.nodes[0].bio_score > resp.nodes[1].bio_score);
        assert!(resp.nodes[1].bio_score > resp.nodes[2].bio_score);
    }

    #[tokio::test]
    async fn bio_respects_limit() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let lid = db.label_tokens.get_or_create("Document").unwrap();
        let now = now_secs();
        let label = LabelId::new(lid.get()).unwrap();

        for id in 1..=10 {
            let node = Node::new(id, &[label], now, None).unwrap();
            db.nodes.append(&[node]).unwrap();
        }

        let state = AppState::test(db);

        let resp = handler(
            axum::extract::Query(BioQuery { limit: 3 }),
            axum::extract::State(state),
        )
        .await;

        assert_eq!(resp.nodes.len(), 3);
        assert_eq!(resp.total, 10);
    }

    #[tokio::test]
    async fn bio_empty_database() {
        let dir = tempfile::tempdir().unwrap();
        let db = Spider::open(dir.path()).unwrap();

        let state = AppState::test(db);

        let resp = handler(
            axum::extract::Query(BioQuery { limit: 20 }),
            axum::extract::State(state),
        )
        .await;

        assert_eq!(resp.nodes.len(), 0);
        assert_eq!(resp.total, 0);
    }

    #[tokio::test]
    async fn bio_skips_deleted_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let lid = db.label_tokens.get_or_create("Document").unwrap();
        let now = now_secs();
        let label = LabelId::new(lid.get()).unwrap();

        for id in 1..=5 {
            let node = Node::new(id, &[label], now, None).unwrap();
            db.nodes.append(&[node]).unwrap();
        }

        // Delete node 3 (index 2).
        db.nodes.set(2, &Node::empty()).unwrap();

        let state = AppState::test(db);

        let resp = handler(
            axum::extract::Query(BioQuery { limit: 20 }),
            axum::extract::State(state),
        )
        .await;

        assert_eq!(resp.total, 4);
        assert_eq!(resp.nodes.len(), 4);
        let ids: Vec<u32> = resp.nodes.iter().map(|n| n.id).collect();
        assert!(!ids.contains(&3));
    }
}
