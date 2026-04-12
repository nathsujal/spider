use serde::{Deserialize, Serialize};
use spider_core::{
    bio::{score, tier::BioTier},
    db::{find, lifecycle::Spider},
    property,
    db::nodes::NodeId,
};

use crate::routes::nodes::NodeLabel;
use crate::server::AppState;

// ── Query parameters ─────────────────────────────────────────────────────────

/// Query parameters for `GET /search`.
///
/// At least one of `label` or (`property_key` + `property_value`) must be
/// provided.
#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    /// Find nodes with this label (e.g. "Document").
    pub label: Option<String>,
    /// Find nodes with this property key (e.g. "name").
    pub property_key: Option<String>,
    /// Find nodes with this property value (e.g. "Mumbai").
    pub property_value: Option<String>,
}

// ── Response types ────────────────────────────────────────────────────────────

/// A single property in the search result.
#[derive(Serialize, Debug)]
pub struct PropEntry {
    pub key: String,
    pub value: String,
}

/// A node in the search result.
#[derive(Serialize, Debug)]
pub struct SearchResultEntry {
    pub id: u32,
    pub labels: Vec<NodeLabel>,
    pub properties: Vec<PropEntry>,
    pub bio_score: f64,
    pub bio_tier: String,
}

/// Response payload for `GET /search`.
#[derive(Serialize, Debug)]
pub struct SearchResponse {
    pub results: Vec<SearchResultEntry>,
    pub total: u32,
}

// ── Handler ──────────────────────────────────────────────────────────────────

/// `GET /search?label=Document` or `GET /search?property_key=name&property_value=Mumbai`
///
/// Searches for nodes by label or property key/value. At least one query
/// parameter is required. Returns 400 if no valid search criteria provided.
pub async fn handler(
    axum::extract::Query(params): axum::extract::Query<SearchQuery>,
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Result<axum::Json<SearchResponse>, (axum::http::StatusCode, &'static str)> {
    // Validate that at least one search criterion is provided.
    let has_label = params.label.as_ref().is_some_and(|s| !s.is_empty());
    let has_property = params.property_key.as_ref().is_some_and(|k| !k.is_empty())
        && params.property_value.as_ref().is_some_and(|v| !v.is_empty());

    if !has_label && !has_property {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "provide at least one of: label, property_key+property_value",
        ));
    }

    let mut db = state.db.lock().await;

    // Collect matching node IDs.
    let node_ids = match (&params.label, &params.property_key, &params.property_value) {
        (Some(label), None, None) => {
            find::find_by_label(&mut db, label)
                .map_err(|_| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "database error"))?
        }
        (None, Some(key), Some(val)) => {
            find::find_by_property(&mut db, key, val)
                .map_err(|_| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "database error"))?
        }
        (Some(label), Some(key), Some(val)) => {
            // Intersection: nodes matching both label AND property.
            let label_ids: Vec<u32> = find::find_by_label(&mut db, label)
                .map_err(|_| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "database error"))?
                .into_iter()
                .map(|id| id.get())
                .collect();
            let prop_ids: Vec<u32> = find::find_by_property(&mut db, key, val)
                .map_err(|_| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "database error"))?
                .into_iter()
                .map(|id| id.get())
                .collect();

            // Simple set intersection.
            let mut set = label_ids.into_iter().collect::<std::collections::HashSet<_>>();
            set.retain(|id| prop_ids.contains(id));

            set.into_iter()
                .map(|id| NodeId::new(id).unwrap())
                .collect()
        }
        _ => return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "property_value requires property_key",
        )),
    };

    // Build full result entries.
    let results: Vec<SearchResultEntry> = node_ids
        .iter()
        .filter_map(|node_id| {
            let node = db.nodes.get(node_id.get() - 1).ok()?;
            if node.is_deleted() {
                return None;
            }
            Some(build_entry(node, &mut db))
        })
        .collect();

    let total = results.len() as u32;

    Ok(axum::Json(SearchResponse { results, total }))
}

fn build_entry(
    node: spider_core::schema::node::Node,
    db: &mut Spider,
) -> SearchResultEntry {
    let bio_score = score::calculate(&node);
    let bio_tier = BioTier::from_score(bio_score);

    let labels: Vec<NodeLabel> = node
        .labels()
        .into_iter()
        .filter_map(|label_opt| {
            let label = label_opt?;
            let raw = label.get();
            let tid = spider_core::schema::token::TokenId::new(raw).ok()?;
            let name = db.label_tokens.get_name(tid)?.to_string();
            Some(NodeLabel { id: raw, name })
        })
        .collect();

    let node_id = NodeId::new(node.id).unwrap();
    let properties = property::list_all(db, node_id)
        .ok()
        .unwrap_or_default()
        .into_iter()
        .map(|entry| PropEntry {
            key: entry.key,
            value: entry.value.to_string(),
        })
        .collect();

    SearchResultEntry {
        id: node.id,
        labels,
        properties,
        bio_score,
        bio_tier: bio_tier.to_string(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use spider_core::db::lifecycle::Spider;
    use spider_core::db::ingest;
    use spider_core::db::ingest::{Entity, IngestRequest, Proposition};
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

    #[tokio::test]
    async fn search_by_label_returns_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let lid_doc = db.label_tokens.get_or_create("DOCUMENT").unwrap();
        let lid_prop = db.label_tokens.get_or_create("PROPOSITION").unwrap();
        let now = now_secs();

        // Create 2 DOCUMENT nodes.
        let d1 = Node::new(1, &[LabelId::new(lid_doc.get()).unwrap()], now, None).unwrap();
        let d2 = Node::new(2, &[LabelId::new(lid_doc.get()).unwrap()], now, None).unwrap();
        db.nodes.append(&[d1, d2]).unwrap();

        // Create 1 PROPOSITION node.
        let p1 = Node::new(3, &[LabelId::new(lid_prop.get()).unwrap()], now, None).unwrap();
        db.nodes.append(&[p1]).unwrap();

        // find_by_label uses metadata.next_node_id as the scan upper bound.
        // Manual node creation doesn't update it, so we must set it ourselves.
        db.metadata.next_node_id = 4;

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        let resp = handler(
            axum::extract::Query(SearchQuery {
                label: Some("DOCUMENT".to_string()),
                property_key: None,
                property_value: None,
            }),
            axum::extract::State(state),
        )
        .await
        .unwrap();

        assert_eq!(resp.total, 2);
        assert_eq!(resp.results.len(), 2);
    }

    #[tokio::test]
    async fn search_by_label_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let lid = db.label_tokens.get_or_create("PERSON").unwrap();
        let now = now_secs();
        let node = Node::new(1, &[LabelId::new(lid.get()).unwrap()], now, None).unwrap();
        db.nodes.append(&[node]).unwrap();
        db.metadata.next_node_id = 2;

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        let resp = handler(
            axum::extract::Query(SearchQuery {
                label: Some("DOCUMENT".to_string()),
                property_key: None,
                property_value: None,
            }),
            axum::extract::State(state),
        )
        .await
        .unwrap();

        assert_eq!(resp.total, 0);
        assert!(resp.results.is_empty());
    }

    #[tokio::test]
    async fn search_by_property_returns_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let req = IngestRequest {
            title: "Test Document",
            propositions: vec![
                Proposition {
                    text: "Mumbai is a large city",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                    ],
                },
            ],
        };
        ingest::index(&mut db, &req).unwrap();

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        let resp = handler(
            axum::extract::Query(SearchQuery {
                label: None,
                property_key: Some("name".to_string()),
                property_value: Some("Mumbai".to_string()),
            }),
            axum::extract::State(state),
        )
        .await
        .unwrap();

        assert_eq!(resp.total, 1);
        assert_eq!(resp.results.len(), 1);
    }

    #[tokio::test]
    async fn search_by_property_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![Proposition {
                text: "Hello",
                entities: vec![],
            }],
        };
        ingest::index(&mut db, &req).unwrap();

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        let resp = handler(
            axum::extract::Query(SearchQuery {
                label: None,
                property_key: Some("name".to_string()),
                property_value: Some("nonexistent".to_string()),
            }),
            axum::extract::State(state),
        )
        .await
        .unwrap();

        assert_eq!(resp.total, 0);
    }

    #[tokio::test]
    async fn search_no_params_returns_400() {
        let dir = tempfile::tempdir().unwrap();
        let db = Spider::open(dir.path()).unwrap();

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        let result = handler(
            axum::extract::Query(SearchQuery {
                label: None,
                property_key: None,
                property_value: None,
            }),
            axum::extract::State(state),
        )
        .await;

        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn search_combined_label_and_property() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let req = IngestRequest {
            title: "Test Doc",
            propositions: vec![
                Proposition {
                    text: "Facts about cities",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                        Entity { name: "Paris", entity_type: "LOCATION" },
                    ],
                },
            ],
        };
        ingest::index(&mut db, &req).unwrap();

        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        // Search for nodes with label ENTITY and property name=Mumbai.
        let resp = handler(
            axum::extract::Query(SearchQuery {
                label: Some("ENTITY".to_string()),
                property_key: Some("name".to_string()),
                property_value: Some("Mumbai".to_string()),
            }),
            axum::extract::State(state),
        )
        .await
        .unwrap();

        assert_eq!(resp.total, 1);
        assert_eq!(resp.results.len(), 1);
    }
}
