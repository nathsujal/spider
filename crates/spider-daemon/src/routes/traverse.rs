use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use spider_core::{
    db::nodes::NodeId,
    db::rels::Direction,
    query::traverse::{self, Neighbor},
    schema::token::TokenId,
};

use crate::routes::nodes::NodeLabel;
use crate::server::AppState;

// ── Query parameters ─────────────────────────────────────────────────────────

/// Query parameters for `GET /traverse/:node_id`.
#[derive(Debug, Deserialize)]
pub struct TraverseQuery {
    /// Traversal direction: `outgoing`, `incoming`, or `both` (default: `outgoing`).
    pub direction: Option<String>,
    /// Maximum depth for multi-hop traversal (default: 1, max: 3).
    pub depth: Option<u8>,
}

// ── Response types ────────────────────────────────────────────────────────────

/// A neighbor node in the traversal result.
#[derive(Serialize, Debug)]
pub struct TraverseNeighbor {
    pub id: u32,
    pub labels: Vec<NodeLabel>,
    pub edge_id: u32,
    pub edge_type: String,
    pub distance: u8,
}

/// Response payload for `GET /traverse/:node_id`.
#[derive(Serialize, Debug)]
pub struct TraverseResponse {
    pub node_id: u32,
    pub neighbors: Vec<TraverseNeighbor>,
    pub total: usize,
    pub direction: String,
    pub depth: u8,
}

// ── Handler ──────────────────────────────────────────────────────────────────

/// `GET /traverse/:node_id?direction=outgoing&depth=1`
///
/// Traverses the graph starting from the given node. Returns neighbors
/// with their edge types and traversal distance.
///
/// Supports `direction` param: `outgoing`, `incoming`, `both` (default: `outgoing`).
/// Supports `depth` for multi-hop traversal (default: 1, max: 3).
pub async fn handler(
    State(state): State<AppState>,
    Path(node_id): Path<u32>,
    Query(params): Query<TraverseQuery>,
) -> Result<Json<TraverseResponse>, (StatusCode, &'static str)> {
    if node_id == 0 {
        return Err((StatusCode::BAD_REQUEST, "node ID must be non-zero"));
    }

    let direction = parse_direction(params.direction.as_deref())?;
    let depth = params.depth.unwrap_or(1).min(3);

    let mut db = state.db.lock().await;

    let node_id_obj = NodeId::new(node_id)
        .map_err(|_| (StatusCode::NOT_FOUND, "node not found"))?;

    let node = db.nodes.get(node_id - 1)
        .map_err(|_| (StatusCode::NOT_FOUND, "node not found"))?;

    if node.is_deleted() {
        return Err((StatusCode::NOT_FOUND, "node not found"));
    }

    if depth == 1 {
        // Single-hop: direct call to spider-core.
        let neighbors = traverse::get_neighbors(&mut db, node_id_obj, direction)
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "traversal failed"))?;

        let entries: Vec<TraverseNeighbor> = neighbors
            .iter()
            .filter_map(|n| build_neighbor(&mut db, n, 1))
            .collect();

        let total = entries.len();
        let dir_str = direction_str(direction);
        Ok(Json(TraverseResponse {
            node_id,
            neighbors: entries,
            total,
            direction: dir_str.to_string(),
            depth: 1,
        }))
    } else {
        // Multi-hop: iterative BFS.
        let mut all_neighbors: Vec<TraverseNeighbor> = Vec::new();
        let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
        visited.insert(node_id);

        let mut frontier: Vec<Neighbor> = traverse::get_neighbors(&mut db, node_id_obj, direction)
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "traversal failed"))?;

        for neighbor in &frontier {
            visited.insert(neighbor.node_id.get());
        }

        for n in &frontier {
            if let Some(entry) = build_neighbor(&mut db, n, 1) {
                all_neighbors.push(entry);
            }
        }

        for d in 2..=depth {
            let mut next_frontier: Vec<Neighbor> = Vec::new();
            for current in &frontier {
                let current_id = match NodeId::new(current.node_id.get()) {
                    Ok(id) => id,
                    Err(_) => continue,
                };
                let neighbors = match traverse::get_neighbors(&mut db, current_id, direction) {
                    Ok(ns) => ns,
                    Err(_) => continue,
                };
                for n in neighbors {
                    if !visited.contains(&n.node_id.get()) {
                        visited.insert(n.node_id.get());
                        next_frontier.push(n);
                    }
                }
            }
            for n in &next_frontier {
                if let Some(entry) = build_neighbor(&mut db, n, d) {
                    all_neighbors.push(entry);
                }
            }
            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }

        let dir_str = direction_str(direction);
        let total = all_neighbors.len();
        Ok(Json(TraverseResponse {
            node_id,
            neighbors: all_neighbors,
            total,
            direction: dir_str.to_string(),
            depth,
        }))
    }
}

fn build_neighbor(db: &mut spider_core::db::lifecycle::Spider, n: &Neighbor, distance: u8) -> Option<TraverseNeighbor> {
    let node = db.nodes.get(n.node_id.get() - 1).ok()?;
    if node.is_deleted() {
        return None;
    }

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

    let edge = db.edges.get(n.edge_id.get() - 1).ok()?;
    let edge_tid = match edge.edge_type() {
        Some(t) => t,
        None => return None,
    };
    let edge_type = db.edge_type_tokens.get_name(
        TokenId::new(edge_tid.get()).ok()?
    )?.to_string();

    Some(TraverseNeighbor {
        id: n.node_id.get(),
        labels,
        edge_id: n.edge_id.get(),
        edge_type,
        distance,
    })
}

fn parse_direction(s: Option<&str>) -> Result<Direction, (StatusCode, &'static str)> {
    match s {
        None | Some("outgoing") => Ok(Direction::Outgoing),
        Some("incoming") => Ok(Direction::Incoming),
        Some("both") => Ok(Direction::Both),
        _ => Err((StatusCode::BAD_REQUEST, "direction must be one of: outgoing, incoming, both")),
    }
}

fn direction_str(d: Direction) -> &'static str {
    match d {
        Direction::Outgoing => "outgoing",
        Direction::Incoming => "incoming",
        Direction::Both => "both",
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use spider_core::db::ingest::{Entity, IngestRequest, Proposition};
    use spider_core::db::lifecycle::Spider;

    #[tokio::test]
    async fn traverse_returns_neighbors() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Mumbai is in India",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                        Entity { name: "India", entity_type: "LOCATION" },
                    ],
                },
            ],
        };
        let ingest_result = spider_core::db::ingest::index(&mut db, &req).unwrap();

        let state = AppState::test(db);

        // Traverse from the document node — should find the proposition.
        let resp = handler(
            State(state),
            Path(ingest_result.document_id.get()),
            Query(TraverseQuery {
                direction: Some("outgoing".into()),
                depth: Some(1),
            }),
        )
        .await
        .unwrap();

        assert_eq!(resp.node_id, ingest_result.document_id.get());
        assert_eq!(resp.neighbors.len(), 1);
        assert_eq!(resp.direction, "outgoing");
        assert_eq!(resp.depth, 1);
    }

    #[tokio::test]
    async fn traverse_multi_hop() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Mumbai is in India",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                    ],
                },
            ],
        };
        spider_core::db::ingest::index(&mut db, &req).unwrap();

        let state = AppState::test(db);

        // Traverse from document with depth 2 — should reach proposition + entity.
        let resp = handler(
            State(state),
            Path(1),
            Query(TraverseQuery {
                direction: Some("outgoing".into()),
                depth: Some(2),
            }),
        )
        .await
        .unwrap();

        assert_eq!(resp.depth, 2);
        // 1 proposition (distance 1) + 1 entity (distance 2)
        assert_eq!(resp.neighbors.len(), 2);

        let distances: Vec<u8> = resp.neighbors.iter().map(|n| n.distance).collect();
        assert!(distances.contains(&1));
        assert!(distances.contains(&2));
    }

    #[tokio::test]
    async fn traverse_nonexistent_node_returns_404() {
        let dir = tempfile::tempdir().unwrap();
        let db = Spider::open(dir.path()).unwrap();
        let state = AppState::test(db);

        let result = handler(
            State(state),
            Path(999),
            Query(TraverseQuery {
                direction: None,
                depth: None,
            }),
        )
        .await;

        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn traverse_invalid_direction_returns_400() {
        let dir = tempfile::tempdir().unwrap();
        let db = Spider::open(dir.path()).unwrap();
        let state = AppState::test(db);

        let result = handler(
            State(state),
            Path(1),
            Query(TraverseQuery {
                direction: Some("invalid".into()),
                depth: None,
            }),
        )
        .await;

        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn traverse_incoming_edges() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Hello",
                    entities: vec![
                        Entity { name: "X", entity_type: "CONCEPT" },
                    ],
                },
            ],
        };
        spider_core::db::ingest::index(&mut db, &req).unwrap();

        // Find the entity node first.
        let entity_nodes = spider_core::db::find::find_by_property(&mut db, "name", "X").unwrap();
        assert_eq!(entity_nodes.len(), 1);
        let entity_id = entity_nodes[0].get();

        let state = AppState::test(db);

        let resp = handler(
            State(state),
            Path(entity_id),
            Query(TraverseQuery {
                direction: Some("incoming".into()),
                depth: Some(1),
            }),
        )
        .await
        .unwrap();

        assert_eq!(resp.neighbors.len(), 1);
        assert_eq!(resp.direction, "incoming");
    }

    #[tokio::test]
    async fn traverse_neighbor_has_labels_and_edge_type() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Hello",
                    entities: vec![
                        Entity { name: "X", entity_type: "CONCEPT" },
                    ],
                },
            ],
        };
        spider_core::db::ingest::index(&mut db, &req).unwrap();

        let state = AppState::test(db);

        // Document → Proposition
        let resp = handler(
            State(state),
            Path(1),
            Query(TraverseQuery { direction: Some("outgoing".into()), depth: Some(1) }),
        )
        .await
        .unwrap();

        assert_eq!(resp.neighbors.len(), 1);
        let neighbor = &resp.neighbors[0];
        assert!(!neighbor.labels.is_empty());
        assert_eq!(neighbor.edge_type, "CONTAINS");
        assert_eq!(neighbor.distance, 1);
    }
}
