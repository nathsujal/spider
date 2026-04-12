//! End-to-end integration tests for spider-daemon.
//!
//! These tests start the actual axum router in-process (no network ports)
//! and make HTTP requests against it using `tower::ServiceExt`.
//! This is faster and more reliable than binding to a real port.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use serde_json::Value;
use spider_core::db::{
    ingest::{self, Entity, IngestRequest, Proposition},
    lifecycle::Spider,
};
use tower::ServiceExt;

use spider_daemon::server::{self, AppState};
use spider_daemon::ws::Broadcaster;
use spider_daemon::jobs::{JobQueue, Worker};

/// Build a test server with a real Spider DB at the given path.
/// Returns the router and a handle to the worker task (which will be dropped
/// when the test ends, causing the worker to stop).
fn make_server(db: Spider) -> (Router, tokio::task::JoinHandle<()>) {
    let (queue, rx) = JobQueue::new();
    let queue = Arc::new(queue);
    let db = Arc::new(tokio::sync::Mutex::new(db));
    let broadcaster = Arc::new(Broadcaster::new());

    // Start the worker in the background.
    let worker = Worker::new(
        rx,
        Arc::clone(&queue),
        Arc::clone(&broadcaster),
        Arc::clone(&db),
    );
    let worker_handle = tokio::spawn(worker.run());

    let state = AppState {
        db,
        broadcaster,
        job_queue: queue,
    };
    let router = server::build_router(state);
    (router, worker_handle)
}

/// Build a test server with a pre-seeded database.
fn make_server_seeded() -> (Router, tempfile::TempDir, tokio::task::JoinHandle<()>) {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Spider::open(dir.path()).unwrap();

    // Ingest a document with entities to populate the graph.
    let req = IngestRequest {
        title: "Integration Test Document",
        propositions: vec![
            Proposition {
                text: "Mumbai is the financial capital of India",
                entities: vec![
                    Entity { name: "Mumbai", entity_type: "LOCATION" },
                    Entity { name: "India", entity_type: "LOCATION" },
                ],
            },
            Proposition {
                text: "Delhi is the political capital of India",
                entities: vec![
                    Entity { name: "Delhi", entity_type: "LOCATION" },
                    Entity { name: "India", entity_type: "LOCATION" },
                ],
            },
        ],
    };
    let ingest_result = ingest::index(&mut db, &req).unwrap();
    assert!(ingest_result.document_id.get() >= 1);

    let (router, worker_handle) = make_server(db);
    (router, dir, worker_handle)
}

/// Extract JSON body as a serde_json::Value.
async fn json_body(response: axum::response::Response) -> Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

// ── Health ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integration_health() {
    let dir = tempfile::tempdir().unwrap();
    let db = Spider::open(dir.path()).unwrap();
    let (app, _worker) = make_server(db);

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let json = json_body(resp).await;
    assert_eq!(json["status"], "ok");
    assert!(json["db_path"].as_str().is_some());
}

// ── Node by ID ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integration_node_by_id() {
    let (app, _dir, _worker) = make_server_seeded();

    // Node 1 should be the Document node.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/nodes/1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let json = json_body(resp).await;
    assert_eq!(json["id"], 1);
    assert!(json["labels"].as_array().is_some());
    assert!(json["bio_score"].as_f64().is_some());
    assert_eq!(json["bio_tier"].as_str().unwrap(), "Hot");
}

#[tokio::test]
async fn integration_node_not_found() {
    let (app, _dir, _worker) = make_server_seeded();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/nodes/9999")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── List Nodes ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integration_list_nodes() {
    let (app, _dir, _worker) = make_server_seeded();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/nodes")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let json = json_body(resp).await;
    assert!(json["total"].as_u64().unwrap() > 0);
    assert!(json["nodes"].as_array().unwrap().len() > 0);
    assert_eq!(json["limit"].as_u64().unwrap(), 100);
    assert_eq!(json["offset"].as_u64().unwrap(), 0);
}

#[tokio::test]
async fn integration_list_nodes_pagination() {
    let (app, _dir, _worker) = make_server_seeded();

    // Request with limit=2.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/nodes?limit=2")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let json = json_body(resp).await;
    assert_eq!(json["nodes"].as_array().unwrap().len(), 2);
    assert_eq!(json["limit"].as_u64().unwrap(), 2);
    // total should be the full count, not just 2.
    assert!(json["total"].as_u64().unwrap() > 2);
}

#[tokio::test]
async fn integration_list_nodes_offset() {
    let (app, _dir, _worker) = make_server_seeded();

    let resp1 = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/nodes?limit=1&offset=0")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let json1 = json_body(resp1).await;
    let first_id = json1["nodes"][0]["id"].as_u64().unwrap();

    let resp2 = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/nodes?limit=1&offset=1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let json2 = json_body(resp2).await;
    let second_id = json2["nodes"][0]["id"].as_u64().unwrap();

    assert_ne!(first_id, second_id);
}

// ── Bio Leaderboard ──────────────────────────────────────────────────────────

#[tokio::test]
async fn integration_bio() {
    let (app, _dir, _worker) = make_server_seeded();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/bio?limit=5")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let json = json_body(resp).await;
    let nodes = json["nodes"].as_array().unwrap();
    assert!(nodes.len() > 0);
    assert!(nodes.len() <= 5);
    // total is the full count of live nodes, which may exceed the limit.
    assert!(json["total"].as_u64().unwrap() >= nodes.len() as u64);

    // Verify sorted descending by bio_score.
    for i in 1..nodes.len() {
        let prev = nodes[i - 1]["bio_score"].as_f64().unwrap();
        let curr = nodes[i]["bio_score"].as_f64().unwrap();
        assert!(prev >= curr, "bio scores should be sorted descending");
    }
}

// ── Search ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integration_search_by_label() {
    let (app, _dir, _worker) = make_server_seeded();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/search?label=DOCUMENT")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let json = json_body(resp).await;
    assert!(json["total"].as_u64().unwrap() >= 1);
    assert_eq!(json["results"].as_array().unwrap()[0]["id"].as_u64().unwrap(), 1);
}

#[tokio::test]
async fn integration_search_by_property() {
    let (app, _dir, _worker) = make_server_seeded();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/search?property_key=name&property_value=Mumbai")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let json = json_body(resp).await;
    assert_eq!(json["total"].as_u64().unwrap(), 1);
    assert_eq!(json["results"].as_array().unwrap()[0]["properties"].as_array().unwrap().len() > 0, true);
}

#[tokio::test]
async fn integration_search_no_params() {
    let (app, _dir, _worker) = make_server_seeded();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/search")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── Traverse ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integration_traverse() {
    let (app, _dir, _worker) = make_server_seeded();

    // Traverse from the Document node (id=1) — should find propositions.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/traverse/1?direction=outgoing&depth=1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let json = json_body(resp).await;
    assert!(json["total"].as_u64().unwrap() >= 1);
    assert_eq!(json["neighbors"].as_array().unwrap().len() as u64, json["total"].as_u64().unwrap());
    assert_eq!(json["direction"].as_str().unwrap(), "outgoing");
    assert_eq!(json["depth"].as_u64().unwrap(), 1);

    // First neighbor should be a PROPOSITION with edge_type CONTAINS.
    let neighbor = &json["neighbors"][0];
    assert_eq!(neighbor["edge_type"].as_str().unwrap(), "CONTAINS");
    assert_eq!(neighbor["distance"].as_u64().unwrap(), 1);
}

#[tokio::test]
async fn integration_traverse_multi_hop() {
    let (app, _dir, _worker) = make_server_seeded();

    // depth=2 from Document → should reach propositions (depth 1) + entities (depth 2).
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/traverse/1?direction=outgoing&depth=2")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let json = json_body(resp).await;
    assert_eq!(json["depth"].as_u64().unwrap(), 2);

    let neighbors = json["neighbors"].as_array().unwrap();
    // Should have propositions at distance 1 and entities at distance 2.
    let distances: Vec<u64> = neighbors.iter().map(|n| n["distance"].as_u64().unwrap()).collect();
    assert!(distances.contains(&1));
    assert!(distances.contains(&2));
}

#[tokio::test]
async fn integration_traverse_404() {
    let (app, _dir, _worker) = make_server_seeded();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/traverse/9999")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn integration_traverse_invalid_direction() {
    let (app, _dir, _worker) = make_server_seeded();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/traverse/1?direction=invalid")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ── Job Lifecycle ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn integration_job_lifecycle() {
    let (app, _dir, _worker) = make_server_seeded();

    // Submit a job via POST /ingest.
    // Since Python isn't installed, this will fail during processing,
    // but we can still test the full lifecycle: queued → running → failed.
    let boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    let body = format!(
        "--{boundary}\r\n\
         Content-Disposition: form-data; name=\"file\"; filename=\"test.txt\"\r\n\
         Content-Type: text/plain\r\n\r\n\
         hello world\r\n\
         --{boundary}\r\n\
         Content-Disposition: form-data; name=\"title\"\r\n\r\n\
         Test Doc\r\n\
         --{boundary}--\r\n"
    );

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ingest")
                .header("content-type", format!("multipart/form-data; boundary={boundary}"))
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let json = json_body(resp).await;
    let job_id = json["job_id"].as_u64().unwrap();
    assert_eq!(json["status"].as_str().unwrap(), "queued");

    // Wait for the worker to process the job (Python will fail, so job goes to Failed).
    // Poll for up to 10 seconds.
    let mut status = String::new();
    for _ in 0..20 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(&format!("/jobs/{job_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let json = json_body(resp).await;
        status = json["status"].as_str().unwrap().to_string();
        if status == "failed" || status == "complete" {
            break;
        }
    }

    assert!(
        status == "failed" || status == "complete",
        "Expected job to be failed or complete, but got: {status}"
    );
}

#[tokio::test]
async fn integration_job_404() {
    let (app, _dir, _worker) = make_server_seeded();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/jobs/9999")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
