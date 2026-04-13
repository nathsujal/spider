use axum::{
    routing::{get, post},
    Router,
};
use spider_core::db::lifecycle::Spider;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::routes::{bio, health, jobs, nodes, query, traverse};
use crate::ws::{self, Broadcaster};
use crate::jobs::{JobQueue, Worker};

/// Shared application state — holds the Spider database handle behind a mutex
/// and the WebSocket event broadcaster.
///
/// We use `Arc<Mutex<Spider>>` because:
/// - `Spider` is not `Send` (mmap handles are tied to the thread that opened them)
/// - axum handlers require `Send` futures
/// - A mutex gives exclusive access to the database for read/write operations
///
/// For production, consider a more granular locking strategy or an actor model.
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Mutex<Spider>>,
    pub broadcaster: Arc<Broadcaster>,
    /// Queryable job queue — used by `POST /ingest` and `GET /jobs/:id` (Phase 4).
    #[allow(dead_code)]
    pub job_queue: Arc<JobQueue>,
}

#[cfg(test)]
impl AppState {
    /// Create an AppState suitable for tests — empty DB and no-op broadcaster.
    pub fn test(db: Spider) -> Self {
        let (queue, _rx) = JobQueue::new();
        Self {
            db: Arc::new(Mutex::new(db)),
            broadcaster: Arc::new(Broadcaster::new()),
            job_queue: Arc::new(queue),
        }
    }

    /// Create an AppState with a specific job queue (for job endpoint tests).
    pub fn test_with_queue(queue: Arc<JobQueue>) -> Self {
        let db = Spider::open(&tempfile::tempdir().unwrap().path().join("test_db")).unwrap();
        Self {
            db: Arc::new(Mutex::new(db)),
            broadcaster: Arc::new(Broadcaster::new()),
            job_queue: queue,
        }
    }

    /// Create an AppState with an empty job queue (for 404 tests).
    pub fn test_default() -> Self {
        let db = Spider::open(&tempfile::tempdir().unwrap().path().join("test_db")).unwrap();
        Self::test(db)
    }
}

/// Response payload for the health check endpoint.
#[derive(serde::Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub db_path: String,
}

/// Build the axum router with all routes registered.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::handler))
        .route("/nodes/:id", get(nodes::get_node))
        .route("/nodes", get(nodes::list_nodes))
        .route("/bio", get(bio::handler))
        .route("/search", get(query::handler))
        .route("/ws", get(ws::broadcaster::handler))
        .route("/ingest", post(jobs::ingest))
        .route("/jobs/:id/stream", get(jobs::handler))
        .route("/jobs/:id", get(jobs::get_job))
        .route("/traverse/:node_id", get(traverse::handler))
        .with_state(state)
        .layer(CorsLayer::permissive())
}

/// Start the HTTP server and block until shutdown.
pub async fn run(db: Spider, addr: SocketAddr) -> anyhow::Result<()> {
    let broadcaster = Arc::new(Broadcaster::new());
    let (job_queue, rx) = JobQueue::new();
    let job_queue = Arc::new(job_queue);

    // Wrap the Spider DB in a mutex for shared access.
    let db = Arc::new(Mutex::new(db));

    // Spawn the background worker — it owns the receiver and processes jobs.
    let worker = Worker::new(
        rx,
        Arc::clone(&job_queue),
        Arc::clone(&broadcaster),
        Arc::clone(&db),
    );
    tokio::spawn(worker.run());

    let state = AppState {
        db,
        broadcaster,
        job_queue,
    };

    let app = build_router(state);

    info!(%addr, "spider-daemon starting");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
