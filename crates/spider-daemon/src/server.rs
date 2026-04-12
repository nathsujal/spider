use axum::{
    routing::{get, Router},
};
use serde::Serialize;
use spider_core::db::lifecycle::Spider;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::routes::{bio, health, nodes, query};

/// Shared application state — holds the Spider database handle behind a mutex.
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
}

/// Response payload for the health check endpoint.
#[derive(Serialize)]
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
        .with_state(state)
        .layer(CorsLayer::permissive())
}

/// Start the HTTP server and block until shutdown.
pub async fn run(db: Spider, addr: SocketAddr) -> anyhow::Result<()> {
    let state = AppState {
        db: Arc::new(Mutex::new(db)),
    };

    let app = build_router(state);

    info!(%addr, "spider-daemon starting");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
