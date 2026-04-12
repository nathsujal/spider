use axum::{
    extract::State,
    Json,
};

use crate::server::{AppState, HealthResponse};

/// `GET /health` — returns daemon health status and database path.
///
/// This is the simplest liveness check. If this endpoint responds,
/// the daemon is running and the database handle is accessible.
pub async fn handler(
    State(state): State<AppState>,
) -> Json<HealthResponse> {
    let db = state.db.lock().await;
    let db_path = db.path().to_string_lossy().to_string();

    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        db_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use spider_core::db::lifecycle::Spider;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn health_returns_ok() {
        let dir = tempfile::tempdir().unwrap();
        let db = Spider::open(dir.path()).unwrap();
        let db_path = db.path().to_string_lossy().to_string();
        let state = AppState {
            db: Arc::new(Mutex::new(db)),
        };

        let response = handler(State(state)).await;
        assert_eq!(response.status, "ok");
        assert_eq!(response.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(response.db_path, db_path);
    }
}
