use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Multipart,
        Path,
        State,
    },
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use futures::StreamExt;
use tokio::sync::broadcast;
use tracing::{info, warn};

use crate::server::AppState;
use crate::ws::Broadcaster;

// ── POST /ingest ─────────────────────────────────────────────────────────────

/// Response payload for `POST /ingest`.
#[derive(serde::Serialize, Debug)]
pub struct IngestResponse {
    pub job_id: u64,
    pub status: String,
}

/// `POST /ingest` — upload a document for ingestion.
///
/// Accepts a multipart form with:
/// - `file` (required) — the document file (PDF, TXT, etc.)
/// - `title` (optional) — human-readable title for the document
///
/// Returns immediately with a `job_id`. The ingestion runs asynchronously
/// in the background worker. Track progress via `GET /jobs/:id` or
/// `GET /jobs/:id/stream`.
pub async fn ingest(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<IngestResponse>, (StatusCode, String)> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut title: Option<String> = None;

    // Parse multipart fields.
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (StatusCode::BAD_REQUEST, format!("failed to read multipart field: {e}"))
    })? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                file_name = field.file_name().map(|s| s.to_string());
                file_bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| (StatusCode::BAD_REQUEST, format!("failed to read file bytes: {e}")))?
                        .to_vec(),
                );
            }
            "title" => {
                title = field
                    .text()
                    .await
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
            }
            _ => {
                // Ignore unknown fields.
            }
        }
    }

    let file_bytes = match file_bytes {
        Some(bytes) => bytes,
        None => return Err((StatusCode::BAD_REQUEST, "missing required field: file".into())),
    };

    // Determine file name for logging.
    let original_name = file_name.unwrap_or_else(|| "unknown".into());

    // Write uploaded file to a temp location.
    let upload_dir = std::env::temp_dir().join("spider-ingest");
    let _ = std::fs::create_dir_all(&upload_dir);

    // Use a UUID-like name to avoid collisions.
    let file_id = uuid_simple();
    let file_path = upload_dir.join(format!("{file_id}_{original_name}"));

    std::fs::write(&file_path, &file_bytes)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("failed to write uploaded file: {e}")))?;

    let file_path_str = file_path.to_string_lossy().to_string();

    info!(file_path = %file_path_str, title, "file uploaded, creating job");

    // Create the ingestion job.
    let job_id = state.job_queue.submit(file_path_str.clone(), title).await;

    info!(job_id, "ingestion job created");

    Ok(Json(IngestResponse {
        job_id,
        status: "queued".into(),
    }))
}

/// Generate a simple unique ID without external dependencies.
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    // Mix in a thread-local counter to reduce collision risk.
    use std::cell::Cell;
    thread_local! {
        static COUNTER: Cell<u64> = const { Cell::new(0) };
    }
    let counter = COUNTER.with(|c| {
        let v = c.get().wrapping_add(1);
        c.set(v);
        v
    });
    format!("{:x}_{:x}", ts, counter)
}

// ── GET /jobs/:id ────────────────────────────────────────────────────────────

/// Response payload for `GET /jobs/:id`.
#[derive(serde::Serialize, Debug)]
pub struct JobResponse {
    pub job_id: u64,
    pub status: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub file_path: Option<String>,
    pub title: Option<String>,
    pub result: Option<crate::jobs::JobResult>,
    pub error: Option<String>,
}

/// `GET /jobs/:id` — query the status of an ingestion job.
///
/// Returns 404 if the job ID is unknown.
pub async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<u64>,
) -> Result<Json<JobResponse>, (StatusCode, &'static str)> {
    let job = state
        .job_queue
        .get(job_id)
        .await
        .ok_or((StatusCode::NOT_FOUND, "job not found"))?;

    Ok(Json(JobResponse {
        job_id: job.id,
        status: job.status.to_string(),
        created_at: job.created_at,
        updated_at: job.updated_at,
        file_path: job.file_path,
        title: job.title,
        result: job.result,
        error: job.error,
    }))
}

/// `GET /jobs/:id/stream` — WebSocket endpoint for receiving events for a
/// specific ingestion job.
///
/// On connect, the client receives a welcome message with the job ID.
/// After that, only `TraceEvent`s matching the requested `job_id` are
/// forwarded. Other events are filtered out.
///
/// Returns 404 if the job doesn't exist in the queue (future, when jobs
/// system is built — for now always accepts).
pub async fn handler(
    ws: WebSocketUpgrade,
    Path(job_id): Path<u64>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_stream(socket, job_id, state.broadcaster))
}

async fn handle_stream(
    mut socket: WebSocket,
    job_id: u64,
    broadcaster: std::sync::Arc<Broadcaster>,
) {
    broadcaster.connect().await;

    let mut rx = broadcaster.subscribe();

    // Send a welcome message with the subscribed job ID.
    let welcome = serde_json::json!({
        "event": "subscribed",
        "job_id": job_id,
    });
    if let Err(e) = socket.send(Message::Text(welcome.to_string())).await {
        warn!("failed to send welcome message: {e}");
        broadcaster.disconnect().await;
        return;
    }

    info!(job_id, "job stream client connected");

    // Filter events by job_id.
    loop {
        tokio::select! {
            event_result = rx.recv() => {
                match event_result {
                    Ok(event) => {
                        // Only forward events for the requested job.
                        if event.job_id != job_id {
                            continue;
                        }

                        let json = serde_json::to_string(&event)
                            .unwrap_or_else(|_| "{}".into());
                        if let Err(e) = socket.send(Message::Text(json)).await {
                            warn!("websocket send error: {e}");
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(job_id, lagged = n, "job stream client lagged behind");
                        let notice = serde_json::json!({
                            "event": "lagged",
                            "dropped": n,
                        });
                        if socket.send(Message::Text(notice.to_string())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            msg = socket.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Text(_) | Message::Binary(_) | Message::Ping(_) | Message::Pong(_))) => {
                        // Ignore client messages.
                    }
                    Some(Err(_)) => break,
                }
            }
        }
    }

    info!(job_id, "job stream client disconnected");
    broadcaster.disconnect().await;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{TraceEvent, TraceKind};

    #[tokio::test]
    async fn job_stream_filters_by_job_id() {
        let b = std::sync::Arc::new(Broadcaster::new());
        let mut rx = b.subscribe();

        // Send events for different jobs.
        b.send(TraceEvent::new(1, TraceKind::JobQueued));
        b.send(TraceEvent::new(2, TraceKind::JobQueued));
        b.send(TraceEvent::new(1, TraceKind::FileStored));
        b.send(TraceEvent::new(3, TraceKind::NodesCreated));

        // Simulate filtering for job 1.
        let mut filtered: Vec<u64> = Vec::new();
        for _ in 0..4 {
            let event = rx.recv().await.unwrap();
            if event.job_id == 1 {
                filtered.push(event.job_id);
            }
        }

        assert_eq!(filtered, vec![1, 1]);
    }

    #[tokio::test]
    async fn job_stream_receives_no_events_for_unknown_job() {
        let b = std::sync::Arc::new(Broadcaster::new());
        let mut rx = b.subscribe();

        b.send(TraceEvent::new(5, TraceKind::JobQueued));
        b.send(TraceEvent::new(6, TraceKind::FileStored));

        // Check that job 99 events would be filtered (we count them).
        let mut count = 0;
        for _ in 0..2 {
            let event = rx.recv().await.unwrap();
            if event.job_id == 99 {
                count += 1;
            }
        }

        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn get_job_returns_queued_status() {
        let (queue, _rx) = crate::jobs::JobQueue::new();
        let queue = std::sync::Arc::new(queue);
        let id = queue.submit("/tmp/test.pdf".into(), Some("Test Doc".into())).await;

        let state = crate::server::AppState::test_with_queue(queue);
        let resp = get_job(State(state), Path(id)).await.unwrap();

        assert_eq!(resp.job_id, id);
        assert_eq!(resp.status, "queued");
        assert_eq!(resp.title.as_deref(), Some("Test Doc"));
        assert_eq!(resp.file_path.as_deref(), Some("/tmp/test.pdf"));
        assert!(resp.result.is_none());
        assert!(resp.error.is_none());
    }

    #[tokio::test]
    async fn get_job_returns_404_for_unknown_id() {
        let state = crate::server::AppState::test_default();
        let result = get_job(State(state), Path(999)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
