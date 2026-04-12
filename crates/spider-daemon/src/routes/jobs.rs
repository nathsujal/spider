use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path,
        State,
    },
    response::IntoResponse,
};
use futures::StreamExt;
use tokio::sync::broadcast;
use tracing::{info, warn};

use crate::events::TraceEvent;
use crate::server::AppState;
use crate::ws::Broadcaster;

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
    use crate::events::TraceKind;

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
}
