//! WebSocket broadcaster — fan-out events to all connected clients.
//!
//! Uses `tokio::sync::broadcast` so every connected WebSocket client
//! receives the same stream of `TraceEvent`s. The broadcaster lives
//! in `Arc<Broadcaster>` and is shared via [`AppState`](crate::server::AppState).

use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures::StreamExt;
use tokio::sync::{broadcast, Mutex};
use tracing::{info, warn};

use crate::events::TraceEvent;
use crate::server::AppState;

/// Number of buffered events per receiver before old events are dropped.
/// If a client falls behind, it will miss events — better than blocking the sender.
const BROADCAST_CAPACITY: usize = 256;

/// Shared broadcaster handle.
#[derive(Debug, Clone)]
pub struct Broadcaster {
    /// The sender half — workers call `send()` on this.
    sender: broadcast::Sender<TraceEvent>,
    /// Track connected client count for logging.
    client_count: Arc<Mutex<u64>>,
}

impl Broadcaster {
    /// Create a new broadcaster with a fresh broadcast channel.
    pub fn new() -> Self {
        let (sender, _receiver) = broadcast::channel(BROADCAST_CAPACITY);
        Self {
            sender,
            client_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Return a new receiver subscribed to the broadcast channel.
    ///
    /// Each call creates an independent receiver that will receive all
    /// subsequent events. The receiver can be attached to a WebSocket
    /// connection for live streaming.
    pub fn subscribe(&self) -> broadcast::Receiver<TraceEvent> {
        self.sender.subscribe()
    }

    /// Broadcast an event to all connected clients.
    ///
    /// Returns the number of clients that received the event.
    /// Clients that are lagging (receiver buffer full) are silently skipped.
    pub fn send(&self, event: TraceEvent) -> usize {
        self.sender.send(event).unwrap_or(0)
    }

    /// Increment the client counter.
    pub async fn connect(&self) {
        let mut count = self.client_count.lock().await;
        *count += 1;
        info!(clients = *count, "websocket client connected");
    }

    /// Decrement the client counter.
    pub async fn disconnect(&self) {
        let mut count = self.client_count.lock().await;
        *count = count.saturating_sub(1);
        info!(clients = *count, "websocket client disconnected");
    }

    /// Return the current number of connected clients.
    #[allow(dead_code)]
    pub async fn client_count(&self) -> u64 {
        *self.client_count.lock().await
    }
}

// ── Handler ──────────────────────────────────────────────────────────────────

/// `GET /ws` — WebSocket endpoint for receiving all trace events.
///
/// On connect, the client receives a `connected` welcome message.
/// After that, every `TraceEvent` broadcast by the daemon is forwarded
/// to the client as JSON.
pub async fn handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state.broadcaster))
}

async fn handle_socket(mut socket: WebSocket, broadcaster: Arc<Broadcaster>) {
    broadcaster.connect().await;

    let mut rx = broadcaster.subscribe();

    // Send a welcome message.
    let welcome = serde_json::json!({
        "event": "connected",
    });
    if let Err(e) = socket.send(Message::Text(welcome.to_string())).await {
        warn!("failed to send welcome message: {e}");
        broadcaster.disconnect().await;
        return;
    }

    // Single loop: select between incoming broadcast events and client messages.
    loop {
        tokio::select! {
            // Broadcast event → send to client.
            event_result = rx.recv() => {
                match event_result {
                    Ok(event) => {
                        let json = serde_json::to_string(&event).unwrap_or_else(|_| "{}".into());
                        if let Err(e) = socket.send(Message::Text(json)).await {
                            warn!("websocket send error: {e}");
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("websocket client lagged behind by {n} events");
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
            // Client message → detect disconnect.
            msg = socket.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Text(_) | Message::Binary(_) | Message::Ping(_) | Message::Pong(_))) => {
                        // axum WebSocket layer handles ping/pong automatically.
                        // Ignore other client messages for now.
                    }
                    Some(Err(_)) => break,
                }
            }
        }
    }

    broadcaster.disconnect().await;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::TraceKind;

    #[tokio::test]
    async fn broadcaster_send_delivers_to_all() {
        let b = Broadcaster::new();
        let mut rx1 = b.subscribe();
        let mut rx2 = b.subscribe();

        let event = TraceEvent::new(1, TraceKind::JobQueued);
        let count = b.send(event.clone());
        assert_eq!(count, 2);

        assert_eq!(rx1.recv().await.unwrap().job_id, 1);
        assert_eq!(rx2.recv().await.unwrap().job_id, 1);
    }

    #[tokio::test]
    async fn broadcaster_subscribe_after_send_misses_old_events() {
        let b = Broadcaster::new();
        let mut rx1 = b.subscribe();

        b.send(TraceEvent::new(1, TraceKind::JobQueued));
        b.send(TraceEvent::new(2, TraceKind::JobQueued));

        // Subscribe after events were sent — should only get new events.
        let mut rx2 = b.subscribe();
        b.send(TraceEvent::new(3, TraceKind::FileStored));

        // rx1 should have received all 3.
        assert_eq!(rx1.recv().await.unwrap().job_id, 1);
        assert_eq!(rx1.recv().await.unwrap().job_id, 2);
        assert_eq!(rx1.recv().await.unwrap().job_id, 3);

        // rx2 should only get the last one.
        assert_eq!(rx2.recv().await.unwrap().job_id, 3);
    }

    #[tokio::test]
    async fn client_count_tracks_connections() {
        let b = Broadcaster::new();
        assert_eq!(b.client_count().await, 0);

        b.connect().await;
        b.connect().await;
        assert_eq!(b.client_count().await, 2);

        b.disconnect().await;
        assert_eq!(b.client_count().await, 1);

        b.disconnect().await;
        assert_eq!(b.client_count().await, 0);

        // Should not go below zero.
        b.disconnect().await;
        assert_eq!(b.client_count().await, 0);
    }

    #[test]
    fn trace_event_serializes_for_websocket() {
        let event = TraceEvent::new(42, TraceKind::NodesCreated)
            .with_data("nodes_created", 5);

        let json = serde_json::to_string(&event).unwrap();
        // Verify it's valid JSON that could be sent over WebSocket.
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["job_id"], 42);
        assert_eq!(parsed["kind"], "nodes_created");
        assert_eq!(parsed["data"]["nodes_created"], 5);
    }
}
