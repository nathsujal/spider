//! TraceEvent data model for WebSocket event broadcasting.
//!
//! Every ingestion step emits a `TraceEvent` that gets broadcast to all
//! connected WebSocket clients. Clients use these events for:
//! - Live graph rendering in Studio (browser)
//! - Live ingestion progress in CLI (`/jobs/:id/stream`)

use serde::{Deserialize, Serialize};

// ── Event kinds ───────────────────────────────────────────────────────────────

/// The type of trace event emitted during an ingestion job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceKind {
    /// Job was created and queued.
    JobQueued,
    /// Raw file was stored (blob deduplicated by SHA-256).
    FileStored,
    /// Text was extracted from the source document.
    ChunksExtracted,
    /// LLM returned propositions for a chunk.
    PropositionsFound,
    /// Nodes were created from propositions/entities.
    NodesCreated,
    /// Edges were wired between nodes.
    EdgesCreated,
    /// Job completed successfully.
    JobComplete,
    /// Job failed with an error.
    JobFailed,
}

impl std::fmt::Display for TraceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JobQueued          => write!(f, "job_queued"),
            Self::FileStored         => write!(f, "file_stored"),
            Self::ChunksExtracted    => write!(f, "chunks_extracted"),
            Self::PropositionsFound  => write!(f, "propositions_found"),
            Self::NodesCreated       => write!(f, "nodes_created"),
            Self::EdgesCreated       => write!(f, "edges_created"),
            Self::JobComplete        => write!(f, "job_complete"),
            Self::JobFailed          => write!(f, "job_failed"),
        }
    }
}

// ── Payload data ──────────────────────────────────────────────────────────────

/// Arbitrary event data.
///
/// Different `TraceKind` values carry different payloads. This is an open
/// map so the Python subprocess and the worker can attach whatever context
/// is useful without needing a new Rust struct for every event type.
///
/// Examples:
/// - `{"chunk_index": 3, "propositions_found": 12}`
/// - `{"nodes_created": 42, "edges_created": 38}`
/// - `{"error": "LLM timeout exceeded"}`
pub type TraceData = serde_json::Map<String, serde_json::Value>;

// ── TraceEvent ────────────────────────────────────────────────────────────────

/// A single trace event emitted during an ingestion job.
///
/// Serialized to JSON and sent over WebSocket to all connected clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    /// The job this event belongs to.
    pub job_id: u64,
    /// What kind of event this is.
    pub kind: TraceKind,
    /// Arbitrary event data.
    #[serde(default)]
    pub data: TraceData,
    /// Unix timestamp in milliseconds when the event was created.
    pub timestamp_ms: u64,
}

impl TraceEvent {
    /// Create a new trace event with the current timestamp.
    pub fn new(job_id: u64, kind: TraceKind) -> Self {
        Self {
            job_id,
            kind,
            data: TraceData::new(),
            timestamp_ms: now_unix_ms(),
        }
    }

    /// Add a key-value pair to the event data.
    pub fn with_data(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.data.insert(key.to_string(), value.into());
        self
    }
}

fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_event_serialization() {
        let event = TraceEvent::new(7, TraceKind::PropositionsFound)
            .with_data("chunk_index", 3)
            .with_data("propositions_found", 12);

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"job_id\":7"));
        assert!(json.contains("\"kind\":\"propositions_found\""));
        assert!(json.contains("\"chunk_index\":3"));
        assert!(json.contains("\"propositions_found\":12"));
        assert!(json.contains("\"timestamp_ms\":"));
    }

    #[test]
    fn trace_event_deserialization() {
        let json = r#"{
            "job_id": 42,
            "kind": "job_complete",
            "data": {"nodes_created": 10, "edges_created": 8},
            "timestamp_ms": 1712934000000
        }"#;

        let event: TraceEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.job_id, 42);
        assert_eq!(event.kind, TraceKind::JobComplete);
        assert_eq!(event.timestamp_ms, 1712934000000);
        assert_eq!(event.data.get("nodes_created").unwrap().as_i64().unwrap(), 10);
        assert_eq!(event.data.get("edges_created").unwrap().as_i64().unwrap(), 8);
    }

    #[test]
    fn trace_event_empty_data_deserializes() {
        let json = r#"{"job_id": 1, "kind": "job_queued", "timestamp_ms": 0}"#;
        let event: TraceEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.job_id, 1);
        assert_eq!(event.kind, TraceKind::JobQueued);
        assert!(event.data.is_empty());
    }

    #[test]
    fn trace_kind_display_matches_serde() {
        for kind in [
            TraceKind::JobQueued,
            TraceKind::FileStored,
            TraceKind::ChunksExtracted,
            TraceKind::PropositionsFound,
            TraceKind::NodesCreated,
            TraceKind::EdgesCreated,
            TraceKind::JobComplete,
            TraceKind::JobFailed,
        ] {
            let serialized = serde_json::to_string(&kind).unwrap();
            // Serialized is "\"snake_case\"", strip quotes
            let s = serialized.trim_matches('"');
            assert_eq!(s, kind.to_string());
        }
    }

    #[test]
    fn trace_event_new_sets_timestamp() {
        let event = TraceEvent::new(1, TraceKind::JobQueued);
        assert!(event.timestamp_ms > 0);
    }

    #[test]
    fn trace_event_round_trip() {
        let original = TraceEvent::new(99, TraceKind::JobFailed)
            .with_data("error", "timeout");

        let json = serde_json::to_string(&original).unwrap();
        let restored: TraceEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.job_id, original.job_id);
        assert_eq!(restored.kind, original.kind);
        assert_eq!(restored.data.get("error").unwrap().as_str().unwrap(), "timeout");
    }
}
