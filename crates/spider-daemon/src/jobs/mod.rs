//! Job state, queue, and worker orchestration for async ingestion.
//!
//! ## Architecture
//!
//! ```text
//! HTTP route ──submit──▶ JobQueue ──mpsc──▶ Worker ──index──▶ Spider
//!       │                    │
//!       │              Arc<Mutex<Hash>>
//!       │                    │
//! GET /jobs/:id ◀────────────┘ (queryable)
//! ```
//!
//! - `JobQueue::new()` creates the mpsc channel + shared HashMap
//! - `submit()` pushes a job into the channel, returns its ID immediately
//! - Worker pulls from the channel, runs the job, updates status
//! - `get()` / `list()` query the shared HashMap from HTTP handlers

pub mod queue;
pub mod worker;

pub use queue::JobQueue;
pub use worker::Worker;

use serde::{Deserialize, Serialize};

// ── Job status ────────────────────────────────────────────────────────────────

/// The lifecycle state of an ingestion job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Job is waiting to be processed.
    Queued,
    /// Job is currently running.
    Running,
    /// Job completed successfully.
    Complete,
    /// Job failed with an error.
    Failed,
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued   => write!(f, "queued"),
            Self::Running  => write!(f, "running"),
            Self::Complete => write!(f, "complete"),
            Self::Failed   => write!(f, "failed"),
        }
    }
}

// ── Job result ────────────────────────────────────────────────────────────────

/// Summary of what an ingestion job created.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    /// Number of nodes created during ingestion.
    pub nodes_created: u32,
    /// Number of edges created during ingestion.
    pub edges_created: u32,
}

// ── Job ───────────────────────────────────────────────────────────────────────

/// A single ingestion job tracked by ID.
///
/// Stored in the shared `HashMap<u64, Job>` inside [`JobQueue`].
/// HTTP handlers read from this map for `GET /jobs/:id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Unique monotonically increasing ID.
    pub id: u64,
    /// Current lifecycle state.
    pub status: JobStatus,
    /// When the job was created (unix milliseconds).
    pub created_at: u64,
    /// When the job was last updated (unix milliseconds).
    pub updated_at: u64,
    /// Result summary (set when status becomes Complete).
    pub result: Option<JobResult>,
    /// Error message (set when status becomes Failed).
    pub error: Option<String>,
    /// Path to the uploaded file on disk (for the worker to process).
    pub file_path: Option<String>,
    /// Optional title for the ingestion.
    pub title: Option<String>,
}

impl Job {
    /// Create a new job in the Queued state.
    pub fn new(id: u64, file_path: String, title: Option<String>) -> Self {
        let now = now_unix_ms();
        Self {
            id,
            status: JobStatus::Queued,
            created_at: now,
            updated_at: now,
            result: None,
            error: None,
            file_path: Some(file_path),
            title,
        }
    }

    /// Transition the job to Running.
    pub fn start(&mut self) {
        self.status = JobStatus::Running;
        self.updated_at = now_unix_ms();
    }

    /// Transition the job to Complete with a result summary.
    pub fn complete(&mut self, result: JobResult) {
        self.status = JobStatus::Complete;
        self.updated_at = now_unix_ms();
        self.result = Some(result);
    }

    /// Transition the job to Failed with an error message.
    #[allow(dead_code)]
    pub fn fail(&mut self, error: String) {
        self.status = JobStatus::Failed;
        self.updated_at = now_unix_ms();
        self.error = Some(error);
    }

    /// Whether the job has reached a terminal state.
    #[allow(dead_code)]
    pub fn is_terminal(&self) -> bool {
        matches!(self.status, JobStatus::Complete | JobStatus::Failed)
    }
}

fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
