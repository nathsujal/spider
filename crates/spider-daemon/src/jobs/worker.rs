//! Background worker — pulls from the job channel and processes each ingestion.
//!
//! ## Ingestion flow
//!
//! 1. Pop a `JobSubmission` from the channel
//! 2. Mark the job Running
//! 3. Spawn the Python ingestion subprocess:
//!    `python3 -m spider.ingest --file <path> --title <title>`
//! 4. Parse the JSON output from stdout:
//!    ```json
//!    {"propositions": [{"text": "...", "entities": [{"name": "...", "entity_type": "..."}]}]}
//!    ```
//! 5. Call `spider_core::db::ingest::index()` on the Spider database
//! 6. Emit TraceEvents to the broadcaster
//! 7. Mark the job Complete or Failed
//!
//! ## When the Python package doesn't exist
//!
//! The subprocess will fail (module not found) and the job is marked Failed
//! with a descriptive error message. The infrastructure is ready for when
//! the Python package is built.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tracing::{error, info};

use spider_core::db::ingest::{self, Entity, IngestRequest, Proposition};

use crate::events::TraceEvent;
use crate::jobs::queue::JobSubmission;
use crate::jobs::JobResult;
use crate::ws::Broadcaster;

/// Default timeout for Python subprocess (5 minutes).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// Deserialized JSON output from the Python ingestion subprocess.
#[derive(Debug, serde::Deserialize)]
struct PythonOutput {
    propositions: Vec<PythonProposition>,
}

#[derive(Debug, serde::Deserialize)]
struct PythonProposition {
    text: String,
    #[serde(default)]
    entities: Vec<PythonEntity>,
}

#[derive(Debug, serde::Deserialize)]
struct PythonEntity {
    name: String,
    #[serde(rename = "entity_type")]
    type_: String,
}

/// Background ingestion worker.
pub struct Worker {
    /// Pulls jobs from the queue.
    rx: mpsc::Receiver<JobSubmission>,
    /// Access to the shared job store for status updates.
    queue: Arc<crate::jobs::queue::JobQueue>,
    /// Broadcast trace events to WebSocket clients.
    broadcaster: Arc<Broadcaster>,
    /// Handle to the Spider database for calling `index()`.
    db: Arc<tokio::sync::Mutex<spider_core::db::lifecycle::Spider>>,
    /// Timeout for Python subprocess.
    timeout: Duration,
}

impl Worker {
    /// Create a new worker bound to the given job receiver, broadcaster, and Spider DB.
    pub fn new(
        rx: mpsc::Receiver<JobSubmission>,
        queue: Arc<crate::jobs::queue::JobQueue>,
        broadcaster: Arc<Broadcaster>,
        db: Arc<tokio::sync::Mutex<spider_core::db::lifecycle::Spider>>,
    ) -> Self {
        Self {
            rx,
            queue,
            broadcaster,
            db,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Run the worker loop — pulls jobs from the channel and processes them.
    ///
    /// Returns when the channel is closed (all senders dropped).
    pub async fn run(mut self) {
        info!("ingestion worker started");

        while let Some(submission) = self.rx.recv().await {
            let job_id = submission.id;
            info!(job_id, file = %submission.file_path, "processing job");

            // Emit job_queued event.
            self.broadcaster.send(
                TraceEvent::new(job_id, crate::events::TraceKind::JobQueued)
                    .with_data("file_path", submission.file_path.clone()),
            );

            // Mark as running.
            self.queue.update(job_id, |job| job.start()).await;

            self.broadcaster.send(
                TraceEvent::new(job_id, crate::events::TraceKind::FileStored)
                    .with_data("file_path", submission.file_path.clone()),
            );

            // Process the file through the Python pipeline.
            match self.run_ingestion(job_id, &submission).await {
                Ok(result) => {
                    self.queue.update(job_id, |job| job.complete(result.clone())).await;
                    info!(job_id, nodes = result.nodes_created, edges = result.edges_created, "job completed");
                }
                Err(e) => {
                    error!(job_id, error = %e, "job failed");
                    self.queue.update(job_id, |job| job.fail(e.clone())).await;
                    self.broadcaster.send(
                        TraceEvent::new(job_id, crate::events::TraceKind::JobFailed)
                            .with_data("error", e.clone()),
                    );
                }
            }
        }

        info!("ingestion worker stopped (channel closed)");
    }

    /// Run the full ingestion pipeline for a single job.
    async fn run_ingestion(
        &self,
        job_id: u64,
        submission: &JobSubmission,
    ) -> Result<JobResult, String> {
        // Step 1: Run the Python subprocess to extract propositions.
        let output = self.run_python_subprocess(job_id, submission).await?;

        // Step 2: Parse the JSON output.
        let python_output: PythonOutput = serde_json::from_str(&output)
            .map_err(|e| format!("failed to parse Python output: {e}"))?;

        // Step 3: Convert to spider-core types and call index().
        let propositions: Vec<Proposition<'_>> = python_output.propositions
            .iter()
            .map(|p| Proposition {
                text: &p.text,
                entities: p.entities
                    .iter()
                    .map(|e| Entity {
                        name: &e.name,
                        entity_type: &e.type_,
                    })
                    .collect(),
            })
            .collect();

        let title = submission.title.as_deref().unwrap_or("Untitled Document");
        let req = IngestRequest { title, propositions };

        let ingest_result = {
            let mut db = self.db.lock().await;
            ingest::index(&mut db, &req)
                .map_err(|e| format!("index failed: {e}"))?
        };

        let result = JobResult {
            nodes_created: (ingest_result.proposition_count
                + ingest_result.entity_count
                + 1) as u32, // +1 for the Document node
            edges_created: ingest_result.edge_count as u32,
        };

        self.broadcaster.send(
            TraceEvent::new(job_id, crate::events::TraceKind::NodesCreated)
                .with_data("nodes_created", result.nodes_created),
        );
        self.broadcaster.send(
            TraceEvent::new(job_id, crate::events::TraceKind::EdgesCreated)
                .with_data("edges_created", result.edges_created),
        );
        self.broadcaster.send(
            TraceEvent::new(job_id, crate::events::TraceKind::JobComplete)
                .with_data("nodes_created", result.nodes_created)
                .with_data("edges_created", result.edges_created),
        );

        Ok(result)
    }

    /// Spawn the Python subprocess and capture its stdout.
    ///
    /// Expected command: `python3 -m spider.ingest --file <path> [--title <title>]`
    /// Expected stdout: JSON with the format `{"propositions": [...]}`
    async fn run_python_subprocess(
        &self,
        job_id: u64,
        submission: &JobSubmission,
    ) -> Result<String, String> {
        // Get the DB path from the Spider handle.
        let db_path = {
            let db = self.db.lock().await;
            db.path().to_string_lossy().to_string()
        };

        let mut cmd = tokio::process::Command::new("python3");
        cmd.args([
            "-m", "spider.ingest",
            "--db-path", &db_path,
            "--file", &submission.file_path,
        ]);

        if let Some(title) = &submission.title {
            cmd.args(["--title", title]);
        }

        let output = tokio::time::timeout(
            self.timeout,
            cmd.output(),
        )
        .await
        .map_err(|_| "Python subprocess timed out".to_string())?
        .map_err(|e| format!("failed to spawn Python subprocess: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);

            // Provide a helpful message when the Python module doesn't exist.
            if stderr.contains("No module named") || stderr.contains("ModuleNotFoundError") {
                return Err(
                    "Python spider package is not installed. \
                     Run `pip install -e python/` in the project root, \
                     or ensure the python/spider/ package exists."
                        .to_string()
                );
            }

            return Err(format!(
                "Python subprocess exited with status {}: {}",
                output.status,
                stderr.lines().next().unwrap_or(&stdout),
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();

        if stdout.trim().is_empty() {
            return Err("Python subprocess produced no output".to_string());
        }

        info!(job_id, bytes = stdout.len(), "Python subprocess completed");
        Ok(stdout)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use spider_core::db::lifecycle::Spider;
    use crate::jobs::JobQueue;
    use crate::jobs::JobStatus;
    use tempfile::TempDir;

    fn make_worker() -> (Worker, Arc<JobQueue>, Arc<tokio::sync::Mutex<Spider>>, TempDir) {
        let (queue, rx) = JobQueue::new();
        let queue = Arc::new(queue);
        let broadcaster = Arc::new(Broadcaster::new());
        let dir = tempfile::tempdir().unwrap();
        let db = Spider::open(dir.path()).unwrap();
        let db = Arc::new(tokio::sync::Mutex::new(db));
        let worker = Worker::new(rx, Arc::clone(&queue), broadcaster, Arc::clone(&db));
        (worker, queue, db, dir)
    }

    #[tokio::test]
    async fn worker_marks_job_failed_when_python_not_available() {
        let (worker, queue, _db, _dir) = make_worker();

        let id = queue.submit("/tmp/test.txt".into(), Some("Test".into())).await;
        queue.close().await;
        worker.run().await;

        let job = queue.get(id).await.unwrap();
        assert_eq!(job.status, JobStatus::Failed);
        assert!(job.error.as_ref().unwrap().contains("Python spider package"));
    }
}
