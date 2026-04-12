//! Background worker — pulls from the job channel and processes each ingestion.
//!
//! ## Current behaviour (stub)
//!
//! The worker marks the job as Running, sleeps briefly, then marks it
//! Complete with dummy results. Phase 4 will replace this with:
//! 1. Store the uploaded file in spider-core blob storage
//! 2. Spawn the Python ingestion subprocess
//! 3. Parse propositions from subprocess stdout
//! 4. Call `index()` on the Spider database
//! 5. Emit TraceEvents to the broadcaster

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tracing::info;

use crate::events::TraceEvent;
use crate::jobs::queue::JobSubmission;
use crate::jobs::JobResult;
use crate::ws::Broadcaster;

/// Background ingestion worker.
pub struct Worker {
    /// Pulls jobs from the queue.
    rx: mpsc::Receiver<JobSubmission>,
    /// Access to the shared job store for status updates.
    queue: Arc<crate::jobs::queue::JobQueue>,
    /// Broadcast trace events to WebSocket clients.
    broadcaster: Arc<Broadcaster>,
}

impl Worker {
    /// Create a new worker bound to the given job receiver and broadcaster.
    pub fn new(
        rx: mpsc::Receiver<JobSubmission>,
        queue: Arc<crate::jobs::queue::JobQueue>,
        broadcaster: Arc<Broadcaster>,
    ) -> Self {
        Self {
            rx,
            queue,
            broadcaster,
        }
    }

    /// Run the worker loop — pulls jobs from the channel and processes them.
    ///
    /// Returns when the channel is closed (all senders dropped).
    pub async fn run(mut self) {
        info!("ingestion worker started");

        while let Some(submission) = self.rx.recv().await {
            let job_id = submission.id;
            info!(job_id, "processing job");

            // Emit job_queued event.
            self.broadcaster.send(
                TraceEvent::new(job_id, crate::events::TraceKind::JobQueued)
                    .with_data("file_path", submission.file_path.clone()),
            );

            // Mark as running.
            self.queue.update(job_id, |job| job.start()).await;

            self.broadcaster.send(
                TraceEvent::new(job_id, crate::events::TraceKind::FileStored)
                    .with_data("status", "stored"),
            );

            // --- STUB: simulate work ---
            // In Phase 4 this becomes the full Python subprocess pipeline.
            tokio::time::sleep(Duration::from_millis(100)).await;

            // --- STUB: dummy result ---
            let result = JobResult {
                nodes_created: 0,
                edges_created: 0,
            };

            self.broadcaster.send(
                TraceEvent::new(job_id, crate::events::TraceKind::NodesCreated)
                    .with_data("nodes_created", result.nodes_created),
            );
            self.broadcaster.send(
                TraceEvent::new(job_id, crate::events::TraceKind::EdgesCreated)
                    .with_data("edges_created", result.edges_created),
            );

            self.queue.update(job_id, |job| job.complete(result.clone())).await;

            self.broadcaster.send(
                TraceEvent::new(job_id, crate::events::TraceKind::JobComplete)
                    .with_data("nodes_created", result.nodes_created)
                    .with_data("edges_created", result.edges_created),
            );

            info!(job_id, "job completed");
        }

        info!("ingestion worker stopped (channel closed)");
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::JobQueue;
    use crate::jobs::JobStatus;

    fn make_worker() -> (Worker, std::sync::Arc<JobQueue>) {
        let (queue, rx) = JobQueue::new();
        let broadcaster = Arc::new(Broadcaster::new());
        let queue = Arc::new(queue);
        let worker = Worker::new(rx, Arc::clone(&queue), broadcaster);
        (worker, queue)
    }

    #[tokio::test]
    async fn worker_processes_single_job() {
        let (worker, queue) = make_worker();

        let id = queue.submit("/tmp/test.pdf".into(), Some("Test".into())).await;
        queue.close().await; // signal no more jobs → worker.run() exits
        worker.run().await;

        let job = queue.get(id).await.unwrap();
        assert_eq!(job.status, JobStatus::Complete);
        assert_eq!(job.title.as_deref(), Some("Test"));
    }

    #[tokio::test]
    async fn worker_processes_multiple_jobs() {
        let (worker, queue) = make_worker();

        queue.submit("/tmp/a.pdf".into(), None).await;
        queue.submit("/tmp/b.pdf".into(), None).await;
        queue.submit("/tmp/c.pdf".into(), None).await;

        queue.close().await;
        worker.run().await;

        let jobs = queue.list().await;
        assert_eq!(jobs.len(), 3);
        assert!(jobs.iter().all(|j| j.status == JobStatus::Complete));
    }

    #[tokio::test]
    async fn worker_updates_job_through_queue() {
        let (worker, queue) = make_worker();

        let id = queue.submit("/tmp/x.pdf".into(), None).await;

        queue.close().await;
        worker.run().await;

        let job = queue.get(id).await.unwrap();
        assert!(job.result.is_some());
        assert!(job.error.is_none());
    }
}
