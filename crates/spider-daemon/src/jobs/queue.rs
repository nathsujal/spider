//! Async job queue — receives ingestion jobs from HTTP routes
//! and hands them to the background worker.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, Mutex};
use tracing::info;

use super::Job;

/// Payload submitted into the job queue.
pub struct JobSubmission {
    pub id: u64,
    pub file_path: String,
    #[allow(dead_code)]
    pub title: Option<String>,
}

/// Job queue handle — shared between HTTP routes (submit) and the worker.
#[derive(Clone)]
pub struct JobQueue {
    /// Sender for pushing jobs into the worker channel.
    /// Wrapped in Mutex<Option<>> so `close()` can drop it through Arc.
    tx: Arc<Mutex<Option<mpsc::Sender<JobSubmission>>>>,
    /// Shared job store — queryable by HTTP routes.
    jobs: Arc<Mutex<HashMap<u64, Job>>>,
}

impl JobQueue {
    /// Create a new job queue and return `(JobQueue, mpsc::Receiver<JobSubmission>)`.
    ///
    /// The receiver should be passed to [`Worker::new`] and spawned as a
    /// background task. The queue only holds the sender.
    pub fn new() -> (Self, mpsc::Receiver<JobSubmission>) {
        let (tx, rx) = mpsc::channel::<JobSubmission>(32);
        let queue = Self {
            tx: Arc::new(Mutex::new(Some(tx))),
            jobs: Arc::new(Mutex::new(HashMap::new())),
        };
        (queue, rx)
    }

    /// Submit a new ingestion job.
    ///
    /// Returns the job ID. The job is immediately queryable via `get()` in
    /// the Queued state.
    pub async fn submit(&self, file_path: String, title: Option<String>) -> u64 {
        let id = {
            let jobs = self.jobs.lock().await;
            // Next ID is max existing key + 1, or 1 if empty.
            jobs.keys().max().copied().map_or(1, |max| max + 1)
        };

        let job = Job::new(id, file_path.clone(), title.clone());

        {
            let mut jobs = self.jobs.lock().await;
            jobs.insert(id, job);
        }

        info!(job_id = id, file_path, "job submitted to queue");

        if let Some(tx) = self.tx.lock().await.as_ref() {
            let _ = tx.send(JobSubmission { id, file_path, title }).await;
        }

        id
    }

    /// Look up a job by ID.
    pub async fn get(&self, id: u64) -> Option<Job> {
        self.jobs.lock().await.get(&id).cloned()
    }

    /// List all jobs with their current status.
    #[allow(dead_code)]
    pub async fn list(&self) -> Vec<Job> {
        let jobs = self.jobs.lock().await;
        let mut jobs: Vec<_> = jobs.values().cloned().collect();
        jobs.sort_by_key(|j| j.created_at);
        jobs
    }

    /// Update a job's state (called by the worker).
    pub async fn update<F>(&self, id: u64, f: F)
    where
        F: FnOnce(&mut Job),
    {
        let mut jobs = self.jobs.lock().await;
        if let Some(job) = jobs.get_mut(&id) {
            f(job);
        }
    }

    /// Return the shared job store for the worker to access directly.
    #[allow(dead_code)]
    pub fn jobs_store(&self) -> Arc<Mutex<HashMap<u64, Job>>> {
        Arc::clone(&self.jobs)
    }

    /// Return the total number of tracked jobs.
    #[allow(dead_code)]
    pub async fn count(&self) -> usize {
        self.jobs.lock().await.len()
    }

    /// Close the submission channel by dropping the sender.
    ///
    /// This signals to the worker that no more jobs will be submitted,
    /// allowing `Worker::run()` to exit after processing remaining jobs.
    #[allow(dead_code)]
    pub async fn close(&self) {
        self.tx.lock().await.take();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::{JobResult, JobStatus};

    fn make_queue() -> (JobQueue, mpsc::Receiver<JobSubmission>) {
        JobQueue::new()
    }

    #[tokio::test]
    async fn submit_cre_job_in_queued_state() {
        let (queue, _rx) = make_queue();
        let id = queue.submit("/tmp/file.pdf".into(), None).await;
        assert_eq!(id, 1);

        let job = queue.get(id).await.unwrap();
        assert_eq!(job.status, JobStatus::Queued);
        assert_eq!(job.file_path.as_deref(), Some("/tmp/file.pdf"));
    }

    #[tokio::test]
    async fn submit_generates_sequential_ids() {
        let (queue, _rx) = make_queue();
        let id1 = queue.submit("/tmp/a.pdf".into(), None).await;
        let id2 = queue.submit("/tmp/b.pdf".into(), None).await;
        let id3 = queue.submit("/tmp/c.pdf".into(), None).await;

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[tokio::test]
    async fn get_returns_none_for_unknown_job() {
        let (queue, _rx) = make_queue();
        assert!(queue.get(999).await.is_none());
    }

    #[tokio::test]
    async fn update_modifies_job() {
        let (queue, _rx) = make_queue();
        let id = queue.submit("/tmp/file.pdf".into(), None).await;

        queue.update(id, |job| {
            job.complete(JobResult {
                nodes_created: 42,
                edges_created: 38,
            });
        })
        .await;

        let job = queue.get(id).await.unwrap();
        assert_eq!(job.status, JobStatus::Complete);
        assert_eq!(job.result.as_ref().unwrap().nodes_created, 42);
        assert_eq!(job.result.as_ref().unwrap().edges_created, 38);
    }

    #[tokio::test]
    async fn list_returns_all_jobs_sorted() {
        let (queue, _rx) = make_queue();
        queue.submit("/tmp/a.pdf".into(), None).await;
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        queue.submit("/tmp/b.pdf".into(), None).await;
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        queue.submit("/tmp/c.pdf".into(), None).await;

        let jobs = queue.list().await;
        assert_eq!(jobs.len(), 3);
        assert_eq!(jobs[0].id, 1);
        assert_eq!(jobs[1].id, 2);
        assert_eq!(jobs[2].id, 3);
    }

    #[tokio::test]
    async fn submit_sends_through_channel() {
        let (queue, mut rx) = make_queue();
        let id = queue.submit("/tmp/test.pdf".into(), Some("My Doc".into())).await;

        let submission = rx.recv().await.unwrap();
        assert_eq!(submission.id, id);
        assert_eq!(submission.file_path, "/tmp/test.pdf");
        assert_eq!(submission.title.as_deref(), Some("My Doc"));
    }

    #[tokio::test]
    async fn jobs_store_is_shared() {
        let (queue, _rx) = make_queue();
        let store = queue.jobs_store();

        queue.submit("/tmp/x.pdf".into(), None).await;

        let jobs = store.lock().await;
        assert_eq!(jobs.len(), 1);
        assert!(jobs.contains_key(&1));
    }

    #[tokio::test]
    async fn count_tracks_jobs() {
        let (queue, _rx) = make_queue();
        assert_eq!(queue.count().await, 0);

        queue.submit("/tmp/a.pdf".into(), None).await;
        queue.submit("/tmp/b.pdf".into(), None).await;

        assert_eq!(queue.count().await, 2);
    }

    #[tokio::test]
    async fn job_terminal_state() {
        use super::Job;

        let mut job = Job::new(1, "/tmp/x.pdf".into(), None);
        assert!(!job.is_terminal());

        job.start();
        assert!(!job.is_terminal());

        job.complete(JobResult {
            nodes_created: 0,
            edges_created: 0,
        });
        assert!(job.is_terminal());

        let mut job2 = Job::new(2, "/tmp/y.pdf".into(), None);
        job2.fail("boom".into());
        assert!(job2.is_terminal());
    }
}
