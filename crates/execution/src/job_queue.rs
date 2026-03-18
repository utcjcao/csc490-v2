use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use ivm_application::{AppError, JobQueue};
use ivm_contracts::worker::RunManifest;

#[derive(Debug, Clone, Default)]
pub struct InMemoryJobQueue {
    manifests: Arc<Mutex<Vec<RunManifest>>>,
}

impl InMemoryJobQueue {
    pub async fn pending_count(&self) -> usize {
        self.manifests.lock().await.len()
    }
}

#[async_trait]
impl JobQueue for InMemoryJobQueue {
    async fn enqueue_run(&self, manifest: RunManifest) -> Result<(), AppError> {
        self.manifests.lock().await.push(manifest);
        Ok(())
    }
}
