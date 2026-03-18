use async_trait::async_trait;
use uuid::Uuid;

use ivm_application::{AppError, WorkerLauncher};

#[derive(Debug, Clone, Default)]
pub struct NoopWorkerLauncher;

#[async_trait]
impl WorkerLauncher for NoopWorkerLauncher {
    async fn launch(&self, _run_id: Uuid) -> Result<(), AppError> {
        Err(AppError::not_implemented(
            "NoopWorkerLauncher::launch is a scaffold placeholder; containerized worker execution is not wired yet",
        ))
    }
}
