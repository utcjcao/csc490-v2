use async_trait::async_trait;
use uuid::Uuid;

use ivm_contracts::{canonical::CanonicalVerificationJobResult, worker::RunManifest};
use ivm_domain::{
    ArtifactBundle, ChangeSet, ModelVersion, NormalizedExecutionSpec, Project, PropertySpec,
    ReusePlan, VerificationRun, VerifierProfile,
};

use crate::{AppError, PersistedCanonicalVerificationRun};

#[async_trait]
pub trait ProjectRepository: Send + Sync {
    async fn create(&self, project: Project) -> Result<Project, AppError>;
    async fn get(&self, id: Uuid) -> Result<Option<Project>, AppError>;
}

#[async_trait]
pub trait ModelRepository: Send + Sync {
    async fn create(&self, model: ModelVersion) -> Result<ModelVersion, AppError>;
    async fn get(&self, id: Uuid) -> Result<Option<ModelVersion>, AppError>;
}

#[async_trait]
pub trait PropertyRepository: Send + Sync {
    async fn create(&self, property: PropertySpec) -> Result<PropertySpec, AppError>;
    async fn get(&self, id: Uuid) -> Result<Option<PropertySpec>, AppError>;
}

#[async_trait]
pub trait VerifierProfileRepository: Send + Sync {
    async fn create(&self, profile: VerifierProfile) -> Result<VerifierProfile, AppError>;
    async fn get(&self, id: Uuid) -> Result<Option<VerifierProfile>, AppError>;
}

#[async_trait]
pub trait ChangeSetRepository: Send + Sync {
    async fn save(&self, changeset: ChangeSet) -> Result<ChangeSet, AppError>;
}

#[async_trait]
pub trait ReusePlanRepository: Send + Sync {
    async fn save(&self, reuse_plan: ReusePlan) -> Result<ReusePlan, AppError>;
    async fn get(&self, id: Uuid) -> Result<Option<ReusePlan>, AppError>;
}

#[async_trait]
pub trait ArtifactRepository: Send + Sync {
    async fn save_bundle(&self, bundle: ArtifactBundle) -> Result<ArtifactBundle, AppError>;
    async fn list_for_run(&self, run_id: Uuid) -> Result<Vec<ArtifactBundle>, AppError>;
}

#[async_trait]
pub trait VerificationRunRepository: Send + Sync {
    async fn create(&self, run: VerificationRun) -> Result<VerificationRun, AppError>;
    async fn get(&self, id: Uuid) -> Result<Option<VerificationRun>, AppError>;
    async fn update(&self, run: VerificationRun) -> Result<VerificationRun, AppError>;
}

#[async_trait]
pub trait BlobStore: Send + Sync {
    async fn put_bytes(&self, path: &str, bytes: Vec<u8>) -> Result<String, AppError>;
    async fn get_bytes(&self, path: &str) -> Result<Vec<u8>, AppError>;
}

#[async_trait]
pub trait JobQueue: Send + Sync {
    async fn enqueue_run(&self, manifest: RunManifest) -> Result<(), AppError>;
}

#[async_trait]
pub trait WorkerLauncher: Send + Sync {
    async fn launch(&self, run_id: Uuid) -> Result<(), AppError>;
}

#[async_trait]
pub trait CanonicalVerificationRunRepository: Send + Sync {
    async fn create(
        &self,
        run: PersistedCanonicalVerificationRun,
    ) -> Result<PersistedCanonicalVerificationRun, AppError>;
    async fn get(
        &self,
        run_id: Uuid,
    ) -> Result<Option<PersistedCanonicalVerificationRun>, AppError>;
    async fn update(
        &self,
        run: PersistedCanonicalVerificationRun,
    ) -> Result<PersistedCanonicalVerificationRun, AppError>;
}

#[async_trait]
pub trait CanonicalVerificationExecutor: Send + Sync {
    fn backend_identifier(&self) -> &str;

    async fn execute(
        &self,
        spec: NormalizedExecutionSpec,
    ) -> Result<CanonicalVerificationJobResult, AppError>;
}
