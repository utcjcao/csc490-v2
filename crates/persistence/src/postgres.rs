use async_trait::async_trait;
use uuid::Uuid;

use ivm_application::{
    AppError, ArtifactRepository, ChangeSetRepository, ModelRepository, ProjectRepository,
    PropertyRepository, ReusePlanRepository, VerificationRunRepository, VerifierProfileRepository,
};
use ivm_domain::{
    ArtifactBundle, ChangeSet, ModelVersion, Project, PropertySpec, ReusePlan, VerificationRun,
    VerifierProfile,
};

use crate::config::PostgresConfig;

#[derive(Debug, Clone)]
pub struct PostgresRepositories {
    pub config: PostgresConfig,
}

impl PostgresRepositories {
    pub fn new(config: PostgresConfig) -> Self {
        Self { config }
    }

    fn stub(area: &'static str) -> AppError {
        AppError::not_implemented(format!(
            "{area} is a Phase 1 scaffold stub; replace it with SQLx-backed PostgreSQL persistence"
        ))
    }
}

#[async_trait]
impl ProjectRepository for PostgresRepositories {
    async fn create(&self, _project: Project) -> Result<Project, AppError> {
        Err(Self::stub("ProjectRepository::create"))
    }

    async fn get(&self, _id: Uuid) -> Result<Option<Project>, AppError> {
        Err(Self::stub("ProjectRepository::get"))
    }
}

#[async_trait]
impl ModelRepository for PostgresRepositories {
    async fn create(&self, _model: ModelVersion) -> Result<ModelVersion, AppError> {
        Err(Self::stub("ModelRepository::create"))
    }

    async fn get(&self, _id: Uuid) -> Result<Option<ModelVersion>, AppError> {
        Err(Self::stub("ModelRepository::get"))
    }
}

#[async_trait]
impl PropertyRepository for PostgresRepositories {
    async fn create(&self, _property: PropertySpec) -> Result<PropertySpec, AppError> {
        Err(Self::stub("PropertyRepository::create"))
    }

    async fn get(&self, _id: Uuid) -> Result<Option<PropertySpec>, AppError> {
        Err(Self::stub("PropertyRepository::get"))
    }
}

#[async_trait]
impl VerifierProfileRepository for PostgresRepositories {
    async fn create(&self, _profile: VerifierProfile) -> Result<VerifierProfile, AppError> {
        Err(Self::stub("VerifierProfileRepository::create"))
    }

    async fn get(&self, _id: Uuid) -> Result<Option<VerifierProfile>, AppError> {
        Err(Self::stub("VerifierProfileRepository::get"))
    }
}

#[async_trait]
impl ChangeSetRepository for PostgresRepositories {
    async fn save(&self, _changeset: ChangeSet) -> Result<ChangeSet, AppError> {
        Err(Self::stub("ChangeSetRepository::save"))
    }
}

#[async_trait]
impl ReusePlanRepository for PostgresRepositories {
    async fn save(&self, _reuse_plan: ReusePlan) -> Result<ReusePlan, AppError> {
        Err(Self::stub("ReusePlanRepository::save"))
    }

    async fn get(&self, _id: Uuid) -> Result<Option<ReusePlan>, AppError> {
        Err(Self::stub("ReusePlanRepository::get"))
    }
}

#[async_trait]
impl ArtifactRepository for PostgresRepositories {
    async fn save_bundle(&self, _bundle: ArtifactBundle) -> Result<ArtifactBundle, AppError> {
        Err(Self::stub("ArtifactRepository::save_bundle"))
    }

    async fn list_for_run(&self, _run_id: Uuid) -> Result<Vec<ArtifactBundle>, AppError> {
        Err(Self::stub("ArtifactRepository::list_for_run"))
    }
}

#[async_trait]
impl VerificationRunRepository for PostgresRepositories {
    async fn create(&self, _run: VerificationRun) -> Result<VerificationRun, AppError> {
        Err(Self::stub("VerificationRunRepository::create"))
    }

    async fn get(&self, _id: Uuid) -> Result<Option<VerificationRun>, AppError> {
        Err(Self::stub("VerificationRunRepository::get"))
    }

    async fn update(&self, _run: VerificationRun) -> Result<VerificationRun, AppError> {
        Err(Self::stub("VerificationRunRepository::update"))
    }
}
