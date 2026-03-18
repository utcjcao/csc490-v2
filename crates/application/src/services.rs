use std::sync::Arc;

use ivm_contracts::canonical::CanonicalVerificationJobResult;
use ivm_domain::{
    ExecutionLimits, LabelConstraint, LinfInputRegion, ModelFormat, ModelVersion,
    NormalizedExecutionSpec, Project, PropertySpec, PropertyType, TransformType, VerificationMode,
    VerificationRun, VerifierProfile,
};
use serde::Deserialize;
use uuid::Uuid;

use crate::{
    AppError, ArtifactRepository, BlobStore, CanonicalVerificationExecutor,
    CanonicalVerificationRunRepository, ChangeSetRepository, JobQueue, ModelRepository,
    PersistedCanonicalVerificationRun, PersistedRunErrorSnapshot, ProjectRepository,
    PropertyRepository, ReusePlanRepository, VerificationRunRepository, VerifierProfileRepository,
    WorkerLauncher,
};

#[derive(Clone)]
pub struct ApplicationServices {
    pub projects: Arc<dyn ProjectRepository>,
    pub models: Arc<dyn ModelRepository>,
    pub properties: Arc<dyn PropertyRepository>,
    pub verifier_profiles: Arc<dyn VerifierProfileRepository>,
    pub change_sets: Arc<dyn ChangeSetRepository>,
    pub reuse_plans: Arc<dyn ReusePlanRepository>,
    pub artifacts: Arc<dyn ArtifactRepository>,
    pub verification_runs: Arc<dyn VerificationRunRepository>,
    pub blob_store: Arc<dyn BlobStore>,
    pub job_queue: Arc<dyn JobQueue>,
    pub worker_launcher: Arc<dyn WorkerLauncher>,
}

#[derive(Debug, Clone)]
pub struct CreateProjectInput {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RegisterModelInput {
    pub lineage_id: Uuid,
    pub parent_model_id: Option<Uuid>,
    pub format: ModelFormat,
    pub sha256: String,
    pub architecture_fingerprint: String,
    pub weights_digest: String,
    pub transform_type: TransformType,
    pub transform_metadata: Option<String>,
    pub storage_uri: String,
}

#[derive(Debug, Clone)]
pub struct RegisterPropertyInput {
    pub project_id: Uuid,
    pub property_type: PropertyType,
    pub input_region: String,
    pub output_constraint: String,
    pub normalization: Option<String>,
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct RegisterVerifierProfileInput {
    pub name: String,
    pub version: String,
    pub adapter_image: String,
    pub supported_formats: Vec<ModelFormat>,
    pub supported_property_types: Vec<PropertyType>,
    pub artifact_types: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CreateVerificationRunInput {
    pub project_id: Uuid,
    pub model_id: Uuid,
    pub property_id: Uuid,
    pub verifier_profile_id: Uuid,
    pub mode: VerificationMode,
    pub reuse_plan_id: Option<Uuid>,
}

#[derive(Debug, Clone)]
pub struct CanonicalVerificationInput {
    pub model_storage_uri: String,
    pub model_sha256: String,
    pub input_region: String,
    pub output_constraint: String,
    pub timeout_seconds: u64,
    pub memory_mb: u64,
}

#[derive(Clone)]
pub struct VerificationApplication {
    services: ApplicationServices,
}

#[derive(Clone)]
pub struct CanonicalVerificationService {
    executor: Arc<dyn CanonicalVerificationExecutor>,
    runs: Arc<dyn CanonicalVerificationRunRepository>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInputRegion {
    eps: f64,
    #[serde(default)]
    norm: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutputConstraint {
    label: u32,
}

impl VerificationApplication {
    pub fn new(services: ApplicationServices) -> Self {
        Self { services }
    }

    pub async fn create_project(&self, input: CreateProjectInput) -> Result<Project, AppError> {
        let project = Project::new(input.name, input.description)?;
        self.services.projects.create(project).await
    }

    pub async fn register_model(
        &self,
        input: RegisterModelInput,
    ) -> Result<ModelVersion, AppError> {
        let model = ModelVersion::new(
            input.lineage_id,
            input.parent_model_id,
            input.format,
            input.sha256,
            input.architecture_fingerprint,
            input.weights_digest,
            input.transform_type,
            input.transform_metadata,
            input.storage_uri,
        )?;

        self.services.models.create(model).await
    }

    pub async fn register_property(
        &self,
        input: RegisterPropertyInput,
    ) -> Result<PropertySpec, AppError> {
        let property = PropertySpec::new(
            input.project_id,
            input.property_type,
            input.input_region,
            input.output_constraint,
            input.normalization,
            input.sha256,
        )?;

        self.services.properties.create(property).await
    }

    pub async fn register_verifier_profile(
        &self,
        input: RegisterVerifierProfileInput,
    ) -> Result<VerifierProfile, AppError> {
        let profile = VerifierProfile::new(
            input.name,
            input.version,
            input.adapter_image,
            input.supported_formats,
            input.supported_property_types,
            input.artifact_types,
        )?;

        self.services.verifier_profiles.create(profile).await
    }

    pub async fn create_verification_run(
        &self,
        input: CreateVerificationRunInput,
    ) -> Result<VerificationRun, AppError> {
        let run = VerificationRun::new(
            input.project_id,
            input.model_id,
            input.property_id,
            input.verifier_profile_id,
            input.mode,
            input.reuse_plan_id,
        );

        let persisted = self.services.verification_runs.create(run).await?;

        // TODO: materialize a real worker manifest and enqueue it once repositories and contracts are wired.
        let _ = &self.services.job_queue;
        let _ = &self.services.worker_launcher;
        let _ = &self.services.blob_store;
        let _ = &self.services.change_sets;
        let _ = &self.services.reuse_plans;
        let _ = &self.services.artifacts;

        Ok(persisted)
    }

    pub async fn preview_reuse_plan(
        &self,
        _source_model_id: Uuid,
        _target_model_id: Uuid,
    ) -> Result<(), AppError> {
        Err(AppError::not_implemented(
            "reuse planning is intentionally deferred until ONNX diffing and artifact policies are implemented",
        ))
    }
}

impl CanonicalVerificationService {
    pub fn new(
        executor: Arc<dyn CanonicalVerificationExecutor>,
        runs: Arc<dyn CanonicalVerificationRunRepository>,
    ) -> Self {
        Self { executor, runs }
    }

    pub fn normalize_input(
        input: CanonicalVerificationInput,
    ) -> Result<NormalizedExecutionSpec, AppError> {
        normalize_canonical_input(input)
    }

    pub async fn get_run(
        &self,
        run_id: Uuid,
    ) -> Result<PersistedCanonicalVerificationRun, AppError> {
        self.runs
            .get(run_id)
            .await?
            .ok_or_else(|| AppError::NotFound(format!("verification run {run_id} was not found")))
    }

    pub async fn run(
        &self,
        input: CanonicalVerificationInput,
    ) -> Result<CanonicalVerificationJobResult, AppError> {
        let spec = Self::normalize_input(input)?;
        let mut run = PersistedCanonicalVerificationRun::new_pending(
            spec.clone(),
            self.executor.backend_identifier().to_string(),
        )?;

        run = self.runs.create(run).await?;
        run.mark_running()?;
        run = self.runs.update(run).await?;

        let result = match self.executor.execute(spec).await {
            Ok(result) => result,
            Err(error) => return self.persist_failure_and_return(run, error).await,
        };

        if let Err(error) = run.apply_result(result.clone()) {
            return self.persist_failure_and_return(run, error).await;
        }

        self.runs.update(run).await?;
        Ok(result)
    }
}

impl CanonicalVerificationService {
    async fn persist_failure_and_return(
        &self,
        mut run: PersistedCanonicalVerificationRun,
        error: AppError,
    ) -> Result<CanonicalVerificationJobResult, AppError> {
        let snapshot = PersistedRunErrorSnapshot::from_app_error(&error);
        match run.mark_failed(snapshot) {
            Ok(()) => {
                if let Err(persist_error) = self.runs.update(run).await {
                    return Err(AppError::external_dependency(format!(
                        "execution failed with '{error}', and persisting the failed run also failed: {persist_error}"
                    )));
                }
            }
            Err(mark_error) => {
                return Err(AppError::invariant_violation(format!(
                    "execution failed with '{error}', but the persisted run could not transition to failed: {mark_error}"
                )));
            }
        }

        Err(error)
    }
}

fn normalize_canonical_input(
    input: CanonicalVerificationInput,
) -> Result<NormalizedExecutionSpec, AppError> {
    validate_required_fields(&input)?;

    let model_storage_uri = normalize_model_storage_uri(&input.model_storage_uri)?;
    let model_sha256 = normalize_model_sha256(&input.model_sha256);
    let input_region = parse_and_validate_input_region(&input.input_region)?;
    let output_constraint = parse_and_validate_output_constraint(&input.output_constraint)?;
    let limits = validate_execution_limits(input.timeout_seconds, input.memory_mb)?;

    let project = Project::new(
        "canonical-demo".to_string(),
        Some("Phase 1 canonical verification flow".to_string()),
    )?;

    let model = ModelVersion::new(
        Uuid::new_v4(),
        None,
        ModelFormat::Onnx,
        model_sha256.clone(),
        "canonical-demo-onnx".to_string(),
        model_sha256,
        TransformType::Root,
        Some("generated by canonical request normalization".to_string()),
        model_storage_uri,
    )?;

    let property = PropertySpec::new(
        project.id,
        PropertyType::LocalRobustnessLinf,
        input_region.to_normalized_json(),
        output_constraint.to_normalized_json(),
        Some(r#"{"norm":"linf"}"#.to_string()),
        build_property_digest(&input_region, &output_constraint),
    )?;

    let verifier = VerifierProfile::new(
        "alpha-beta-crown".to_string(),
        "stub-0.1.0".to_string(),
        "python -m alpha_beta_crown_adapter".to_string(),
        vec![ModelFormat::Onnx],
        vec![PropertyType::LocalRobustnessLinf],
        vec!["stub-proof-summary".to_string()],
    )?;

    let run = VerificationRun::new(
        project.id,
        model.id,
        property.id,
        verifier.id,
        VerificationMode::Full,
        None,
    );

    NormalizedExecutionSpec::new(
        run,
        model,
        property,
        verifier,
        limits,
        input_region,
        output_constraint,
    )
    .map_err(AppError::from)
}

fn validate_required_fields(input: &CanonicalVerificationInput) -> Result<(), AppError> {
    if input.model_storage_uri.trim().is_empty() {
        return Err(AppError::input_validation("model_storage_uri cannot be empty"));
    }

    if input.model_sha256.trim().is_empty() {
        return Err(AppError::input_validation("model_sha256 cannot be empty"));
    }

    if input.input_region.trim().is_empty() {
        return Err(AppError::input_validation("input_region cannot be empty"));
    }

    if input.output_constraint.trim().is_empty() {
        return Err(AppError::input_validation("output_constraint cannot be empty"));
    }

    Ok(())
}

fn normalize_model_storage_uri(value: &str) -> Result<String, AppError> {
    let normalized = value.trim().to_string();

    if normalized.chars().any(char::is_whitespace) {
        return Err(AppError::input_validation("model_storage_uri cannot contain whitespace"));
    }

    if !normalized.contains("://") {
        return Err(AppError::input_validation("model_storage_uri must include a URI scheme"));
    }

    if !normalized.to_ascii_lowercase().ends_with(".onnx") {
        return Err(AppError::unsupported_feature(
            "only ONNX model artifacts are supported in Phase 1",
        ));
    }

    Ok(normalized)
}

fn normalize_model_sha256(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn parse_and_validate_input_region(raw: &str) -> Result<LinfInputRegion, AppError> {
    let payload: RawInputRegion = parse_json_field("input_region", raw)?;
    let norm = payload.norm.unwrap_or_else(|| "linf".to_string());

    if !norm.eq_ignore_ascii_case("linf") {
        return Err(AppError::unsupported_feature(format!(
            "input_region.norm '{}' is not supported; only 'linf' is allowed in Phase 1",
            norm
        )));
    }

    if !payload.eps.is_finite() {
        return Err(AppError::input_validation("input_region.eps must be a finite number"));
    }

    if !(0.0 < payload.eps && payload.eps <= 1.0) {
        return Err(AppError::input_validation(
            "input_region.eps must be greater than 0 and less than or equal to 1",
        ));
    }

    let eps_micros = (payload.eps * 1_000_000.0).round() as u64;
    LinfInputRegion::new(eps_micros).map_err(AppError::from)
}

fn parse_and_validate_output_constraint(raw: &str) -> Result<LabelConstraint, AppError> {
    let payload: RawOutputConstraint = parse_json_field("output_constraint", raw)?;
    LabelConstraint::new(payload.label).map_err(AppError::from)
}

fn validate_execution_limits(
    timeout_seconds: u64,
    memory_mb: u64,
) -> Result<ExecutionLimits, AppError> {
    if timeout_seconds > 3_600 {
        return Err(AppError::unsupported_feature(
            "timeouts above 3600 seconds are not supported in Phase 1",
        ));
    }

    if memory_mb > 16_384 {
        return Err(AppError::unsupported_feature(
            "memory limits above 16384 MB are not supported in Phase 1",
        ));
    }

    ExecutionLimits::new(timeout_seconds, memory_mb).map_err(AppError::from)
}

fn parse_json_field<T>(field_name: &str, raw: &str) -> Result<T, AppError>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_str(raw).map_err(|error| {
        AppError::input_validation(format!(
            "{field_name} must be valid JSON matching the canonical schema: {error}"
        ))
    })
}

fn build_property_digest(
    input_region: &LinfInputRegion,
    output_constraint: &LabelConstraint,
) -> String {
    format!("canonical-property:linf:{}:{}", input_region.eps_micros(), output_constraint.label())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    use async_trait::async_trait;
    use ivm_contracts::canonical::{CanonicalVerificationJobResult, CanonicalVerificationMetrics};
    use uuid::Uuid;

    use super::{
        AppError, CanonicalVerificationExecutor, CanonicalVerificationInput,
        CanonicalVerificationRunRepository, CanonicalVerificationService, NormalizedExecutionSpec,
        PersistedCanonicalVerificationRun,
    };
    use crate::PersistedRunStatus;

    #[derive(Default)]
    struct RecordingExecutor {
        seen_spec: Mutex<Option<NormalizedExecutionSpec>>,
    }

    #[async_trait]
    impl CanonicalVerificationExecutor for RecordingExecutor {
        fn backend_identifier(&self) -> &str {
            "recording-executor"
        }

        async fn execute(
            &self,
            spec: NormalizedExecutionSpec,
        ) -> Result<CanonicalVerificationJobResult, AppError> {
            let run_id = spec.run.id;
            *self.seen_spec.lock().expect("lock should succeed") = Some(spec.clone());

            Ok(CanonicalVerificationJobResult {
                run_id,
                status: "completed".to_string(),
                outcome: "proved".to_string(),
                verifier_name: spec.verifier.name,
                summary: "stub".to_string(),
                metrics: CanonicalVerificationMetrics::default(),
                failure: None,
            })
        }
    }

    struct FailingExecutor;

    #[async_trait]
    impl CanonicalVerificationExecutor for FailingExecutor {
        fn backend_identifier(&self) -> &str {
            "failing-executor"
        }

        async fn execute(
            &self,
            _spec: NormalizedExecutionSpec,
        ) -> Result<CanonicalVerificationJobResult, AppError> {
            Err(AppError::adapter_runtime("simulated adapter failure"))
        }
    }

    #[derive(Default)]
    struct InMemoryRunRepository {
        runs: Mutex<HashMap<Uuid, PersistedCanonicalVerificationRun>>,
    }

    #[async_trait]
    impl CanonicalVerificationRunRepository for InMemoryRunRepository {
        async fn create(
            &self,
            run: PersistedCanonicalVerificationRun,
        ) -> Result<PersistedCanonicalVerificationRun, AppError> {
            self.runs.lock().expect("lock should succeed").insert(run.run_id, run.clone());
            Ok(run)
        }

        async fn get(
            &self,
            run_id: Uuid,
        ) -> Result<Option<PersistedCanonicalVerificationRun>, AppError> {
            Ok(self.runs.lock().expect("lock should succeed").get(&run_id).cloned())
        }

        async fn update(
            &self,
            run: PersistedCanonicalVerificationRun,
        ) -> Result<PersistedCanonicalVerificationRun, AppError> {
            self.runs.lock().expect("lock should succeed").insert(run.run_id, run.clone());
            Ok(run)
        }
    }

    fn valid_input() -> CanonicalVerificationInput {
        CanonicalVerificationInput {
            model_storage_uri: " demo://models/canonical-demo.onnx ".to_string(),
            model_sha256: " Demo-Model-SHA256 ".to_string(),
            input_region: r#"{"eps":0.01}"#.to_string(),
            output_constraint: r#"{"label":1}"#.to_string(),
            timeout_seconds: 60,
            memory_mb: 1024,
        }
    }

    #[test]
    fn canonical_request_maps_to_normalized_execution_spec() {
        let spec = CanonicalVerificationService::normalize_input(valid_input())
            .expect("normalization should succeed");

        assert_eq!(spec.model.storage_uri, "demo://models/canonical-demo.onnx");
        assert_eq!(spec.model.sha256, "demo-model-sha256");
        assert_eq!(spec.property.input_region, r#"{"eps":0.01,"norm":"linf"}"#);
        assert_eq!(spec.property.output_constraint, r#"{"label":1}"#);
        assert_eq!(spec.limits.timeout_seconds, 60);
        assert_eq!(spec.limits.memory_mb, 1024);
    }

    #[test]
    fn normalization_rejects_semantically_invalid_epsilon() {
        let mut input = valid_input();
        input.input_region = r#"{"eps":0}"#.to_string();

        let error = CanonicalVerificationService::normalize_input(input)
            .expect_err("invalid epsilon should fail");

        assert_eq!(
            error.to_string(),
            "input validation error: input_region.eps must be greater than 0 and less than or equal to 1"
        );
    }

    #[test]
    fn normalization_rejects_unsupported_norm_with_explicit_error() {
        let mut input = valid_input();
        input.input_region = r#"{"eps":0.01,"norm":"l2"}"#.to_string();

        let error = CanonicalVerificationService::normalize_input(input)
            .expect_err("unsupported norm should fail");

        assert_eq!(
            error.to_string(),
            "unsupported feature error: input_region.norm 'l2' is not supported; only 'linf' is allowed in Phase 1"
        );
    }

    #[tokio::test]
    async fn run_preserves_adapter_runtime_error_classification() {
        let repo = Arc::new(InMemoryRunRepository::default());
        let service = CanonicalVerificationService::new(Arc::new(FailingExecutor), repo.clone());

        let error = service.run(valid_input()).await.expect_err("adapter failure should surface");

        assert_eq!(error.to_string(), "adapter/runtime error: simulated adapter failure");

        let run = repo
            .runs
            .lock()
            .expect("lock should succeed")
            .values()
            .next()
            .cloned()
            .expect("failed run should be persisted");
        assert_eq!(run.status, PersistedRunStatus::Failed);
        assert_eq!(
            run.error_snapshot.expect("error snapshot should exist").code,
            "adapter_runtime_error"
        );
    }

    #[tokio::test]
    async fn run_passes_normalized_spec_to_executor() {
        let executor = Arc::new(RecordingExecutor::default());
        let repo = Arc::new(InMemoryRunRepository::default());
        let service = CanonicalVerificationService::new(executor.clone(), repo.clone());

        service.run(valid_input()).await.expect("run should succeed");

        let seen = executor
            .seen_spec
            .lock()
            .expect("lock should succeed")
            .clone()
            .expect("executor should receive a spec");

        assert_eq!(seen.property.input_region, r#"{"eps":0.01,"norm":"linf"}"#);
        assert_eq!(seen.output_constraint.label(), 1);

        let run = repo
            .runs
            .lock()
            .expect("lock should succeed")
            .values()
            .next()
            .cloned()
            .expect("completed run should be persisted");
        assert_eq!(run.status, PersistedRunStatus::Completed);
        assert!(run.result_snapshot.is_some());
    }

    #[tokio::test]
    async fn invalid_request_does_not_create_a_run() {
        let executor = Arc::new(RecordingExecutor::default());
        let repo = Arc::new(InMemoryRunRepository::default());
        let service = CanonicalVerificationService::new(executor, repo.clone());

        let mut input = valid_input();
        input.input_region = r#"{"eps":0}"#.to_string();

        let error = service.run(input).await.expect_err("invalid request should fail");

        assert_eq!(
            error.to_string(),
            "input validation error: input_region.eps must be greater than 0 and less than or equal to 1"
        );
        assert!(repo.runs.lock().expect("lock should succeed").is_empty());
    }
}
