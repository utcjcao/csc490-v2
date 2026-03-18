use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelReference {
    pub model_id: Uuid,
    pub storage_uri: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PropertyPayload {
    pub property_id: Uuid,
    pub property_type: String,
    pub input_region: String,
    pub output_constraint: String,
    pub normalization: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactReference {
    pub artifact_id: Uuid,
    pub artifact_type: String,
    pub storage_uri: String,
    pub artifact_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BaselineContext {
    pub run_id: Uuid,
    pub artifact_refs: Vec<ArtifactReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecomputeStepPayload {
    pub name: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReusePlanPayload {
    pub reuse_plan_id: Uuid,
    pub selected_artifact_ids: Vec<Uuid>,
    pub recompute_steps: Vec<RecomputeStepPayload>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerifierProfilePayload {
    pub verifier_profile_id: Uuid,
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionLimits {
    pub timeout_seconds: u64,
    pub memory_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunManifest {
    pub run_id: Uuid,
    pub mode: String,
    pub model: ModelReference,
    pub property: PropertyPayload,
    pub baseline: Option<BaselineContext>,
    pub reuse_plan: Option<ReusePlanPayload>,
    pub verifier_profile: VerifierProfilePayload,
    pub limits: ExecutionLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RunMetricsPayload {
    pub wall_time_ms: u64,
    pub cpu_time_ms: u64,
    pub reused_artifact_count: u32,
    pub recomputed_step_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactManifest {
    pub artifact_type: String,
    pub storage_uri: String,
    pub artifact_hash: String,
    pub validity_scope: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CounterexamplePayload {
    pub input_blob_uri: String,
    pub observed_output: String,
    pub expected_constraint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerFailure {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResultManifest {
    pub run_id: Uuid,
    pub status: String,
    pub outcome: String,
    pub metrics: RunMetricsPayload,
    pub artifacts: Vec<ArtifactManifest>,
    pub counterexample: Option<CounterexamplePayload>,
    pub failure: Option<WorkerFailure>,
}
