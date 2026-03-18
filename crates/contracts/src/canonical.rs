use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalVerificationJobRequest {
    pub run_id: Uuid,
    pub model_storage_uri: String,
    pub model_sha256: String,
    pub input_region: String,
    pub output_constraint: String,
    pub verifier_name: String,
    pub timeout_seconds: u64,
    pub memory_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct CanonicalVerificationMetrics {
    pub wall_time_ms: u64,
    pub cpu_time_ms: u64,
    pub reused_artifact_count: u32,
    pub recomputed_step_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalVerificationFailure {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalVerificationJobResult {
    pub run_id: Uuid,
    pub status: String,
    pub outcome: String,
    pub verifier_name: String,
    pub summary: String,
    pub metrics: CanonicalVerificationMetrics,
    pub failure: Option<CanonicalVerificationFailure>,
}
