use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::canonical::CanonicalVerificationJobResult;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
}

impl ErrorResponse {
    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self { code: "not_implemented".to_string(), message: message.into() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CreateProjectRequest {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectResponse {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationModeDto {
    Full,
    Incremental,
    Audit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegisterModelRequest {
    pub project_id: Uuid,
    pub lineage_id: Uuid,
    pub parent_model_id: Option<Uuid>,
    pub format: String,
    pub sha256: String,
    pub architecture_fingerprint: String,
    pub weights_digest: String,
    pub transform_type: String,
    pub transform_metadata: Option<String>,
    pub storage_uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegisterPropertyRequest {
    pub project_id: Uuid,
    pub property_type: String,
    pub input_region: String,
    pub output_constraint: String,
    pub normalization: Option<String>,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CreateVerificationRunRequest {
    pub project_id: Uuid,
    pub model_id: Uuid,
    pub property_id: Uuid,
    pub verifier_profile_id: Uuid,
    pub mode: VerificationModeDto,
    pub reuse_plan_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerificationRunResponse {
    pub run_id: Uuid,
    pub status: String,
    pub outcome: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalVerificationRequest {
    pub model_storage_uri: String,
    pub model_sha256: String,
    pub input_region: String,
    pub output_constraint: String,
    pub timeout_seconds: u64,
    pub memory_mb: u64,
}

pub type CanonicalVerificationResponse = CanonicalVerificationJobResult;
