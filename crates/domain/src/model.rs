use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::DomainError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelFormat {
    Onnx,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TransformType {
    Root,
    CheckpointUpdate,
    OnnxExport,
    Quantization,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelVersion {
    pub id: Uuid,
    pub lineage_id: Uuid,
    pub parent_model_id: Option<Uuid>,
    pub format: ModelFormat,
    pub sha256: String,
    pub architecture_fingerprint: String,
    pub weights_digest: String,
    pub transform_type: TransformType,
    pub transform_metadata: Option<String>,
    pub storage_uri: String,
    pub created_at: DateTime<Utc>,
}

impl ModelVersion {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lineage_id: Uuid,
        parent_model_id: Option<Uuid>,
        format: ModelFormat,
        sha256: String,
        architecture_fingerprint: String,
        weights_digest: String,
        transform_type: TransformType,
        transform_metadata: Option<String>,
        storage_uri: String,
    ) -> Result<Self, DomainError> {
        let sha256 = sha256.trim().to_ascii_lowercase();
        if sha256.is_empty() {
            return Err(DomainError::validation("model sha256 cannot be empty"));
        }

        let storage_uri = storage_uri.trim().to_string();
        if storage_uri.is_empty() {
            return Err(DomainError::validation("model storage URI cannot be empty"));
        }

        Ok(Self {
            id: Uuid::new_v4(),
            lineage_id,
            parent_model_id,
            format,
            sha256,
            architecture_fingerprint,
            weights_digest,
            transform_type,
            transform_metadata,
            storage_uri,
            created_at: Utc::now(),
        })
    }
}
