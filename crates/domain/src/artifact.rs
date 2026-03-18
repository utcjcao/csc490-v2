use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::DomainError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArtifactType {
    NeuronBounds,
    SearchDecisions,
    Certificates,
    Counterexamples,
    VerifierSpecific(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactBundle {
    pub id: Uuid,
    pub source_run_id: Uuid,
    pub artifact_type: ArtifactType,
    pub storage_uri: String,
    pub artifact_hash: String,
    pub validity_scope: String,
    pub verifier_profile_id: Uuid,
    pub schema_version: String,
    pub created_at: DateTime<Utc>,
}

impl ArtifactBundle {
    pub fn new(
        source_run_id: Uuid,
        artifact_type: ArtifactType,
        storage_uri: String,
        artifact_hash: String,
        validity_scope: String,
        verifier_profile_id: Uuid,
        schema_version: String,
    ) -> Result<Self, DomainError> {
        if storage_uri.trim().is_empty() {
            return Err(DomainError::validation("artifact storage URI cannot be empty"));
        }

        Ok(Self {
            id: Uuid::new_v4(),
            source_run_id,
            artifact_type,
            storage_uri,
            artifact_hash,
            validity_scope,
            verifier_profile_id,
            schema_version,
            created_at: Utc::now(),
        })
    }
}
