use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{error::DomainError, model::ModelFormat, property::PropertyType};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerifierProfile {
    pub id: Uuid,
    pub name: String,
    pub version: String,
    pub adapter_image: String,
    pub supported_formats: Vec<ModelFormat>,
    pub supported_property_types: Vec<PropertyType>,
    pub artifact_types: Vec<String>,
    pub created_at: DateTime<Utc>,
}

impl VerifierProfile {
    pub fn new(
        name: String,
        version: String,
        adapter_image: String,
        supported_formats: Vec<ModelFormat>,
        supported_property_types: Vec<PropertyType>,
        artifact_types: Vec<String>,
    ) -> Result<Self, DomainError> {
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(DomainError::validation("verifier name cannot be empty"));
        }

        let version = version.trim().to_string();
        if version.is_empty() {
            return Err(DomainError::validation("verifier version cannot be empty"));
        }

        Ok(Self {
            id: Uuid::new_v4(),
            name,
            version,
            adapter_image,
            supported_formats,
            supported_property_types,
            artifact_types,
            created_at: Utc::now(),
        })
    }
}
