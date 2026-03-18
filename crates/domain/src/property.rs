use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::DomainError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PropertyType {
    LocalRobustnessLinf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PropertySpec {
    pub id: Uuid,
    pub project_id: Uuid,
    pub property_type: PropertyType,
    pub input_region: String,
    pub output_constraint: String,
    pub normalization: Option<String>,
    pub sha256: String,
    pub created_at: DateTime<Utc>,
}

impl PropertySpec {
    pub fn new(
        project_id: Uuid,
        property_type: PropertyType,
        input_region: String,
        output_constraint: String,
        normalization: Option<String>,
        sha256: String,
    ) -> Result<Self, DomainError> {
        let input_region = input_region.trim().to_string();
        if input_region.is_empty() {
            return Err(DomainError::validation("input_region cannot be empty"));
        }

        let output_constraint = output_constraint.trim().to_string();
        if output_constraint.is_empty() {
            return Err(DomainError::validation("output_constraint cannot be empty"));
        }

        let sha256 = sha256.trim().to_string();
        if sha256.is_empty() {
            return Err(DomainError::validation("property sha256 cannot be empty"));
        }

        Ok(Self {
            id: Uuid::new_v4(),
            project_id,
            property_type,
            input_region,
            output_constraint,
            normalization,
            sha256,
            created_at: Utc::now(),
        })
    }
}
