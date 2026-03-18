use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::DomainError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Project {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl Project {
    pub fn new(name: String, description: Option<String>) -> Result<Self, DomainError> {
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(DomainError::validation("project name cannot be empty"));
        }

        Ok(Self { id: Uuid::new_v4(), name, description, created_at: Utc::now() })
    }
}

#[cfg(test)]
mod tests {
    use super::Project;

    #[test]
    fn rejects_empty_name() {
        let result = Project::new(String::new(), None);
        assert!(result.is_err());
    }
}
