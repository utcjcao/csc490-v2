use chrono::{DateTime, Utc};
use ivm_contracts::canonical::CanonicalVerificationJobResult;
use ivm_domain::NormalizedExecutionSpec;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::AppError;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PersistedRunStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl PersistedRunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistedRunErrorSnapshot {
    pub kind: String,
    pub code: String,
    pub message: String,
}

impl PersistedRunErrorSnapshot {
    pub fn from_app_error(error: &AppError) -> Self {
        match error {
            AppError::InputValidation(message) => Self {
                kind: "input_validation".to_string(),
                code: "input_validation_error".to_string(),
                message: message.clone(),
            },
            AppError::UnsupportedFeature(message) => Self {
                kind: "unsupported_feature".to_string(),
                code: "unsupported_feature_error".to_string(),
                message: message.clone(),
            },
            AppError::NotFound(message) => Self {
                kind: "not_found".to_string(),
                code: "not_found".to_string(),
                message: message.clone(),
            },
            AppError::Conflict(message) => Self {
                kind: "conflict".to_string(),
                code: "conflict".to_string(),
                message: message.clone(),
            },
            AppError::AdapterRuntime(message) => Self {
                kind: "adapter_runtime".to_string(),
                code: "adapter_runtime_error".to_string(),
                message: message.clone(),
            },
            AppError::ExternalDependency(message) => Self {
                kind: "external_dependency".to_string(),
                code: "external_dependency_error".to_string(),
                message: message.clone(),
            },
            AppError::InvariantViolation(message) => Self {
                kind: "invariant_violation".to_string(),
                code: "internal_invariant_violation".to_string(),
                message: message.clone(),
            },
            AppError::NotImplemented(message) => Self {
                kind: "not_implemented".to_string(),
                code: "not_implemented".to_string(),
                message: message.clone(),
            },
        }
    }

    pub fn from_result(result: &CanonicalVerificationJobResult) -> Option<Self> {
        result.failure.as_ref().map(|failure| Self {
            kind: "verifier_failure".to_string(),
            code: failure.code.clone(),
            message: failure.message.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistedCanonicalVerificationRun {
    pub run_id: Uuid,
    pub status: PersistedRunStatus,
    pub submitted_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub normalized_execution_spec_snapshot: NormalizedExecutionSpec,
    pub result_snapshot: Option<CanonicalVerificationJobResult>,
    pub error_snapshot: Option<PersistedRunErrorSnapshot>,
    pub backend_identifier: String,
}

impl PersistedCanonicalVerificationRun {
    pub fn new_pending(
        spec: NormalizedExecutionSpec,
        backend_identifier: String,
    ) -> Result<Self, AppError> {
        let backend_identifier = backend_identifier.trim().to_string();
        if backend_identifier.is_empty() {
            return Err(AppError::invariant_violation("backend identifier cannot be empty"));
        }

        let now = Utc::now();
        Ok(Self {
            run_id: spec.run.id,
            status: PersistedRunStatus::Pending,
            submitted_at: now,
            updated_at: now,
            normalized_execution_spec_snapshot: spec,
            result_snapshot: None,
            error_snapshot: None,
            backend_identifier,
        })
    }

    pub fn mark_running(&mut self) -> Result<(), AppError> {
        if self.status != PersistedRunStatus::Pending {
            return Err(AppError::invariant_violation(format!(
                "cannot transition persisted run {} from {} to running",
                self.run_id,
                self.status.as_str()
            )));
        }

        self.status = PersistedRunStatus::Running;
        self.updated_at = Utc::now();
        Ok(())
    }

    pub fn apply_result(&mut self, result: CanonicalVerificationJobResult) -> Result<(), AppError> {
        if self.status != PersistedRunStatus::Running {
            return Err(AppError::invariant_violation(format!(
                "cannot apply a result to persisted run {} while in {} state",
                self.run_id,
                self.status.as_str()
            )));
        }

        if result.run_id != self.run_id {
            return Err(AppError::adapter_runtime(
                "adapter returned a run ID that did not match the submitted request",
            ));
        }

        self.status = match result.status.as_str() {
            "completed" => PersistedRunStatus::Completed,
            "failed" => PersistedRunStatus::Failed,
            other => {
                return Err(AppError::adapter_runtime(format!(
                    "adapter returned unsupported terminal status '{other}'"
                )));
            }
        };
        self.updated_at = Utc::now();
        self.error_snapshot = PersistedRunErrorSnapshot::from_result(&result);
        self.result_snapshot = Some(result);
        Ok(())
    }

    pub fn mark_failed(&mut self, error: PersistedRunErrorSnapshot) -> Result<(), AppError> {
        match self.status {
            PersistedRunStatus::Pending | PersistedRunStatus::Running => {
                self.status = PersistedRunStatus::Failed;
                self.updated_at = Utc::now();
                self.error_snapshot = Some(error);
                Ok(())
            }
            _ => Err(AppError::invariant_violation(format!(
                "cannot mark persisted run {} as failed from {} state",
                self.run_id,
                self.status.as_str()
            ))),
        }
    }
}
