use ivm_domain::DomainError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("input validation error: {0}")]
    InputValidation(String),
    #[error("unsupported feature error: {0}")]
    UnsupportedFeature(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("adapter/runtime error: {0}")]
    AdapterRuntime(String),
    #[error("external dependency error: {0}")]
    ExternalDependency(String),
    #[error("internal invariant violation: {0}")]
    InvariantViolation(String),
    #[error("not implemented: {0}")]
    NotImplemented(String),
}

impl AppError {
    pub fn input_validation(message: impl Into<String>) -> Self {
        Self::InputValidation(message.into())
    }

    pub fn unsupported_feature(message: impl Into<String>) -> Self {
        Self::UnsupportedFeature(message.into())
    }

    pub fn adapter_runtime(message: impl Into<String>) -> Self {
        Self::AdapterRuntime(message.into())
    }

    pub fn external_dependency(message: impl Into<String>) -> Self {
        Self::ExternalDependency(message.into())
    }

    pub fn invariant_violation(message: impl Into<String>) -> Self {
        Self::InvariantViolation(message.into())
    }

    pub fn not_implemented(area: impl Into<String>) -> Self {
        Self::NotImplemented(area.into())
    }
}

impl From<DomainError> for AppError {
    fn from(value: DomainError) -> Self {
        match value {
            DomainError::Validation(message) => Self::InputValidation(message),
            DomainError::InvariantViolation(message) => Self::InvariantViolation(message),
            DomainError::NotImplemented(area) => Self::NotImplemented(area.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AppError;
    use ivm_domain::DomainError;

    #[test]
    fn maps_domain_invariant_violations_to_application_invariant_violations() {
        let error = AppError::from(DomainError::invariant_violation("broken internal state"));

        assert_eq!(error.to_string(), "internal invariant violation: broken internal state");
    }
}
