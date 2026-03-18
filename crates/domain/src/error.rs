use thiserror::Error;

#[derive(Debug, Error)]
pub enum DomainError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("invariant violation: {0}")]
    InvariantViolation(String),
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
}

impl DomainError {
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation(message.into())
    }

    pub fn invariant_violation(message: impl Into<String>) -> Self {
        Self::InvariantViolation(message.into())
    }

    pub fn not_implemented(area: &'static str) -> Self {
        Self::NotImplemented(area)
    }
}
