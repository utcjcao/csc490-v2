use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationMode {
    Full,
    Incremental,
    Audit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunStatus {
    Pending,
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunOutcome {
    Proved,
    Disproved,
    Inconclusive,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RunMetrics {
    pub wall_time_ms: Option<u64>,
    pub cpu_time_ms: Option<u64>,
    pub reused_artifact_count: u32,
    pub recomputed_step_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VerificationRun {
    pub id: Uuid,
    pub project_id: Uuid,
    pub model_id: Uuid,
    pub property_id: Uuid,
    pub verifier_profile_id: Uuid,
    pub mode: VerificationMode,
    pub status: RunStatus,
    pub outcome: Option<RunOutcome>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub metrics: RunMetrics,
    pub reuse_plan_id: Option<Uuid>,
}

impl VerificationRun {
    pub fn new(
        project_id: Uuid,
        model_id: Uuid,
        property_id: Uuid,
        verifier_profile_id: Uuid,
        mode: VerificationMode,
        reuse_plan_id: Option<Uuid>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            project_id,
            model_id,
            property_id,
            verifier_profile_id,
            mode,
            status: RunStatus::Pending,
            outcome: None,
            started_at: None,
            ended_at: None,
            metrics: RunMetrics::default(),
            reuse_plan_id,
        }
    }
}
