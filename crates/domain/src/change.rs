use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChangeClass {
    WeightOnly,
    Structural,
    Export,
    Quantization,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChangeSet {
    pub id: Uuid,
    pub source_model_id: Uuid,
    pub target_model_id: Uuid,
    pub change_class: ChangeClass,
    pub layer_deltas: String,
    pub numeric_delta_summary: String,
    pub compatible_for_incremental: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecomputeStep {
    pub name: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReusePlan {
    pub id: Uuid,
    pub changeset_id: Uuid,
    pub baseline_run_id: Uuid,
    pub selected_artifact_ids: Vec<Uuid>,
    pub invalidated_artifact_ids: Vec<Uuid>,
    pub recompute_steps: Vec<RecomputeStep>,
    pub soundness_basis: String,
    pub planner_version: String,
    pub created_at: DateTime<Utc>,
}
