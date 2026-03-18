use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BenchmarkRow {
    pub baseline_run_id: Uuid,
    pub candidate_run_id: Uuid,
    pub baseline_wall_time_ms: Option<u64>,
    pub candidate_wall_time_ms: Option<u64>,
    pub reused_artifact_count: u32,
    pub outcome_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BenchmarkReport {
    pub project_id: Uuid,
    pub rows: Vec<BenchmarkRow>,
}
