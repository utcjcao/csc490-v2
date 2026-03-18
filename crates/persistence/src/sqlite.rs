use std::{fs, path::PathBuf};

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};
use uuid::Uuid;

use ivm_application::{
    AppError, CanonicalVerificationRunRepository, PersistedCanonicalVerificationRun,
    PersistedRunStatus,
};

#[derive(Debug, Clone)]
pub struct SqliteCanonicalVerificationRunRepository {
    database_path: PathBuf,
}

impl SqliteCanonicalVerificationRunRepository {
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, AppError> {
        let database_path = path.into();

        if let Some(parent) = database_path.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .map_err(|error| AppError::external_dependency(error.to_string()))?;
        }

        let repository = Self { database_path };
        repository.init_schema()?;
        Ok(repository)
    }

    pub fn count_runs(&self) -> Result<usize, AppError> {
        let connection = self.open_connection()?;
        let count = connection
            .query_row("SELECT COUNT(*) FROM canonical_verification_runs", [], |row| {
                row.get::<_, i64>(0)
            })
            .map_err(|error| AppError::external_dependency(error.to_string()))?;

        usize::try_from(count).map_err(|error| AppError::invariant_violation(error.to_string()))
    }

    fn init_schema(&self) -> Result<(), AppError> {
        let connection = self.open_connection()?;
        connection
            .execute_batch(
                "
                CREATE TABLE IF NOT EXISTS canonical_verification_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    normalized_execution_spec_json TEXT NOT NULL,
                    result_snapshot_json TEXT,
                    error_snapshot_json TEXT,
                    backend_identifier TEXT NOT NULL
                );
                ",
            )
            .map_err(|error| AppError::external_dependency(error.to_string()))?;
        Ok(())
    }

    fn open_connection(&self) -> Result<Connection, AppError> {
        Connection::open(&self.database_path)
            .map_err(|error| AppError::external_dependency(error.to_string()))
    }

    fn load_run(
        &self,
        connection: &Connection,
        run_id: Uuid,
    ) -> Result<Option<PersistedCanonicalVerificationRun>, AppError> {
        let row = connection
            .query_row(
                "
                SELECT
                    run_id,
                    status,
                    submitted_at,
                    updated_at,
                    normalized_execution_spec_json,
                    result_snapshot_json,
                    error_snapshot_json,
                    backend_identifier
                FROM canonical_verification_runs
                WHERE run_id = ?1
                ",
                [run_id.to_string()],
                hydrate_run,
            )
            .optional()
            .map_err(|error| AppError::external_dependency(error.to_string()))?;

        Ok(row)
    }
}

#[async_trait]
impl CanonicalVerificationRunRepository for SqliteCanonicalVerificationRunRepository {
    async fn create(
        &self,
        run: PersistedCanonicalVerificationRun,
    ) -> Result<PersistedCanonicalVerificationRun, AppError> {
        let connection = self.open_connection()?;
        let normalized_execution_spec_json =
            serde_json::to_string(&run.normalized_execution_spec_snapshot)
                .map_err(|error| AppError::invariant_violation(error.to_string()))?;
        let result_snapshot_json = serde_json::to_string(&run.result_snapshot)
            .map_err(|error| AppError::invariant_violation(error.to_string()))?;
        let error_snapshot_json = serde_json::to_string(&run.error_snapshot)
            .map_err(|error| AppError::invariant_violation(error.to_string()))?;

        connection
            .execute(
                "
                INSERT INTO canonical_verification_runs (
                    run_id,
                    status,
                    submitted_at,
                    updated_at,
                    normalized_execution_spec_json,
                    result_snapshot_json,
                    error_snapshot_json,
                    backend_identifier
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                ",
                params![
                    run.run_id.to_string(),
                    run.status.as_str(),
                    run.submitted_at.to_rfc3339(),
                    run.updated_at.to_rfc3339(),
                    normalized_execution_spec_json,
                    result_snapshot_json,
                    error_snapshot_json,
                    run.backend_identifier,
                ],
            )
            .map_err(|error| AppError::external_dependency(error.to_string()))?;

        Ok(run)
    }

    async fn get(
        &self,
        run_id: Uuid,
    ) -> Result<Option<PersistedCanonicalVerificationRun>, AppError> {
        let connection = self.open_connection()?;
        self.load_run(&connection, run_id)
    }

    async fn update(
        &self,
        run: PersistedCanonicalVerificationRun,
    ) -> Result<PersistedCanonicalVerificationRun, AppError> {
        let connection = self.open_connection()?;
        let normalized_execution_spec_json =
            serde_json::to_string(&run.normalized_execution_spec_snapshot)
                .map_err(|error| AppError::invariant_violation(error.to_string()))?;
        let result_snapshot_json = serde_json::to_string(&run.result_snapshot)
            .map_err(|error| AppError::invariant_violation(error.to_string()))?;
        let error_snapshot_json = serde_json::to_string(&run.error_snapshot)
            .map_err(|error| AppError::invariant_violation(error.to_string()))?;

        connection
            .execute(
                "
                UPDATE canonical_verification_runs
                SET
                    status = ?2,
                    submitted_at = ?3,
                    updated_at = ?4,
                    normalized_execution_spec_json = ?5,
                    result_snapshot_json = ?6,
                    error_snapshot_json = ?7,
                    backend_identifier = ?8
                WHERE run_id = ?1
                ",
                params![
                    run.run_id.to_string(),
                    run.status.as_str(),
                    run.submitted_at.to_rfc3339(),
                    run.updated_at.to_rfc3339(),
                    normalized_execution_spec_json,
                    result_snapshot_json,
                    error_snapshot_json,
                    run.backend_identifier,
                ],
            )
            .map_err(|error| AppError::external_dependency(error.to_string()))?;

        Ok(run)
    }
}

fn hydrate_run(row: &rusqlite::Row<'_>) -> rusqlite::Result<PersistedCanonicalVerificationRun> {
    let run_id = row.get::<_, String>(0)?;
    let status = row.get::<_, String>(1)?;
    let submitted_at = row.get::<_, String>(2)?;
    let updated_at = row.get::<_, String>(3)?;
    let normalized_execution_spec_json = row.get::<_, String>(4)?;
    let result_snapshot_json = row.get::<_, String>(5)?;
    let error_snapshot_json = row.get::<_, String>(6)?;
    let backend_identifier = row.get::<_, String>(7)?;

    Ok(PersistedCanonicalVerificationRun {
        run_id: Uuid::parse_str(&run_id).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?,
        status: parse_status(&status).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                1,
                rusqlite::types::Type::Text,
                Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, error)),
            )
        })?,
        submitted_at: chrono::DateTime::parse_from_rfc3339(&submitted_at)
            .map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    2,
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?
            .with_timezone(&chrono::Utc),
        updated_at: chrono::DateTime::parse_from_rfc3339(&updated_at)
            .map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    3,
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?
            .with_timezone(&chrono::Utc),
        normalized_execution_spec_snapshot: serde_json::from_str(&normalized_execution_spec_json)
            .map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                4,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?,
        result_snapshot: serde_json::from_str(&result_snapshot_json).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                5,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?,
        error_snapshot: serde_json::from_str(&error_snapshot_json).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                6,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?,
        backend_identifier,
    })
}

fn parse_status(value: &str) -> Result<PersistedRunStatus, String> {
    match value {
        "pending" => Ok(PersistedRunStatus::Pending),
        "running" => Ok(PersistedRunStatus::Running),
        "completed" => Ok(PersistedRunStatus::Completed),
        "failed" => Ok(PersistedRunStatus::Failed),
        other => Err(format!("unknown persisted run status '{other}'")),
    }
}
