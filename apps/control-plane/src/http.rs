use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use ivm_application::{
    AppError, CanonicalVerificationInput, CanonicalVerificationService,
    PersistedCanonicalVerificationRun,
};
use ivm_contracts::api::{
    CanonicalVerificationRequest, CanonicalVerificationResponse, ErrorResponse, HealthResponse,
};
use ivm_execution::SubprocessCanonicalVerificationExecutor;
use ivm_persistence::SqliteCanonicalVerificationRunRepository;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    canonical_service: Arc<CanonicalVerificationService>,
}

pub fn router() -> Router {
    let database_path = std::env::var("IVM_CANONICAL_RUN_DB")
        .unwrap_or_else(|_| "data/dev/verification-runs.sqlite3".to_string());
    let run_repository = Arc::new(
        SqliteCanonicalVerificationRunRepository::new(database_path)
            .expect("default canonical run store should initialize"),
    );
    let canonical_service = Arc::new(CanonicalVerificationService::new(
        Arc::new(SubprocessCanonicalVerificationExecutor::from_workspace()),
        run_repository,
    ));

    router_with_service(canonical_service)
}

pub fn router_with_service(canonical_service: Arc<CanonicalVerificationService>) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/verification-runs", post(create_verification_run))
        .route("/v1/verification-runs/{run_id}", get(fetch_verification_run))
        .route("/v1/demo/verification-jobs", post(run_canonical_verification))
        .with_state(AppState { canonical_service })
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok".to_string(), service: "ivm-control-plane".to_string() })
}

async fn create_verification_run() -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse::not_implemented(
            "verification-run submission is scaffolded but not wired yet",
        )),
    )
}

async fn run_canonical_verification(
    State(state): State<AppState>,
    request: Result<Json<CanonicalVerificationRequest>, JsonRejection>,
) -> Result<Json<CanonicalVerificationResponse>, ApiError> {
    let Json(request) = request.map_err(ApiError::from)?;
    let response = state
        .canonical_service
        .run(map_canonical_request(request))
        .await
        .map_err(ApiError::from)?;

    Ok(Json(response))
}

async fn fetch_verification_run(
    State(state): State<AppState>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<PersistedCanonicalVerificationRun>, ApiError> {
    let run = state.canonical_service.get_run(run_id).await.map_err(ApiError::from)?;

    Ok(Json(run))
}

fn map_canonical_request(request: CanonicalVerificationRequest) -> CanonicalVerificationInput {
    CanonicalVerificationInput {
        model_storage_uri: request.model_storage_uri,
        model_sha256: request.model_sha256,
        input_region: request.input_region,
        output_constraint: request.output_constraint,
        timeout_seconds: request.timeout_seconds,
        memory_mb: request.memory_mb,
    }
}

struct ApiError {
    status: StatusCode,
    body: ErrorResponse,
}

impl From<AppError> for ApiError {
    fn from(error: AppError) -> Self {
        match error {
            AppError::InputValidation(message) => Self {
                status: StatusCode::BAD_REQUEST,
                body: ErrorResponse { code: "input_validation_error".to_string(), message },
            },
            AppError::UnsupportedFeature(message) => Self {
                status: StatusCode::UNPROCESSABLE_ENTITY,
                body: ErrorResponse { code: "unsupported_feature_error".to_string(), message },
            },
            AppError::NotFound(message) => Self {
                status: StatusCode::NOT_FOUND,
                body: ErrorResponse { code: "not_found".to_string(), message },
            },
            AppError::Conflict(message) => Self {
                status: StatusCode::CONFLICT,
                body: ErrorResponse { code: "conflict".to_string(), message },
            },
            AppError::AdapterRuntime(message) => Self {
                status: StatusCode::BAD_GATEWAY,
                body: ErrorResponse { code: "adapter_runtime_error".to_string(), message },
            },
            AppError::ExternalDependency(message) => Self {
                status: StatusCode::BAD_GATEWAY,
                body: ErrorResponse { code: "external_dependency_error".to_string(), message },
            },
            AppError::InvariantViolation(message) => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body: ErrorResponse { code: "internal_invariant_violation".to_string(), message },
            },
            AppError::NotImplemented(message) => Self {
                status: StatusCode::NOT_IMPLEMENTED,
                body: ErrorResponse { code: "not_implemented".to_string(), message },
            },
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(error: JsonRejection) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorResponse {
                code: "input_validation_error".to_string(),
                message: format!("malformed request body: {error}"),
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}
