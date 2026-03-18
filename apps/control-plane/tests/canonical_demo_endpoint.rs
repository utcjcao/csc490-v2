use std::{fs, path::PathBuf, sync::Arc};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use ivm_application::CanonicalVerificationService;
use ivm_control_plane::router_with_service;
use ivm_execution::SubprocessCanonicalVerificationExecutor;
use ivm_persistence::SqliteCanonicalVerificationRunRepository;
use tower::util::ServiceExt;
use uuid::Uuid;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("fixtures").join(name)
}

fn load_fixture(name: &str) -> String {
    fs::read_to_string(fixture_path(name)).expect("fixture should load")
}

fn build_test_router() -> (Router, Arc<SqliteCanonicalVerificationRunRepository>) {
    let database_path = std::env::temp_dir().join(format!("ivm-test-{}.sqlite3", Uuid::new_v4()));
    let repository = Arc::new(
        SqliteCanonicalVerificationRunRepository::new(database_path)
            .expect("sqlite repository should initialize"),
    );
    let service = Arc::new(CanonicalVerificationService::new(
        Arc::new(SubprocessCanonicalVerificationExecutor::from_workspace()),
        repository.clone(),
    ));

    (router_with_service(service), repository)
}

#[tokio::test]
async fn canonical_demo_happy_path_returns_structured_success() {
    let (router, repository) = build_test_router();
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/demo/verification-jobs")
                .header("content-type", "application/json")
                .body(Body::from(load_fixture("canonical_verification_request.valid.json")))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);

    let payload: serde_json::Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX).await.expect("body should read"),
    )
    .expect("body should be valid JSON");

    assert_eq!(payload["status"], "completed");
    assert_eq!(payload["outcome"], "proved");
    assert_eq!(payload["verifier_name"], "alpha-beta-crown");

    let run_id = payload["run_id"].as_str().expect("run_id should be a string").to_string();
    assert_eq!(repository.count_runs().expect("count should succeed"), 1);

    let response = build_test_router_for_existing_repo(repository.clone())
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/verification-runs/{run_id}"))
                .body(Body::empty())
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);

    let payload: serde_json::Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX).await.expect("body should read"),
    )
    .expect("body should be valid JSON");

    assert_eq!(payload["status"], "completed");
    assert_eq!(payload["backend_identifier"], "alpha-beta-crown-subprocess");
    assert_eq!(payload["result_snapshot"]["outcome"], "proved");
}

#[tokio::test]
async fn canonical_demo_malformed_request_returns_bad_request() {
    let (router, _) = build_test_router();
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/demo/verification-jobs")
                .header("content-type", "application/json")
                .body(Body::from(load_fixture("canonical_verification_request.malformed.json")))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let payload: serde_json::Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX).await.expect("body should read"),
    )
    .expect("body should be valid JSON");

    assert_eq!(payload["code"], "input_validation_error");
    assert!(
        payload["message"]
            .as_str()
            .expect("message should be a string")
            .contains("malformed request body")
    );
}

#[tokio::test]
async fn canonical_demo_semantic_validation_failure_returns_bad_request() {
    let (router, repository) = build_test_router();
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/demo/verification-jobs")
                .header("content-type", "application/json")
                .body(Body::from(load_fixture(
                    "canonical_verification_request.semantic_invalid.json",
                )))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let payload: serde_json::Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX).await.expect("body should read"),
    )
    .expect("body should be valid JSON");

    assert_eq!(payload["code"], "input_validation_error");
    assert_eq!(
        payload["message"],
        "input_region.eps must be greater than 0 and less than or equal to 1"
    );
    assert_eq!(repository.count_runs().expect("count should succeed"), 0);
}

#[tokio::test]
async fn canonical_demo_unsupported_request_returns_unprocessable_entity() {
    let (router, _) = build_test_router();
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/demo/verification-jobs")
                .header("content-type", "application/json")
                .body(Body::from(load_fixture("canonical_verification_request.unsupported.json")))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);

    let payload: serde_json::Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX).await.expect("body should read"),
    )
    .expect("body should be valid JSON");

    assert_eq!(payload["code"], "unsupported_feature_error");
    assert_eq!(
        payload["message"],
        "input_region.norm 'l2' is not supported; only 'linf' is allowed in Phase 1"
    );
}

#[tokio::test]
async fn canonical_demo_adapter_error_returns_structured_error_result() {
    let request_body = serde_json::json!({
        "model_storage_uri": "demo://models/adapter-error.onnx",
        "model_sha256": "demo-model-sha256",
        "input_region": "{\"eps\":0.01}",
        "output_constraint": "{\"label\":1}",
        "timeout_seconds": 60,
        "memory_mb": 1024
    });

    let (router, repository) = build_test_router();
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/demo/verification-jobs")
                .header("content-type", "application/json")
                .body(Body::from(request_body.to_string()))
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);

    let payload: serde_json::Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX).await.expect("body should read"),
    )
    .expect("body should be valid JSON");

    assert_eq!(payload["status"], "failed");
    assert_eq!(payload["outcome"], "error");
    assert_eq!(payload["failure"]["code"], "stub_adapter_error");

    let run_id = payload["run_id"].as_str().expect("run_id should be a string").to_string();
    let response = build_test_router_for_existing_repo(repository.clone())
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/verification-runs/{run_id}"))
                .body(Body::empty())
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::OK);

    let payload: serde_json::Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX).await.expect("body should read"),
    )
    .expect("body should be valid JSON");

    assert_eq!(payload["status"], "failed");
    assert_eq!(payload["error_snapshot"]["code"], "stub_adapter_error");
    assert_eq!(payload["result_snapshot"]["status"], "failed");
}

#[tokio::test]
async fn fetching_unknown_run_returns_not_found() {
    let (router, _) = build_test_router();
    let response = router
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/verification-runs/{}", Uuid::new_v4()))
                .body(Body::empty())
                .expect("request should build"),
        )
        .await
        .expect("router should respond");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let payload: serde_json::Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX).await.expect("body should read"),
    )
    .expect("body should be valid JSON");

    assert_eq!(payload["code"], "not_found");
}

fn build_test_router_for_existing_repo(
    repository: Arc<SqliteCanonicalVerificationRunRepository>,
) -> Router {
    let service = Arc::new(CanonicalVerificationService::new(
        Arc::new(SubprocessCanonicalVerificationExecutor::from_workspace()),
        repository,
    ));

    router_with_service(service)
}
