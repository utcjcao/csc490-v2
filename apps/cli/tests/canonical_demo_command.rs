use std::{path::PathBuf, process::Command};
use uuid::Uuid;

#[test]
fn canonical_demo_command_runs_end_to_end() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("canonical_verification_request.valid.json");
    let database_path =
        std::env::temp_dir().join(format!("ivm-cli-test-{}.sqlite3", Uuid::new_v4()));

    let output = Command::new(env!("CARGO_BIN_EXE_ivm-cli"))
        .env("IVM_CANONICAL_RUN_DB", &database_path)
        .args([
            "verify",
            "demo",
            "--input",
            fixture_path.to_str().expect("fixture path should be valid UTF-8"),
        ])
        .output()
        .expect("cli command should run");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("cli should emit JSON");

    assert_eq!(payload["status"], "completed");
    assert_eq!(payload["outcome"], "proved");
    assert_eq!(payload["verifier_name"], "alpha-beta-crown");
}

#[test]
fn canonical_demo_status_fetches_persisted_run() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("canonical_verification_request.valid.json");
    let database_path =
        std::env::temp_dir().join(format!("ivm-cli-test-{}.sqlite3", Uuid::new_v4()));

    let submit_output = Command::new(env!("CARGO_BIN_EXE_ivm-cli"))
        .env("IVM_CANONICAL_RUN_DB", &database_path)
        .args([
            "verify",
            "demo",
            "--input",
            fixture_path.to_str().expect("fixture path should be valid UTF-8"),
        ])
        .output()
        .expect("cli submit command should run");

    assert!(
        submit_output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&submit_output.stdout),
        String::from_utf8_lossy(&submit_output.stderr)
    );

    let submit_payload: serde_json::Value =
        serde_json::from_slice(&submit_output.stdout).expect("cli should emit JSON");
    let run_id = submit_payload["run_id"].as_str().expect("run_id should be present").to_string();

    let status_output = Command::new(env!("CARGO_BIN_EXE_ivm-cli"))
        .env("IVM_CANONICAL_RUN_DB", &database_path)
        .args(["verify", "status", &run_id])
        .output()
        .expect("cli status command should run");

    assert!(
        status_output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&status_output.stdout),
        String::from_utf8_lossy(&status_output.stderr)
    );

    let status_payload: serde_json::Value =
        serde_json::from_slice(&status_output.stdout).expect("cli should emit JSON");

    assert_eq!(status_payload["run_id"], run_id);
    assert_eq!(status_payload["status"], "completed");
    assert_eq!(status_payload["backend_identifier"], "alpha-beta-crown-subprocess");
}
