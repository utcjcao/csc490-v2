use std::sync::Arc;

use ivm_application::CanonicalVerificationService;
use ivm_control_plane::{config::ControlPlaneConfig, router_with_service};
use ivm_execution::SubprocessCanonicalVerificationExecutor;
use ivm_observability::init_tracing;
use ivm_persistence::SqliteCanonicalVerificationRunRepository;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing("ivm-control-plane");

    let config = ControlPlaneConfig::from_env();
    let run_repository = Arc::new(SqliteCanonicalVerificationRunRepository::new(
        config.canonical_run_database_path.clone(),
    )?);
    let canonical_service = Arc::new(CanonicalVerificationService::new(
        Arc::new(SubprocessCanonicalVerificationExecutor::from_workspace()),
        run_repository,
    ));
    let listener = tokio::net::TcpListener::bind(config.bind_address).await?;

    tracing::info!(address = %config.bind_address, "starting control-plane scaffold");

    axum::serve(listener, router_with_service(canonical_service)).await?;
    Ok(())
}
