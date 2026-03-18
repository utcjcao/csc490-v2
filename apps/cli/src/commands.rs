use std::{env, fs, path::PathBuf, sync::Arc};

use clap::{Args, Parser, Subcommand};
use ivm_application::{CanonicalVerificationInput, CanonicalVerificationService};
use ivm_contracts::api::CanonicalVerificationRequest;
use ivm_execution::SubprocessCanonicalVerificationExecutor;
use ivm_persistence::SqliteCanonicalVerificationRunRepository;
use serde_json::json;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(name = "ivm", about = "Incremental verification CLI scaffold")]
pub struct Cli {
    #[command(subcommand)]
    pub command: TopLevelCommand,
}

#[derive(Debug, Subcommand)]
pub enum TopLevelCommand {
    Project(ProjectCommand),
    Model(ModelCommand),
    Property(PropertyCommand),
    Verify(VerifyCommand),
    Reuse(ReuseCommand),
    Report(ReportCommand),
}

#[derive(Debug, Args)]
pub struct ProjectCommand {
    #[command(subcommand)]
    pub command: ProjectSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum ProjectSubcommand {
    Create {
        name: String,
        #[arg(long)]
        description: Option<String>,
    },
}

#[derive(Debug, Args)]
pub struct ModelCommand {
    #[command(subcommand)]
    pub command: ModelSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum ModelSubcommand {
    Register {
        #[arg(long)]
        project_id: String,
        #[arg(long)]
        path: String,
    },
}

#[derive(Debug, Args)]
pub struct PropertyCommand {
    #[command(subcommand)]
    pub command: PropertySubcommand,
}

#[derive(Debug, Subcommand)]
pub enum PropertySubcommand {
    Register {
        #[arg(long)]
        project_id: String,
        #[arg(long)]
        file: String,
    },
}

#[derive(Debug, Args)]
pub struct VerifyCommand {
    #[command(subcommand)]
    pub command: VerifySubcommand,
}

#[derive(Debug, Subcommand)]
pub enum VerifySubcommand {
    Run {
        #[arg(long)]
        project_id: String,
        #[arg(long)]
        model_id: String,
        #[arg(long)]
        property_id: String,
        #[arg(long)]
        verifier_profile_id: String,
    },
    Demo {
        #[arg(long)]
        input: String,
    },
    Status {
        run_id: String,
    },
}

#[derive(Debug, Args)]
pub struct ReuseCommand {
    #[command(subcommand)]
    pub command: ReuseSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum ReuseSubcommand {
    Preview {
        #[arg(long)]
        source_model_id: String,
        #[arg(long)]
        target_model_id: String,
    },
}

#[derive(Debug, Args)]
pub struct ReportCommand {
    #[command(subcommand)]
    pub command: ReportSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum ReportSubcommand {
    Export {
        #[arg(long)]
        project_id: String,
    },
}

pub async fn handle(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let response = match cli.command {
        TopLevelCommand::Project(ProjectCommand {
            command: ProjectSubcommand::Create { name, description },
        }) => json!({
            "status": "not_implemented",
            "action": "project.create",
            "name": name,
            "description": description,
        }),
        TopLevelCommand::Model(ModelCommand {
            command: ModelSubcommand::Register { project_id, path },
        }) => json!({
            "status": "not_implemented",
            "action": "model.register",
            "project_id": project_id,
            "path": path,
        }),
        TopLevelCommand::Property(PropertyCommand {
            command: PropertySubcommand::Register { project_id, file },
        }) => json!({
            "status": "not_implemented",
            "action": "property.register",
            "project_id": project_id,
            "file": file,
        }),
        TopLevelCommand::Verify(VerifyCommand {
            command:
                VerifySubcommand::Run { project_id, model_id, property_id, verifier_profile_id },
        }) => json!({
            "status": "not_implemented",
            "action": "verify.run",
            "project_id": project_id,
            "model_id": model_id,
            "property_id": property_id,
            "verifier_profile_id": verifier_profile_id,
        }),
        TopLevelCommand::Verify(VerifyCommand { command: VerifySubcommand::Demo { input } }) => {
            let request = load_canonical_request(&input)?;
            let service = build_canonical_service()?;
            let result = service.run(map_canonical_request(request)).await?;
            serde_json::to_value(result)?
        }
        TopLevelCommand::Verify(VerifyCommand { command: VerifySubcommand::Status { run_id } }) => {
            let service = build_canonical_service()?;
            let run = service.get_run(Uuid::parse_str(&run_id)?).await?;
            serde_json::to_value(run)?
        }
        TopLevelCommand::Reuse(ReuseCommand {
            command: ReuseSubcommand::Preview { source_model_id, target_model_id },
        }) => json!({
            "status": "not_implemented",
            "action": "reuse.preview",
            "source_model_id": source_model_id,
            "target_model_id": target_model_id,
        }),
        TopLevelCommand::Report(ReportCommand {
            command: ReportSubcommand::Export { project_id },
        }) => json!({
            "status": "not_implemented",
            "action": "report.export",
            "project_id": project_id,
        }),
    };

    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}

fn load_canonical_request(
    path: &str,
) -> Result<CanonicalVerificationRequest, Box<dyn std::error::Error>> {
    let payload = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&payload)?)
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

fn build_canonical_service() -> Result<CanonicalVerificationService, Box<dyn std::error::Error>> {
    let database_path = env::var("IVM_CANONICAL_RUN_DB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data/dev/verification-runs.sqlite3"));
    let run_repository = Arc::new(SqliteCanonicalVerificationRunRepository::new(database_path)?);

    Ok(CanonicalVerificationService::new(
        Arc::new(SubprocessCanonicalVerificationExecutor::from_workspace()),
        run_repository,
    ))
}
