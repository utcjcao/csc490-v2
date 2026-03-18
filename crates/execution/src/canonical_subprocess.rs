use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use tokio::process::Command;
use uuid::Uuid;

use ivm_application::{AppError, CanonicalVerificationExecutor};
use ivm_contracts::canonical::{CanonicalVerificationJobRequest, CanonicalVerificationJobResult};
use ivm_domain::NormalizedExecutionSpec;

#[derive(Debug, Clone)]
pub struct SubprocessCanonicalVerificationExecutor {
    python_executable: OsString,
    repo_root: PathBuf,
}

impl Default for SubprocessCanonicalVerificationExecutor {
    fn default() -> Self {
        Self::from_workspace()
    }
}

impl SubprocessCanonicalVerificationExecutor {
    pub fn from_workspace() -> Self {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(Path::parent)
            .unwrap_or(manifest_dir.as_path())
            .to_path_buf();

        Self {
            python_executable: OsString::from(
                env::var("IVM_PYTHON").unwrap_or_else(|_| "python".to_string()),
            ),
            repo_root,
        }
    }

    fn create_run_paths(&self, run_id: Uuid) -> Result<(PathBuf, PathBuf, PathBuf), AppError> {
        let base_dir = env::temp_dir().join(format!("ivm-canonical-{run_id}"));
        fs::create_dir_all(&base_dir)
            .map_err(|error| AppError::adapter_runtime(error.to_string()))?;

        let input_path = base_dir.join("request.json");
        let output_path = base_dir.join("result.json");
        Ok((base_dir, input_path, output_path))
    }

    fn python_path(&self) -> Result<OsString, AppError> {
        let mut paths = Vec::new();

        paths.push(self.repo_root.join("python").join("adapter-common").join("src"));
        paths.push(self.repo_root.join("python").join("alpha-beta-crown-adapter").join("src"));

        if let Some(existing) = env::var_os("PYTHONPATH") {
            paths.extend(env::split_paths(&existing));
        }

        env::join_paths(paths).map_err(|error| AppError::adapter_runtime(error.to_string()))
    }
}

#[async_trait]
impl CanonicalVerificationExecutor for SubprocessCanonicalVerificationExecutor {
    fn backend_identifier(&self) -> &str {
        "alpha-beta-crown-subprocess"
    }

    async fn execute(
        &self,
        spec: NormalizedExecutionSpec,
    ) -> Result<CanonicalVerificationJobResult, AppError> {
        let request = lower_to_adapter_request(spec);
        let (work_dir, input_path, output_path) = self.create_run_paths(request.run_id)?;
        let request_body = serde_json::to_vec_pretty(&request)
            .map_err(|error| AppError::adapter_runtime(error.to_string()))?;

        fs::write(&input_path, request_body)
            .map_err(|error| AppError::adapter_runtime(error.to_string()))?;

        let output = Command::new(&self.python_executable)
            .arg("-m")
            .arg("alpha_beta_crown_adapter")
            .arg("--input")
            .arg(&input_path)
            .arg("--output")
            .arg(&output_path)
            .env("PYTHONPATH", self.python_path()?)
            .output()
            .await
            .map_err(|error| AppError::adapter_runtime(error.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let message = if stderr.is_empty() { stdout } else { stderr };
            let _ = fs::remove_dir_all(&work_dir);
            return Err(AppError::adapter_runtime(format!(
                "python adapter exited unsuccessfully: {}",
                message
            )));
        }

        let response_bytes =
            fs::read(&output_path).map_err(|error| AppError::adapter_runtime(error.to_string()))?;
        let result = serde_json::from_slice::<CanonicalVerificationJobResult>(&response_bytes)
            .map_err(|error| AppError::adapter_runtime(error.to_string()))?;

        let _ = fs::remove_dir_all(&work_dir);
        Ok(result)
    }
}

fn lower_to_adapter_request(spec: NormalizedExecutionSpec) -> CanonicalVerificationJobRequest {
    CanonicalVerificationJobRequest {
        run_id: spec.run.id,
        model_storage_uri: spec.model.storage_uri,
        model_sha256: spec.model.sha256,
        input_region: spec.property.input_region,
        output_constraint: spec.property.output_constraint,
        verifier_name: spec.verifier.name,
        timeout_seconds: spec.limits.timeout_seconds,
        memory_mb: spec.limits.memory_mb,
    }
}
