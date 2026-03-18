use std::{
    fs,
    path::{Path, PathBuf},
};

use async_trait::async_trait;

use ivm_application::{AppError, BlobStore};

#[derive(Debug, Clone)]
pub struct LocalBlobStore {
    root: PathBuf,
}

impl LocalBlobStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let sanitized = path.replace('\\', "/").trim_start_matches('/').to_string();
        self.root.join(Path::new(&sanitized))
    }
}

#[async_trait]
impl BlobStore for LocalBlobStore {
    async fn put_bytes(&self, path: &str, bytes: Vec<u8>) -> Result<String, AppError> {
        let resolved = self.resolve_path(path);

        if let Some(parent) = resolved.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| AppError::external_dependency(error.to_string()))?;
        }

        fs::write(&resolved, bytes)
            .map_err(|error| AppError::external_dependency(error.to_string()))?;

        Ok(resolved.display().to_string())
    }

    async fn get_bytes(&self, path: &str) -> Result<Vec<u8>, AppError> {
        let resolved = self.resolve_path(path);
        fs::read(&resolved).map_err(|error| AppError::external_dependency(error.to_string()))
    }
}
