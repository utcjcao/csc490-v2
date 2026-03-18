use async_trait::async_trait;

use ivm_application::{AppError, BlobStore};

#[derive(Debug, Clone, Default)]
pub struct S3BlobStore;

#[async_trait]
impl BlobStore for S3BlobStore {
    async fn put_bytes(&self, _path: &str, _bytes: Vec<u8>) -> Result<String, AppError> {
        Err(AppError::not_implemented(
            "S3BlobStore::put_bytes is pending the MinIO/S3 integration task",
        ))
    }

    async fn get_bytes(&self, _path: &str) -> Result<Vec<u8>, AppError> {
        Err(AppError::not_implemented(
            "S3BlobStore::get_bytes is pending the MinIO/S3 integration task",
        ))
    }
}
