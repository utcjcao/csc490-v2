#[derive(Debug, Clone)]
pub struct PostgresConfig {
    pub database_url: String,
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self { database_url: "postgres://ivm:ivm@localhost:5432/ivm".to_string() }
    }
}
