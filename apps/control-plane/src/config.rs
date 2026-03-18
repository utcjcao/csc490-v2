use std::{
    env,
    net::{Ipv4Addr, SocketAddr},
    path::PathBuf,
};

#[derive(Debug, Clone)]
pub struct ControlPlaneConfig {
    pub bind_address: SocketAddr,
    pub canonical_run_database_path: PathBuf,
}

impl ControlPlaneConfig {
    pub fn from_env() -> Self {
        let host =
            env::var("IVM_CONTROL_PLANE_HOST").unwrap_or_else(|_| Ipv4Addr::LOCALHOST.to_string());
        let port = env::var("IVM_CONTROL_PLANE_PORT")
            .ok()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(3000);

        let bind_address = format!("{host}:{port}")
            .parse()
            .unwrap_or_else(|_| SocketAddr::from((Ipv4Addr::LOCALHOST, port)));
        let canonical_run_database_path = env::var("IVM_CANONICAL_RUN_DB")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data/dev/verification-runs.sqlite3"));

        Self { bind_address, canonical_run_database_path }
    }
}
