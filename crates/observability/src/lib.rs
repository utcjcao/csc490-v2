use std::sync::Once;

use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

pub fn init_tracing(service_name: &str) {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(format!("{service_name}=info")));

        tracing_subscriber::fmt().with_env_filter(filter).with_target(true).compact().init();
    });
}
