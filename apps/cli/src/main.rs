mod commands;

use clap::Parser;
use commands::{Cli, handle};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    handle(cli).await
}
