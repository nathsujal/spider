mod routes;
mod server;

use anyhow::Result;
use clap::Parser;
use spider_core::db::lifecycle::Spider;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// CLI arguments parsed by the daemon.
#[derive(Debug, clap::Parser)]
#[command(name = "spider-daemon")]
#[command(about = "HTTP/WebSocket daemon for Spider memory graph")]
struct Cli {
    /// Database directory path (default: ~/.local/share/spider/default/)
    #[arg(short, long)]
    db_path: Option<PathBuf>,

    /// Bind address (default: 127.0.0.1:7777)
    #[arg(short, long, default_value = "127.0.0.1:7777")]
    bind: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new("spider_daemon=info,tower_http=info")
        }))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    // Open (or create) the Spider database
    let db_path = cli.db_path.unwrap_or_else(|| {
        spider_core::db::lifecycle::default_db_path()
    });

    tracing::info!(path = %db_path.display(), "opening spider database");
    let db = Spider::open(&db_path)?;

    // Start the HTTP server
    let addr = cli.bind.parse::<std::net::SocketAddr>()?;
    server::run(db, addr).await?;

    Ok(())
}
