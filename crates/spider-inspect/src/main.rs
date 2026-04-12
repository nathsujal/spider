//! Interactive TUI for inspecting Spider databases.
//!
//! Opens a database and provides a split-pane terminal UI
//! for examining nodes, edges, properties, and bio scores.

use anyhow::{Context as _, Result};
use clap::Parser;
use std::path::PathBuf;

mod commands;
mod context;
mod output;
mod output_globals;
mod sink;
mod tui;

/// Spider Inspect — Interactive TUI for debugging Spider databases.
#[derive(Parser, Debug)]
#[command(
    name = "spider-inspect",
    about = "Interactive TUI for debugging Spider databases",
    long_about = None,
)]
struct Args {
    /// Path to the Spider database directory.
    /// If omitted, uses the platform default location.
    db_path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Open database.
    let ctx = context::Context::open(args.db_path.as_deref())
        .with_context(|| "failed to open database")?;

    // Launch the TUI.
    tui::run_tui(ctx)
}
