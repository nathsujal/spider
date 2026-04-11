//! Interactive REPL for inspecting Spider databases.
//!
//! Opens a database (default or user-specified) and provides commands
//! for examining nodes, edges, properties, and bio scores.

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

mod commands;
mod context;
mod output;
mod repl;

/// Spider Inspect — Interactive REPL for debugging Spider databases.
#[derive(Parser, Debug)]
#[command(
    name = "spider-inspect",
    about = "Interactive REPL for debugging Spider databases",
    long_about = None,
)]
struct Args {
    /// Path to the Spider database directory.
    /// If omitted, uses the platform default location.
    db_path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    repl::run(args.db_path)
}
