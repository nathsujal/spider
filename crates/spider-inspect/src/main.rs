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
mod output_globals;
#[cfg(feature = "repl")]
mod repl;
mod sink;

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

    #[cfg(feature = "repl")]
    {
        return repl::run(args.db_path);
    }

    #[cfg(not(feature = "repl"))]
    {
        eprintln!("REPL feature not enabled. Build with --features repl or --features tui");
        Ok(())
    }
}
