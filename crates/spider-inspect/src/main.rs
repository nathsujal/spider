//! Interactive REPL for inspecting Spider databases.
//!
//! Opens a database (default or user-specified) and provides commands
//! for examining nodes, edges, properties, and bio scores.

use anyhow::{Context as _, Result};
use clap::Parser;
use std::path::PathBuf;

mod commands;
mod context;
mod output;
mod output_globals;
#[cfg(feature = "repl")]
mod repl;
mod sink;
#[cfg(feature = "tui")]
mod tui;

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

    /// Use the TUI instead of the line-based REPL.
    #[arg(long, short)]
    #[cfg(feature = "tui")]
    tui: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Open database.
    let db_path = args.db_path.clone();
    let ctx = context::Context::open(db_path.as_deref())
        .with_context(|| "failed to open database")?;

    #[cfg(feature = "tui")]
    {
        if args.tui {
            return tui::run_tui(ctx);
        }
    }

    #[cfg(feature = "repl")]
    {
        return repl::run(args.db_path);
    }

    #[allow(unreachable_code)]
    {
        eprintln!("No features enabled. Build with --features repl or --features tui");
        Ok(())
    }
}
