//! Interactive REPL loop using rustyline.

use std::path::PathBuf;

use anyhow::Result;
use colored::Colorize;
use rustyline::{DefaultEditor, Config, EditMode};

use crate::commands::{Command, Status};
use crate::context::Context;
use crate::output;

/// Runs the REPL. Opens the database, enters the read-eval-print loop,
/// and closes the database on exit.
pub fn run(db_path: Option<PathBuf>) -> Result<()> {
    let mut ctx = Context::open(db_path.as_deref())?;

    let welcome = format!(
        "Spider Inspect — {}",
        if let Some(p) = &db_path {
            p.display().to_string()
        } else {
            "default database".to_string()
        }
    );
    println!("{}", welcome.bold());
    println!("{}", "Type 'help' for commands.\n".dimmed());

    let config = Config::builder()
        .edit_mode(EditMode::Emacs)
        .max_history_size(1000)?
        .build();
    let mut rl = DefaultEditor::with_config(config)?;

    let prompt = format!("spider ({})> ", ctx.db_label());

    loop {
        let line = match rl.readline(&prompt) {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("\nbye");
                break;
            }
            Err(e) => {
                output::print_error(&format!("readline error: {e}"));
                continue;
            }
        };

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        // Add to history.
        let _ = rl.add_history_entry(&line);

        // Dispatch.
        match Command::parse(input) {
            Some(cmd) => match cmd.execute(&mut ctx) {
                Ok(Status::Continue) => {}
                Ok(Status::Quit) => {
                    println!("{}", "bye".dimmed());
                    break;
                }
                Err(e) => {
                    output::print_error(&format!("{e:#}"));
                }
            },
            None => {
                output::print_error(&format!("unknown command: '{input}'. Type 'help' for commands."));
            }
        }
    }

    // Spider drops and flushes automatically, but be explicit for clarity.
    ctx.db.close().map_err(|e| anyhow::anyhow!("failed to close database: {e}"))?;

    Ok(())
}
