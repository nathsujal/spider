//! Command definitions and dispatch.

use anyhow::Result;

use crate::context::Context;

mod stats;

/// Available REPL commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    Help,
    Quit,
    Stats,
}

impl Command {
    /// Parse a command from user input.
    pub fn parse(input: &str) -> Option<Self> {
        match input.trim().to_lowercase().as_str() {
            "help" | "?" => Some(Command::Help),
            "quit" | "exit" | "q" => Some(Command::Quit),
            "stats" => Some(Command::Stats),
            _ => None,
        }
    }

    /// Execute the command.
    pub fn execute(&self, ctx: &mut Context) -> Result<Status> {
        match self {
            Command::Help => {
                print_help();
                Ok(Status::Continue)
            }
            Command::Quit => Ok(Status::Quit),
            Command::Stats => stats::run(ctx),
        }
    }
}

/// Command execution result.
#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    Continue,
    Quit,
}

fn print_help() {
    println!(
        "\n\
Available commands:\n\
  help, ?     — Show this help\n\
  quit, q     — Exit the REPL\n\
  stats       — Show database overview\n\
"
    );
}
