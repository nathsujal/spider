//! Command definitions and dispatch.

use anyhow::Result;

use crate::context::Context;

mod bio;
mod show;
mod stats;

/// Available REPL commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    Help,
    Quit,
    Stats,
    Show,
    Bio,
}

impl Command {
    /// Parse a command from user input.
    /// Returns `(Command, args)` where args are space-separated tokens after the command.
    pub fn parse(input: &str) -> Option<(Self, Vec<&str>)> {
        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }
        let cmd = match parts[0].to_lowercase().as_str() {
            "help" | "?" => Command::Help,
            "quit" | "exit" | "q" => Command::Quit,
            "stats" => Command::Stats,
            "show" => Command::Show,
            "bio" => Command::Bio,
            _ => return None,
        };
        Some((cmd, parts[1..].to_vec()))
    }

    /// Execute the command with the given arguments.
    pub fn execute(&self, ctx: &mut Context, args: &[&str]) -> Result<Status> {
        match self {
            Command::Help => {
                print_help();
                Ok(Status::Continue)
            }
            Command::Quit => Ok(Status::Quit),
            Command::Stats => stats::run(ctx),
            Command::Show => show::run(ctx, args),
            Command::Bio => bio::run(ctx),
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
  help, ?       — Show this help\n\
  quit, q       — Exit the REPL\n\
  stats         — Show database overview\n\
  show <id>     — Show full node detail\n\
  bio           — Vitality leaderboard\n\
"
    );
}
