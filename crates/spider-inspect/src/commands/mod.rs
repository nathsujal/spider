//! Command definitions and dispatch.

use anyhow::Result;

use crate::context::Context;

mod bio;
mod broken;
mod export_cmd;
mod propositions;
mod show;
mod stats;
mod trace;
mod why_dead;

/// Available REPL commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    Help,
    Quit,
    Stats,
    Show,
    Bio,
    WhyDead,
    Propositions,
    Trace,
    Broken,
    Export,
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
            "why-dead" => Command::WhyDead,
            "propositions" | "props" => Command::Propositions,
            "trace" => Command::Trace,
            "broken" => Command::Broken,
            "export" => Command::Export,
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
            Command::WhyDead => why_dead::run(ctx, args),
            Command::Propositions => propositions::run(ctx, args),
            Command::Trace => trace::run(ctx, args),
            Command::Broken => broken::run(ctx),
            Command::Export => export_cmd::run(ctx, args),
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
  help, ?                — Show this help\n\
  quit, q                — Exit the REPL\n\
  stats                  — Show database overview\n\
  show <id>              — Show full node detail\n\
  bio                    — Vitality leaderboard\n\
  why-dead <id>          — Explain why a node has a low bio score\n\
  props <doc_id>         — List propositions for a document\n\
  trace <doc_id>         — Replay ingestion trace\n\
  broken                 — Run integrity check\n\
  export trace <id> <f>  — Export trace as JSON\n\
"
    );
}
