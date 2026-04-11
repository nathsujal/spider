//! Command definitions and dispatch.

use anyhow::Result;

use crate::context::Context;

mod bio;
mod broken;
mod cmd_set;
mod cmd_sig;
mod cmd_touch;
mod create_edge;
mod create_node;
mod delete_edge;
mod delete_node;
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
    CreateNode,
    CreateEdge,
    Set,
    DeleteNode,
    DeleteEdge,
    Touch,
    Sig,
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
            "create" => {
                if parts.len() > 1 {
                    match parts[1].to_lowercase().as_str() {
                        "node" => Command::CreateNode,
                        "edge" => Command::CreateEdge,
                        _ => return None,
                    }
                } else {
                    return None;
                }
            }
            "set" => Command::Set,
            "delete" => {
                if parts.len() > 1 {
                    match parts[1].to_lowercase().as_str() {
                        "node" => Command::DeleteNode,
                        "edge" => Command::DeleteEdge,
                        _ => return None,
                    }
                } else {
                    return None;
                }
            }
            "touch" => Command::Touch,
            "sig" => Command::Sig,
            _ => return None,
        };

        // For two-word commands (create node, delete node, etc.), skip the subcommand.
        let skip = match cmd {
            Command::CreateNode | Command::CreateEdge
            | Command::DeleteNode | Command::DeleteEdge => 2,
            _ => 1,
        };
        Some((cmd, parts[skip..].to_vec()))
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
            Command::CreateNode => create_node::run(ctx, args),
            Command::CreateEdge => create_edge::run(ctx, args),
            Command::Set => cmd_set::run(ctx, args),
            Command::DeleteNode => delete_node::run(ctx, args),
            Command::DeleteEdge => delete_edge::run(ctx, args),
            Command::Touch => cmd_touch::run(ctx, args),
            Command::Sig => cmd_sig::run(ctx, args),
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
  create node <L> [k=v]  — Create a new node with label and optional properties\n\
  create edge <s> <T> <d> — Create a directed edge between two nodes\n\
  set <id> <k> <v>       — Set a string property on a node\n\
  delete node <id>       — Soft-delete a node (warns about live edges)\n\
  delete edge <id>       — Soft-delete an edge\n\
  touch <id>             — Increment access count (affects bio score)\n\
  sig <id> <0-255>       — Set node significance, prints new bio score\n\
\n\
Note: changes are persisted when you exit the REPL.\n\
"
    );
}
