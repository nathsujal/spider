//! `set <node_id> <key> <value>` — set a string property on an existing node.
//!
//! Value is the rest of the line after the key (spaces allowed).

use anyhow::{anyhow, Result};

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.len() < 2 {
        output::print_error("usage: set <node_id> <key> <value>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let key = args[1];
    // Value is everything after the key — rejoin args, strip quotes.
    let raw = args[2..].join(" ");
    let value = strip_quotes(&raw);

    output::set_string_prop(&mut ctx.db, id, key, value)?;
    output::print_ok(&format!("Set property '{}' on node #{}: \"{}\"", key, id, value));
    Ok(Status::Continue)
}

/// Strip leading/trailing double quotes if present.
fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        &s[1..s.len() - 1]
    } else {
        s
    }
}
