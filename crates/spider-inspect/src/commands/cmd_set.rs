//! `set <node_id> <key> <value>` — set a string property on an existing node.

use anyhow::{anyhow, Result};

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.len() < 3 {
        output::print_error("usage: set <node_id> <key> <value>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let key = args[1];
    // Value is everything after the key (allow spaces).
    let value = args[2..].join(" ");

    output::set_string_prop(&mut ctx.db, id, key, &value)?;
    output::print_ok(&format!("Set property '{}' on node #{}: \"{}\"", key, id, value));
    Ok(Status::Continue)
}
