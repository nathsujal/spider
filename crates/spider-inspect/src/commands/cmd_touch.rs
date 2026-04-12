//! `touch <id>` — increment access_count and update last_accessed_at on a node.

use anyhow::{anyhow, Result};

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: touch <id>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let db = &mut ctx.db;

    let idx = id - 1;
    let mut node = db.nodes.get(idx).map_err(|_| anyhow!("node #{} not found", id))?;
    if node.is_deleted() {
        output::print_error(&format!("node #{} is deleted", id));
        return Ok(Status::Continue);
    }

    node.access_count = node.access_count.saturating_add(1);
    node.last_accessed_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;

    db.nodes.set(idx, &node)?;

    output::print_ok(&format!("Touched node #{} — access_count: {}", id, node.access_count));
    Ok(Status::Continue)
}
