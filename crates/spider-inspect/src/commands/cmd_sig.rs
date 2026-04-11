//! `sig <id> <value>` — set the significance byte (0–255) on a node.

use anyhow::{anyhow, Result};
use spider_core::bio::score::calculate;

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.len() < 2 {
        output::print_error("usage: sig <id> <value>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let sig: u8 = args[1].parse().map_err(|_| anyhow!("invalid significance value: '{}'", args[1]))?;

    let db = &mut ctx.db;
    let idx = id - 1;
    let mut node = db.nodes.get(idx).map_err(|_| anyhow!("node #{} not found", id))?;
    if node.is_deleted() {
        output::print_error(&format!("node #{} is deleted", id));
        return Ok(Status::Continue);
    }

    node.significance = sig;
    db.nodes.set(idx, &node)?;

    let score = calculate(&node);
    output::print_ok(&format!(
        "Set significance of node #{} to {} — bio score: {:.2}",
        id, sig, score
    ));
    Ok(Status::Continue)
}
