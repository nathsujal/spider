//! `delete edge <id>` — soft-delete an edge by ID.

use anyhow::{anyhow, Result};
use spider_core::db::rels::EdgeId;

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: delete edge <id>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid edge ID: '{}'", args[0]))?;
    let edge_id = EdgeId::new(id)?;

    ctx.db.edge_ops().delete(edge_id)?;

    output::print_ok(&format!("Deleted edge #{}", id));
    Ok(Status::Continue)
}
