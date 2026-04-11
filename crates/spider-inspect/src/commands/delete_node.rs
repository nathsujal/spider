//! `delete node <id>` — soft-delete a node (writes tombstone).
use colored::Colorize;


use anyhow::{anyhow, Result};
use spider_core::db::nodes::NodeId;
use spider_core::db::rels::Direction;
use spider_core::query::traverse::get_relationships;

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: delete node <id>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let node_id = NodeId::new(id)?;
    let db = &mut ctx.db;

    // Check node exists.
    let node = db.nodes.get(id - 1).map_err(|_| anyhow!("node #{} not found", id))?;
    if node.is_deleted() {
        output::print_error(&format!("node #{} is already deleted", id));
        return Ok(Status::Continue);
    }

    // Warn about live edges.
    let edge_count = get_relationships(db, node_id, Direction::Both)
        .map(|e| e.len())
        .unwrap_or(0);
    if edge_count > 0 {
        eprintln!("{} node #{} has {} live edge(s) — they will become dangling",
            "warning:".yellow().bold(), id, edge_count);
    }

    // Write tombstone.
    let tombstone = spider_core::schema::node::Node::default();
    db.nodes.set(id - 1, &tombstone)?;

    output::print_ok(&format!("Deleted node #{}", id));
    Ok(Status::Continue)
}
