//! `create edge <src_id> <TYPE> <dst_id>` — create a directed edge between two nodes.

use anyhow::{anyhow, Result};
use spider_core::db::nodes::NodeId;
use spider_core::schema::edge::EdgeTypeId;

use crate::commands::Status;
use crate::context::Context;
use crate::output;
use crate::output_globals;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.len() < 3 {
        output::print_error("usage: create edge <src_id> <TYPE> <dst_id>");
        return Ok(Status::Continue);
    }

    let src_id: u32 = args[0].parse().map_err(|_| anyhow!("invalid source node ID: '{}'", args[0]))?;
    let dst_id: u32 = args.last().unwrap().parse()
        .map_err(|_| anyhow!("invalid target node ID: '{}'", args.last().unwrap()))?;
    // Edge type is everything between src and dst (rejoin for spaces).
    let edge_type = args[1..args.len() - 1].join(" ");
    let edge_type = strip_quotes(&edge_type);

    let db = &mut ctx.db;

    // Register edge type token.
    let type_tid = db.edge_type_tokens.get_or_create(edge_type)
        .map_err(|e| anyhow!("failed to register edge type '{}': {}", edge_type, e))?;
    let type_id = EdgeTypeId::new(type_tid.get())
        .map_err(|_| anyhow!("edge type token ID out of range"))?;

    let edge_id = db.edge_ops().create(
        NodeId::new(src_id)?,
        NodeId::new(dst_id)?,
        type_id,
    )?;

    output::print_ok(&format!("Created edge #{} ({} → {} [{}])", edge_id.get(), src_id, dst_id, edge_type));
    output_globals::set_tree_view(format!("Created edge #{}\n{} → {} [{}]\nUse 'graph {}' to visualize", edge_id.get(), src_id, dst_id, edge_type, src_id));
    output_globals::set_node_id(edge_id.get());
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
