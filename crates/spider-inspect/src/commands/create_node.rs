//! `create node <LABEL> [key=value ...]` — create a new node with label and optional properties.

use anyhow::{anyhow, Result};
use spider_core::schema::node::{LabelId, Node};

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: create node <LABEL> [key=value ...]");
        return Ok(Status::Continue);
    }

    let label = args[0];
    let db = &mut ctx.db;

    // Register label token.
    let label_tid = db.label_tokens.get_or_create(label)
        .map_err(|e| anyhow!("failed to register label '{}': {}", label, e))?;
    let label_id = LabelId::new(label_tid.get())
        .map_err(|_| anyhow!("label token ID out of range"))?;

    // Allocate node ID.
    let node_id = db.metadata.next_node_id;
    db.metadata.next_node_id += 1;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;

    let node = Node::new(node_id, &[label_id], now, None)
        .map_err(|e| anyhow!("failed to create node: {}", e))?;

    db.nodes.append(&[node])?;

    // Set optional properties.
    for arg in &args[1..] {
        if let Some(eq_pos) = arg.find('=') {
            let key = &arg[..eq_pos];
            let value = &arg[eq_pos + 1..];
            output::set_string_prop(db, node_id, key, value)?;
        } else {
            output::print_error(&format!("ignoring invalid arg (expected key=value): '{}'", arg));
        }
    }

    output::print_ok(&format!("Created node #{} with label [{}]", node_id, label));
    Ok(Status::Continue)
}
