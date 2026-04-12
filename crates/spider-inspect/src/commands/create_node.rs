//! `create node <LABEL> [key=value ...]` — create a new node with label and optional properties.

use anyhow::{anyhow, Result};
use spider_core::schema::node::{LabelId, Node};

use crate::commands::Status;
use crate::context::Context;
use crate::output;
use crate::output_globals;

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

    // Parse key=value pairs from remaining args, handling quoted values.
    let props = parse_props(&args[1..])?;
    for (key, value) in props {
        output::set_string_prop(db, node_id, &key, &value)?;
    }

    output::print_ok(&format!("Created node #{} with label [{}]", node_id, label));
    output_globals::set_tree_view(format!("Created node #{}\nLabel: {}\nUse 'show {}' for details", node_id, label, node_id));
    output_globals::set_node_id(node_id);
    Ok(Status::Continue)
}

/// Parse key=value pairs from whitespace-split args.
/// Handles quoted values: `title="My Doc"` → `("title", "My Doc")`.
fn parse_props(args: &[&str]) -> Result<Vec<(String, String)>> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i];
        if let Some(eq_pos) = arg.find('=') {
            let key = arg[..eq_pos].to_string();
            let rest = &arg[eq_pos + 1..];

            if let Some(unquoted) = rest.strip_prefix('"') {
                // Quoted value — collect tokens until closing quote.
                let mut value = unquoted.to_string();
                while !value.ends_with('"') && i + 1 < args.len() {
                    i += 1;
                    value.push(' ');
                    value.push_str(args[i]);
                }
                // Strip trailing ".
                if value.ends_with('"') {
                    value.pop();
                }
                result.push((key, value));
            } else {
                // Unquoted value (no spaces).
                result.push((key, rest.to_string()));
            }
        }
        i += 1;
    }
    Ok(result)
}
