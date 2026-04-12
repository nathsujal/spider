//! `hex <id>` — hex dump of a node record with field annotations.

use anyhow::{anyhow, Result};
use colored::Colorize;

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: hex <id>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let db = &mut ctx.db;

    // Read the raw bytes from nodes.db.
    let idx = id - 1;
    let node = match db.nodes.get(idx) {
        Ok(n) => n,
        Err(_) => {
            output::print_error(&format!("node #{} not found", id));
            return Ok(Status::Continue);
        }
    };

    // Get raw bytes.
    let raw = node.to_bytes();

    println!("\n{}", format!("Node #{} Hex Dump ({} bytes)", id, raw.len()).bold());
    println!("{}", "─".repeat(60));

    // Print hex dump with field annotations.
    let fields: Vec<(&str, usize, usize)> = vec![
        ("id", 0, 4),
        ("first_edge_id", 4, 4),
        ("first_prop_id", 8, 4),
        ("labels[0..3]", 12, 4),
        ("access_count", 16, 4),
        ("created_at", 20, 4),
        ("last_accessed_at", 24, 4),
        ("significance", 28, 1),
    ];

    let bytes_per_row = 8;
    for offset in (0..raw.len()).step_by(bytes_per_row) {
        let end = (offset + bytes_per_row).min(raw.len());

        // Hex column.
        let hex: String = raw[offset..end].iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(" ");

        // Find annotation for this row.
        let annotation: String = fields.iter()
            .filter(|(_, start, _len)| {
                *start >= offset && *start < end
            })
            .map(|(name, start, _len)| {
                let rel = start - offset;
                let pad = " ".repeat(rel * 3);
                format!("{}← {}", pad, name)
            })
            .collect::<Vec<_>>()
            .join("\n");

        println!("  {:04x}  {:<24}  {}", offset, hex, annotation);
    }

    // Print decoded values.
    println!("\nDecoded fields:");
    println!("  id                = {}", node.id);
    println!("  first_edge_id     = {}", node.first_edge_id);
    println!("  first_prop_id     = {}", node.first_prop_id);
    println!("  labels            = {:?}", node.labels());
    println!("  access_count      = {}", node.access_count);
    println!("  created_at        = {}", node.created_at);
    println!("  last_accessed_at  = {}", node.last_accessed_at);
    println!("  significance      = {}", node.significance);

    println!();
    Ok(Status::Continue)
}
