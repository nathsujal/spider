//! `propositions <doc_id>` — list all propositions connected to a document.

use anyhow::{anyhow, Result};
use colored::Colorize;
use spider_core::db::nodes::NodeId;
use spider_core::db::rels::Direction;
use spider_core::query::traverse::get_relationships;
use spider_core::property::get_string;
use spider_core::schema::token::TokenId;

use crate::commands::Status;
use crate::context::Context;
use crate::output::{self, table};

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: propositions <doc_id>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let node_id = NodeId::new(id)?;

    let node = ctx.db.nodes.get(id - 1).map_err(|_| anyhow!("node #{} not found", id))?;
    if node.is_deleted() {
        output::print_error(&format!("node #{} is deleted", id));
        return Ok(Status::Continue);
    }

    // Verify it's a document.
    let is_document = node.labels().iter().flatten().any(|lid| {
        TokenId::new(lid.get())
            .ok()
            .and_then(|tid| ctx.db.label_tokens.get_name(tid)) == Some("DOCUMENT")
    });
    if !is_document {
        output::print_error(&format!("node #{} is not a DOCUMENT", id));
        return Ok(Status::Continue);
    }

    // Get the title.
    let title = get_string(&mut ctx.db, node_id, "title").unwrap_or_default()
        .unwrap_or_else(|| "(untitled)".to_string());

    // Find CONTAINS edges (outgoing from document).
    let edges = get_relationships(&mut ctx.db, node_id, Direction::Outgoing)
        .unwrap_or_default();
    let contains_type_id = ctx.db.edge_type_tokens.get_id("CONTAINS").map(|t| t.get());

    let mut rows = Vec::new();
    let mut count = 0;

    for e in &edges {
        if let Some(cid) = contains_type_id {
            if e.edge_type().map(|t| t.get()) != Some(cid) {
                continue;
            }
        }

        let text = get_string(&mut ctx.db, NodeId::new(e.target_id)?, "text")
            .unwrap_or_default()
            .unwrap_or_else(|| "(no text)".to_string());

        // Truncate long text for display.
        let display = if text.len() > 80 {
            format!("{}...", &text[..77])
        } else {
            text
        };

        rows.push(vec![e.target_id.to_string(), display]);
        count += 1;
    }

    println!("\n{}", format!("Document #{}: \"{}\"", id, title).bold());
    println!("Propositions: {}", count);
    println!("{}", "─".repeat(70));

    if rows.is_empty() {
        println!("(none)");
    } else {
        println!("{}", table(&["Node", "Text"], rows));
    }

    println!();
    Ok(Status::Continue)
}
