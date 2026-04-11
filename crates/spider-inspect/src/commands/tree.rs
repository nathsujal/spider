//! `tree <doc_id>` — prints a tree view of a document and all connected nodes.

use anyhow::{anyhow, Result};
use colored::Colorize;
use spider_core::db::nodes::NodeId;
use spider_core::db::rels::Direction;
use spider_core::query::traverse::get_relationships;
use spider_core::property::get_string;
use spider_core::schema::token::TokenId;

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: tree <doc_id>");
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

    let title = get_string(&mut ctx.db, node_id, "title").unwrap_or_default()
        .unwrap_or_else(|| "(untitled)".to_string());

    println!();
    println!("{} \"{}\"", format!("Document #{}", id).bold(), title);

    let contains_tid = ctx.db.edge_type_tokens.get_id("CONTAINS").map(|t| t.get());
    let mentions_tid = ctx.db.edge_type_tokens.get_id("MENTIONS").map(|t| t.get());

    let edges = get_relationships(&mut ctx.db, node_id, Direction::Outgoing).unwrap_or_default();
    let prop_edges: Vec<_> = edges.iter().filter(|e| {
        contains_tid.map_or(true, |cid| e.edge_type().map(|t| t.get()) == Some(cid))
    }).collect();

    for (i, edge) in prop_edges.iter().enumerate() {
        let is_last = i == prop_edges.len() - 1;
        let prop_id = edge.target_id;

        let prop_text = get_string(&mut ctx.db, NodeId::new(prop_id)?, "text")
            .unwrap_or_default()
            .unwrap_or_else(|| "(no text)".to_string());

        let prefix = if is_last { "└──" } else { "├──" };
        let type_name = edge.edge_type()
            .and_then(|t| TokenId::new(t.get()).ok())
            .and_then(|tid| ctx.db.edge_type_tokens.get_name(tid).map(String::from))
            .unwrap_or_else(|| "?".to_string());

        println!("{} {} → Proposition #{} \"{}\"", prefix, type_name, prop_id, truncate(&prop_text, 60));

        let prop_edges_out = get_relationships(&mut ctx.db, NodeId::new(prop_id)?, Direction::Outgoing)
            .unwrap_or_default();
        let entity_edges: Vec<_> = prop_edges_out.iter().filter(|e| {
            mentions_tid.map_or(true, |mid| e.edge_type().map(|t| t.get()) == Some(mid))
        }).collect();

        let child_prefix = if is_last { "    " } else { "│   " };

        for (j, ent_edge) in entity_edges.iter().enumerate() {
            let ent_is_last = j == entity_edges.len() - 1;
            let ent_id = ent_edge.target_id;

            let ent_name = get_string(&mut ctx.db, NodeId::new(ent_id)?, "name")
                .unwrap_or_default()
                .unwrap_or_else(|| "?".to_string());
            let ent_type = get_string(&mut ctx.db, NodeId::new(ent_id)?, "entity_type")
                .unwrap_or_default()
                .unwrap_or_else(|| "?".to_string());

            let ent_prefix = if ent_is_last { "└──" } else { "├──" };
            let ent_type_name = ent_edge.edge_type()
                .and_then(|t| TokenId::new(t.get()).ok())
                .and_then(|tid| ctx.db.edge_type_tokens.get_name(tid).map(String::from))
                .unwrap_or_else(|| "?".to_string());

            println!("{} {} {} → Entity #{} \"{}\" [{}]",
                child_prefix, ent_prefix, ent_type_name, ent_id, ent_name, ent_type);
        }
    }

    println!();
    Ok(Status::Continue)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max.saturating_sub(3)])
    } else {
        s.to_string()
    }
}
