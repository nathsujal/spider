//! `trace <doc_id>` — replay ingestion trace for a document.

use anyhow::{anyhow, Result};
use colored::Colorize;
use spider_core::db::nodes::NodeId;
use spider_core::db::rels::Direction;
use spider_core::query::traverse::get_relationships;
use spider_core::property::{get_string, list_all};

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: trace <doc_id>");
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
        spider_core::schema::token::TokenId::new(lid.get())
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

    // Find propositions via CONTAINS edges.
    let edges = get_relationships(&mut ctx.db, node_id, Direction::Outgoing).unwrap_or_default();
    let contains_type_id = ctx.db.edge_type_tokens.get_id("CONTAINS").map(|t| t.get());

    let mut prop_details = Vec::new();
    let mut total_entities = 0usize;
    let mut unique_entity_names = std::collections::HashSet::new();

    for e in &edges {
        if let Some(cid) = contains_type_id {
            if e.edge_type().map(|t| t.get()) != Some(cid) {
                continue;
            }
        }

        let prop_id = NodeId::new(e.target_id)?;
        let text = get_string(&mut ctx.db, prop_id, "text")
            .unwrap_or_default()
            .unwrap_or_else(|| "(no text)".to_string());

        // Find MENTIONS edges from this proposition.
        let prop_edges = get_relationships(&mut ctx.db, prop_id, Direction::Outgoing).unwrap_or_default();
        let mentions_type_id = ctx.db.edge_type_tokens.get_id("MENTIONS").map(|t| t.get());

        let mut entities = Vec::new();
        for pe in &prop_edges {
            if let Some(mid) = mentions_type_id {
                if pe.edge_type().map(|t| t.get()) != Some(mid) {
                    continue;
                }
            }

            let ent_props = list_all(&mut ctx.db, NodeId::new(pe.target_id)?)
                .unwrap_or_default();
            let name = ent_props.iter().find(|p| p.key == "name")
                .map(|p| p.value.to_string())
                .unwrap_or_else(|| "?".to_string());
            let ent_type = ent_props.iter().find(|p| p.key == "entity_type")
                .map(|p| p.value.to_string())
                .unwrap_or_else(|| "?".to_string());

            unique_entity_names.insert(name.clone());
            total_entities += 1;
            entities.push((pe.target_id, name, ent_type));
        }

        prop_details.push((e.target_id, text, entities));
    }

    let total_edges = edges.iter().filter(|e| {
        contains_type_id.map_or(true, |cid| e.edge_type().map(|t| t.get()) == Some(cid))
    }).count() + total_entities;

    // Print summary.
    println!("\n{}", format!("Ingestion Trace for Document #{}: \"{}\"", id, title).bold());
    println!("{}", "─".repeat(60));
    println!("Document node:  #{}  [DOCUMENT]", id);
    println!("Propositions:   {}", prop_details.len());
    println!("Entities:       {}  ({} unique)", total_entities, unique_entity_names.len());
    println!("Edges:          {}  ({} CONTAINS + {} MENTIONS)",
        total_edges, prop_details.len(), total_entities);
    println!();

    // Print each proposition with its entities.
    for (prop_id, text, entities) in &prop_details {
        println!("Proposition #{}: \"{}\"", prop_id, text);
        for (ent_id, name, ent_type) in entities {
            println!("  \u{2192} MENTIONS \u{2192} Entity #{}: \"{}\" [{}]", ent_id, name, ent_type);
        }
        println!();
    }

    Ok(Status::Continue)
}
