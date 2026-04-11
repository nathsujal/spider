//! `export trace <doc_id> <file>` — export ingestion trace as JSON.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, Result};
use serde::Serialize;
use spider_core::db::nodes::NodeId;
use spider_core::db::rels::Direction;
use spider_core::query::traverse::get_relationships;
use spider_core::property::{get_string, list_all};

use crate::commands::Status;
use crate::context::Context;
use crate::output;

#[derive(Serialize)]
struct ExportDoc {
    document_id: u32,
    title: String,
    propositions: Vec<ExportProp>,
    stats: ExportStats,
}

#[derive(Serialize)]
struct ExportProp {
    node_id: u32,
    text: String,
    entities: Vec<ExportEntity>,
}

#[derive(Serialize)]
struct ExportEntity {
    node_id: u32,
    name: String,
    #[serde(rename = "type")]
    entity_type: String,
}

#[derive(Serialize)]
struct ExportStats {
    total_propositions: usize,
    total_entities: usize,
    unique_entities: usize,
    total_edges: usize,
}

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.len() < 2 {
        output::print_error("usage: export trace <doc_id> <file>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let file_path = Path::new(args[1]);
    let node_id = NodeId::new(id)?;

    let node = ctx.db.nodes.get(id - 1).map_err(|_| anyhow!("node #{} not found", id))?;
    if node.is_deleted() {
        output::print_error(&format!("node #{} is deleted", id));
        return Ok(Status::Continue);
    }

    let title = get_string(&mut ctx.db, node_id, "title").unwrap_or_default()
        .unwrap_or_else(|| "(untitled)".to_string());

    // Gather propositions.
    let edges = get_relationships(&mut ctx.db, node_id, Direction::Outgoing).unwrap_or_default();
    let contains_type_id = ctx.db.edge_type_tokens.get_id("CONTAINS").map(|t| t.get());
    let mentions_type_id = ctx.db.edge_type_tokens.get_id("MENTIONS").map(|t| t.get());

    let mut propositions = Vec::new();
    let mut unique_entity_names = std::collections::HashSet::new();
    let mut total_entities = 0usize;

    for e in &edges {
        if let Some(cid) = contains_type_id {
            if e.edge_type().map(|t| t.get()) != Some(cid) {
                continue;
            }
        }

        let text = get_string(&mut ctx.db, NodeId::new(e.target_id)?, "text")
            .unwrap_or_default()
            .unwrap_or_else(|| "(no text)".to_string());

        let prop_edges = get_relationships(&mut ctx.db, NodeId::new(e.target_id)?, Direction::Outgoing)
            .unwrap_or_default();

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
            entities.push(ExportEntity { node_id: pe.target_id, name, entity_type: ent_type });
        }

        propositions.push(ExportProp { node_id: e.target_id, text, entities });
    }

    let total_edges = propositions.len() + total_entities;

    let prop_count = propositions.len();
    let doc = ExportDoc {
        document_id: id,
        title,
        stats: ExportStats {
            total_propositions: prop_count,
            total_entities,
            unique_entities: unique_entity_names.len(),
            total_edges,
        },
        propositions,
    };

    let json = serde_json::to_string_pretty(&doc)
        .map_err(|e| anyhow!("failed to serialize JSON: {}", e))?;

    fs::write(file_path, &json)
        .map_err(|e| anyhow!("failed to write {}: {}", file_path.display(), e))?;

    output::print_ok(&format!("Exported {} propositions to {}", prop_count, file_path.display()));
    Ok(Status::Continue)
}
