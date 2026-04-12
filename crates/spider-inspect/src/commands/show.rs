//! `show <node_id>` — full node detail with properties, edges, bio score.

use anyhow::{anyhow, Result};
use colored::Colorize;
use spider_core::bio::score::calculate;
use spider_core::db::nodes::NodeId;
use spider_core::db::rels::Direction;
use spider_core::query::traverse::get_relationships;
use spider_core::property::list_all;

use crate::commands::Status;
use crate::context::Context;
use crate::output::{self, table};
use crate::output_globals;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: show <node_id>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let node_id = NodeId::new(id)?;

    let node = ctx.db.nodes.get(id - 1).map_err(|_| anyhow!("node #{} not found", id))?;
    if node.is_deleted() {
        output::print_error(&format!("node #{} is deleted", id));
        return Ok(Status::Continue);
    }

    // Labels.
    let labels: Vec<String> = node.labels()
        .into_iter()
        .flatten()
        .map(|lid| output::resolve_label(&mut ctx.db.label_tokens, lid.get()))
        .collect();
    let labels_str = if labels.is_empty() { "(none)".to_string() } else { labels.join(", ") };

    // Bio score.
    let score = calculate(&node);

    // Properties.
    let props = list_all(&mut ctx.db, node_id).unwrap_or_default();

    // Edges.
    let edges = get_relationships(&mut ctx.db, node_id, Direction::Both).unwrap_or_default();
    let outgoing: Vec<_> = edges.iter().filter(|e| e.source_id == id).collect();
    let incoming: Vec<_> = edges.iter().filter(|e| e.target_id == id).collect();

    // Print header.
    println!("\n{}", format!("Node #{}", id).bold());
    println!("{}", "─".repeat(50));

    // Summary.
    let sig = output::format_significance(node.significance);
    let summary_rows = vec![
        vec!["Labels".to_string(), labels_str.clone()],
        vec!["Created".to_string(), output::format_timestamp(node.created_at)],
        vec!["Last accessed".to_string(), output::format_timestamp(node.last_accessed_at)],
        vec!["Access count".to_string(), node.access_count.to_string()],
        vec!["Significance".to_string(), sig],
        vec!["".to_string(), "".to_string()],
        vec!["Bio Score".to_string(), output::format_bio(score)],
    ];
    println!("{}", table(&["", ""], summary_rows));

    // Properties table.
    if !props.is_empty() {
        println!("\nProperties:");
        let prop_rows: Vec<_> = props.iter()
            .map(|p| vec![p.key.clone(), p.value.to_string()])
            .collect();
        println!("{}", table(&["Key", "Value"], prop_rows));
    }

    // Edges table.
    if !outgoing.is_empty() || !incoming.is_empty() {
        println!("\nEdges ({} outgoing, {} incoming):", outgoing.len(), incoming.len());
        let mut edge_rows = Vec::new();

        for e in &outgoing {
            let type_name = e.edge_type()
                .map(|tid| output::resolve_edge_type(&mut ctx.db.edge_type_tokens, tid.get()))
                .unwrap_or_else(|| "?".to_string());
            let target_labels = resolve_node_labels(&mut ctx.db, e.target_id);
            edge_rows.push(vec![
                e.id.to_string(),
                type_name,
                format!("#{}", e.target_id),
                target_labels,
            ]);
        }

        for e in &incoming {
            let type_name = e.edge_type()
                .map(|tid| output::resolve_edge_type(&mut ctx.db.edge_type_tokens, tid.get()))
                .unwrap_or_else(|| "?".to_string());
            let source_labels = resolve_node_labels(&mut ctx.db, e.source_id);
            edge_rows.push(vec![
                e.id.to_string(),
                type_name,
                format!("#{} (from)", e.source_id),
                source_labels,
            ]);
        }

        println!("{}", table(&["ID", "Type", "Node", "Labels"], edge_rows));
    }

    // Update TUI graph view.
    let mut tree_text = format!("Node #{}\n", id);
    tree_text += &format!("Labels: {}\n", &labels_str);
    tree_text += &format!("Score:  {:.2} [{}]\n", score, spider_core::bio::tier::BioTier::from_score(score));
    tree_text += &format!("Access: {}\n", node.access_count);
    for e in &outgoing {
        let type_name = if let Some(tid) = e.edge_type() {
            output::resolve_edge_type(&mut ctx.db.edge_type_tokens, tid.get())
        } else {
            "?".to_string()
        };
        tree_text += &format!("  → {} → #{}\n", type_name, e.target_id);
    }
    output_globals::set_tree_view(tree_text);
    output_globals::set_node_id(id);

    println!();
    Ok(Status::Continue)
}

fn resolve_node_labels(db: &mut spider_core::db::lifecycle::Spider, node_id: u32) -> String {
    match db.nodes.get(node_id - 1) {
        Ok(n) => n.labels()
            .into_iter()
            .flatten()
            .map(|lid| output::resolve_label(&mut db.label_tokens, lid.get()))
            .collect::<Vec<_>>()
            .join(", "),
        Err(_) => "(deleted)".to_string(),
    }
}
