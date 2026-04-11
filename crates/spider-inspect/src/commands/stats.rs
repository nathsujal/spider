//! `stats` command — database overview.

use std::collections::HashMap;
use std::cmp::Reverse;

use anyhow::Result;
use colored::Colorize;
use spider_core::store::record::Record;

use crate::commands::Status;
use crate::context::Context;
use crate::output::{self, table};

pub fn run(ctx: &mut Context) -> Result<Status> {
    let db = &mut ctx.db;

    // Count live nodes, tally labels.
    let mut live_nodes = 0u32;
    let mut label_counts: HashMap<u8, u32> = HashMap::new();

    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        if let Ok(node) = db.nodes.get(idx) {
            if !node.is_deleted() {
                live_nodes += 1;
                for lid in node.labels().into_iter().flatten() {
                    *label_counts.entry(lid.get()).or_insert(0) += 1;
                }
            }
        }
    }

    // Count live edges.
    let mut live_edges = 0u32;
    for idx in 0..db.metadata.next_rel_id.saturating_sub(1) {
        if let Ok(edge) = db.edges.get(idx) {
            if !edge.is_deleted() {
                live_edges += 1;
            }
        }
    }

    // Count live properties.
    let mut live_props = 0u32;
    for idx in 0..db.metadata.next_prop_id.saturating_sub(1) {
        if let Ok(prop) = db.properties.get(idx) {
            if !prop.is_deleted() {
                live_props += 1;
            }
        }
    }

    // Count dynamic string records.
    let mut live_strings = 0u32;
    for idx in 0..db.metadata.next_string_id.saturating_sub(1) {
        if let Ok(rec) = db.strings.get(idx) {
            if rec.is_in_use() {
                live_strings += 1;
            }
        }
    }

    // Sort labels by count descending.
    let mut labels: Vec<_> = label_counts.into_iter().collect();
    labels.sort_by_key(|&(_, count)| Reverse(count));

    let m = &db.metadata;
    let prop_keys = db.prop_key_tokens.len();

    // Print header.
    let db_path = db.path().display();
    println!("\n{}", format!("Database: {db_path}").bold());
    println!("{}", "─".repeat(50));

    // Summary table.
    let summary_rows = vec![
        vec!["Nodes".to_string(), live_nodes.to_string()],
        vec!["Edges".to_string(), live_edges.to_string()],
        vec!["Properties".to_string(), live_props.to_string()],
        vec!["Dynamic strings".to_string(), live_strings.to_string()],
        vec!["Property keys".to_string(), prop_keys.to_string()],
        vec!["".to_string(), "".to_string()],
        vec!["Bio params".to_string(),
             format!("w_sig={}  w_freq={}  gravity={}", m.bio_w_sig, m.bio_w_freq, m.bio_gravity)],
    ];
    println!("{}", table(&["", ""], summary_rows));

    // Label breakdown.
    if !labels.is_empty() {
        println!("  Labels:");
        for (tid, count) in &labels {
            let name = output::resolve_label(&db.label_tokens, *tid);
            println!("    {:<15} {}", name, count);
        }
    }

    println!();
    Ok(Status::Continue)
}
