//! `broken` — integrity check for orphaned nodes, dangling edges, broken properties.

use anyhow::Result;
use colored::Colorize;
use spider_core::store::record::Record;

use crate::commands::Status;
use crate::context::Context;

pub fn run(ctx: &mut Context) -> Result<Status> {
    let db = &mut ctx.db;
    let mut issues = Vec::new();

    // Scan nodes.
    let mut node_count = 0u32;
    let mut deleted_count = 0u32;
    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        if let Ok(node) = db.nodes.get(idx) {
            if node.is_deleted() {
                deleted_count += 1;
                continue;
            }
            node_count += 1;

            // Check first_edge_id points to a valid edge.
            if node.first_edge_id != 0 {
                let eid = node.first_edge_id - 1;
                match db.edges.get(eid) {
                    Ok(e) if e.is_deleted() => {
                        issues.push(format!(
                            "Node #{}: first_edge_id={} points to deleted edge",
                            node.id, node.first_edge_id
                        ));
                    }
                    Err(_) => {
                        issues.push(format!(
                            "Node #{}: first_edge_id={} out of range",
                            node.id, node.first_edge_id
                        ));
                    }
                    _ => {}
                }
            }

            // Check first_prop_id points to a valid property.
            if node.first_prop_id != 0 {
                let pid = node.first_prop_id - 1;
                match db.properties.get(pid) {
                    Ok(p) if p.is_deleted() => {
                        issues.push(format!(
                            "Node #{}: first_prop_id={} points to deleted property",
                            node.id, node.first_prop_id
                        ));
                    }
                    Err(_) => {
                        issues.push(format!(
                            "Node #{}: first_prop_id={} out of range",
                            node.id, node.first_prop_id
                        ));
                    }
                    _ => {}
                }
            }
        }
    }

    // Scan edges.
    let mut edge_count = 0u32;
    for idx in 0..db.metadata.next_rel_id.saturating_sub(1) {
        if let Ok(edge) = db.edges.get(idx) {
            if edge.is_deleted() {
                continue;
            }
            edge_count += 1;

            // Check source node exists and is live.
            if let Ok(src) = db.nodes.get(edge.source_id - 1) {
                if src.is_deleted() {
                    issues.push(format!(
                        "Edge #{}: source node #{} is deleted", edge.id, edge.source_id
                    ));
                }
            } else {
                issues.push(format!(
                    "Edge #{}: source node #{} does not exist", edge.id, edge.source_id
                ));
            }

            // Check target node exists and is live.
            if let Ok(tgt) = db.nodes.get(edge.target_id - 1) {
                if tgt.is_deleted() {
                    issues.push(format!(
                        "Edge #{}: target node #{} is deleted", edge.id, edge.target_id
                    ));
                }
            } else {
                issues.push(format!(
                    "Edge #{}: target node #{} does not exist", edge.id, edge.target_id
                ));
            }
        }
    }

    // Report.
    println!("\n{}", "Integrity Check".bold());
    println!("{}", "─".repeat(50));
    println!("Nodes scanned:    {}", node_count);
    println!("  (deleted: {})", deleted_count);
    println!("Edges scanned:    {}", edge_count);
    println!("Properties:       {} IDs allocated", db.metadata.next_prop_id - 1);
    println!();

    if issues.is_empty() {
        println!("{}", "Issues found: 0  ✓".green().bold());
        println!("\n(All references are valid)");
    } else {
        println!("{} {}", "Issues found:".red().bold(), issues.len().to_string().red().bold());
        println!();
        for issue in &issues {
            println!("  ⚠ {}", issue);
        }
    }

    println!();
    Ok(Status::Continue)
}
