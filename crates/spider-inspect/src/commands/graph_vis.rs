//! `graph <id> [depth]` — renders a neighborhood subgraph centered on a node.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};
use colored::Colorize;
use spider_core::bio::{score::calculate, tier::BioTier};
use spider_core::db::nodes::NodeId;
use spider_core::db::rels::Direction;
use spider_core::query::traverse::get_relationships;
use spider_core::schema::token::TokenId;

use crate::commands::Status;
use crate::context::Context;
use crate::output;

const MAX_NEIGHBORS_PER_HOP: usize = 20;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: graph <id> [depth]");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let depth: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);

    let node = ctx.db.nodes.get(id - 1).map_err(|_| anyhow!("node #{} not found", id))?;
    if node.is_deleted() {
        output::print_error(&format!("node #{} is deleted", id));
        return Ok(Status::Continue);
    }

    // BFS from the root node.
    let mut visited: HashSet<u32> = HashSet::new();
    let mut current_layer = vec![id];
    visited.insert(id);

    let mut all_edges: Vec<(u32, String, u32)> = Vec::new();
    let mut next_layer: Vec<u32> = Vec::new();

    for _hop in 0..depth {
        next_layer.clear();

        for &nid in &current_layer {
            let edges = match get_relationships(&mut ctx.db, NodeId::new(nid)?, Direction::Both) {
                Ok(e) => e,
                Err(_) => continue,
            };

            let mut neighbors: Vec<_> = edges.iter()
                .map(|e| {
                    let other = if e.source_id == nid { e.target_id } else { e.source_id };
                    (other, e)
                })
                .filter(|(other, _)| !visited.contains(other))
                .collect();

            neighbors.sort_by(|(a, _), (b, _)| {
                let score_a = node_score(&mut ctx.db, *a);
                let score_b = node_score(&mut ctx.db, *b);
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });

            let limited: Vec<_> = neighbors.into_iter().take(MAX_NEIGHBORS_PER_HOP).collect();

            for (other, edge) in &limited {
                visited.insert(*other);
                next_layer.push(*other);

                let type_name = edge.edge_type()
                    .and_then(|t| TokenId::new(t.get()).ok())
                    .and_then(|tid| ctx.db.edge_type_tokens.get_name(tid).map(String::from))
                    .unwrap_or_else(|| "?".to_string());

                all_edges.push((nid, type_name, *other));
            }
        }

        if next_layer.is_empty() {
            break;
        }
        current_layer = next_layer.clone();
    }

    // Build node info map.
    let mut node_info: HashMap<u32, NodeInfo> = HashMap::new();
    for &nid in &visited {
        if let Ok(n) = ctx.db.nodes.get(nid - 1) {
            if !n.is_deleted() {
                let labels: Vec<String> = n.labels().iter().flatten()
                    .filter_map(|lid| TokenId::new(lid.get()).ok())
                    .filter_map(|tid| ctx.db.label_tokens.get_name(tid).map(String::from))
                    .collect();
                let score = calculate(&n);
                let tier = BioTier::from_score(score);
                node_info.insert(nid, NodeInfo { labels, score, tier });
            }
        }
    }

    // Print.
    println!();
    println!("{}", format!("Subgraph around Node #{} (depth={}, {} nodes, {} edges)", id, depth, node_info.len(), all_edges.len()).bold());
    println!("{}", "─".repeat(70));

    if let Some(info) = node_info.get(&id) {
        print_node_box(id, info, "ROOT");
    }

    let mut seen_nodes: HashSet<u32> = HashSet::new();
    seen_nodes.insert(id);

    for (_src, type_name, tgt) in &all_edges {
        if !seen_nodes.contains(tgt) {
            if let Some(info) = node_info.get(tgt) {
                println!("  ● ──{}──▶ #{} {}", type_name, tgt, tier_badge(&info.tier));
                print_node_box(*tgt, info, "");
                seen_nodes.insert(*tgt);
            }
        } else if let Some(tgt_info) = node_info.get(tgt) {
            println!("  ● ──{}──▶ #{} {}", type_name, tgt, tier_badge(&tgt_info.tier));
        }
    }

    println!();
    Ok(Status::Continue)
}

fn node_score(db: &mut spider_core::db::lifecycle::Spider, id: u32) -> f64 {
    match db.nodes.get(id - 1) {
        Ok(n) if !n.is_deleted() => calculate(&n),
        _ => 0.0,
    }
}

fn tier_badge(tier: &BioTier) -> String {
    match tier {
        BioTier::Hot => format!("[{}]", tier).green().to_string(),
        BioTier::Warm => format!("[{}]", tier).yellow().to_string(),
        BioTier::Cold => format!("[{}]", tier).dimmed().to_string(),
        BioTier::Pruned => format!("[{}]", tier).red().to_string(),
    }
}

struct NodeInfo {
    labels: Vec<String>,
    score: f64,
    tier: BioTier,
}

fn print_node_box(id: u32, info: &NodeInfo, tag: &str) {
    let labels = if info.labels.is_empty() { "(none)".to_string() } else { info.labels.join(", ") };
    let tag_str = if tag.is_empty() { "".to_string() } else { format!(" ({})", tag) };

    println!("  ╭──── Node #{}{} ───────────────────────────", id, tag_str);
    println!("  │  Labels: {}", labels);
    println!("  │  Score:  {:.2} {}", info.score, tier_badge(&info.tier));
    println!("  ╰─────────────────────────────────────────────");
}
