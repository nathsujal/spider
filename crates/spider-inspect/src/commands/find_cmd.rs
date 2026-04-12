//! `find label <LABEL>` / `find prop <key> <value>` / `find bio <min>` — filtered node queries.

use anyhow::{anyhow, Result};
use spider_core::bio::{score::calculate, tier::BioTier};
use spider_core::db::nodes::NodeId;
use spider_core::property::{get, list_all};
use spider_core::schema::token::TokenId;

use crate::commands::Status;
use crate::context::Context;
use crate::output::{self, print_paged_table};

pub fn run(ctx: &mut Context, sub: &str, args: &[&str]) -> Result<Status> {
    match sub {
        "label" => find_by_label(ctx, args),
        "prop" => find_by_prop(ctx, args),
        "bio" => find_by_bio(ctx, args),
        _ => {
            output::print_error("usage: find [label|prop|bio] ...");
            Ok(Status::Continue)
        }
    }
}

fn find_by_label(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: find label <LABEL>");
        return Ok(Status::Continue);
    }
    let label = args[0];

    let db = &mut ctx.db;
    let label_tid = match db.label_tokens.get_id(label) {
        Some(t) => t,
        None => {
            output::print_ok(&format!("Label '{}' not found — no nodes have this label", label));
            return Ok(Status::Continue);
        }
    };
    let label_id = label_tid.get();

    // Pass 1: collect matching node IDs.
    let mut matching_ids = Vec::new();
    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        if let Ok(node) = db.nodes.get(idx) {
            if node.is_deleted() { continue; }
            if node.labels().iter().flatten().any(|lid| lid.get() == label_id) {
                matching_ids.push(node.id);
            }
        }
    }

    // Pass 2: format each node.
    println!("\nNodes with label [{}]: {}", label, matching_ids.len());
    let rows = format_node_rows(ctx, &matching_ids, None, None)?;
    print_paged_table(&["ID", "Labels", "First Prop", "Score", "Tier"], rows);
    println!();
    Ok(Status::Continue)
}

fn find_by_prop(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.len() < 2 {
        output::print_error("usage: find prop <key> <value>");
        return Ok(Status::Continue);
    }
    let key = args[0];
    let value = args[1..].join(" ");
    let value = strip_quotes(&value);

    let db = &mut ctx.db;
    let _key_tid = match db.prop_key_tokens.get_id(key) {
        Some(t) => t,
        None => {
            output::print_ok(&format!("Property key '{}' not found — no matches", key));
            return Ok(Status::Continue);
        }
    };

    // Pass 1: collect matching node IDs.
    let mut matching_ids = Vec::new();
    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        if let Ok(node) = db.nodes.get(idx) {
            if node.is_deleted() { continue; }
            if node.first_prop_id == 0 { continue; }
            match get(db, NodeId::new(node.id).unwrap(), key) {
                Ok(Some(v)) if v.to_string() == value => matching_ids.push(node.id),
                _ => {}
            }
        }
    }

    let match_str = format!("{}={}", key, value);
    println!("\nNodes with {}: {}", match_str, matching_ids.len());
    let rows = format_node_rows(ctx, &matching_ids, Some(match_str), None)?;
    print_paged_table(&["ID", "Labels", "Match", "Score", "Tier"], rows);
    println!();
    Ok(Status::Continue)
}

fn find_by_bio(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: find bio <min_score>");
        return Ok(Status::Continue);
    }
    let min: f64 = args[0].parse().map_err(|_| anyhow!("invalid score: '{}'", args[0]))?;

    let db = &mut ctx.db;
    // Pass 1: collect matching node IDs.
    let mut matching_ids = Vec::new();
    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        if let Ok(node) = db.nodes.get(idx) {
            if node.is_deleted() { continue; }
            if calculate(&node) >= min {
                matching_ids.push(node.id);
            }
        }
    }

    println!("\nNodes with bio score >= {:.2}: {}", min, matching_ids.len());
    let rows = format_node_rows(ctx, &matching_ids, None, Some(min))?;
    print_paged_table(&["ID", "Labels", "First Prop", "Score", "Tier"], rows);
    println!();
    Ok(Status::Continue)
}

/// Format node rows from a list of node IDs.
fn format_node_rows(
    ctx: &mut Context,
    ids: &[u32],
    match_str: Option<String>,
    min_score: Option<f64>,
) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    for nid in ids {
        let node = match ctx.db.nodes.get(nid - 1) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let score = calculate(&node);
        if let Some(min) = min_score {
            if score < min { continue; }
        }
        let tier = BioTier::from_score(score);

        let labels: Vec<_> = node.labels().iter().flatten()
            .filter_map(|lid| TokenId::new(lid.get()).ok())
            .filter_map(|tid| ctx.db.label_tokens.get_name(tid).map(String::from))
            .collect();

        let preview = match list_all(&mut ctx.db, NodeId::new(*nid).unwrap()) {
            Ok(props) => props.first().map(|p| {
                let val = if p.value.to_string().len() > 30 {
                    format!("{}...", &p.value.to_string()[..27])
                } else {
                    p.value.to_string()
                };
                format!("{}={}", p.key, val)
            }).unwrap_or_default(),
            Err(_) => String::new(),
        };

        let row = if let Some(ref ms) = match_str {
            vec![
                nid.to_string(),
                labels.join(", "),
                ms.clone(),
                format!("{:.2}", score),
                tier.to_string(),
            ]
        } else {
            vec![
                nid.to_string(),
                labels.join(", "),
                preview,
                format!("{:.2}", score),
                tier.to_string(),
            ]
        };
        rows.push(row);
    }

    // Sort by score descending if min_score was specified.
    if min_score.is_some() {
        rows.sort_by(|a, b| {
            let sa: f64 = a[3].parse().unwrap_or(0.0);
            let sb: f64 = b[3].parse().unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    Ok(rows)
}

fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        &s[1..s.len() - 1]
    } else {
        s
    }
}
