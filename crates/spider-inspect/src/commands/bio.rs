//! `bio` — vitality leaderboard, all nodes ranked by bio score.

use anyhow::Result;
use colored::Colorize;
use spider_core::bio::{score::calculate, tier::BioTier};

use crate::commands::Status;
use crate::context::Context;
use crate::output::{self, table};

struct BioEntry {
    id: u32,
    labels: String,
    score: f64,
    tier: BioTier,
    access_count: u32,
    significance: f64,
}

pub fn run(ctx: &mut Context) -> Result<Status> {
    let db = &mut ctx.db;

    let mut entries = Vec::new();
    let mut tier_counts = [0u32; 4]; // Pruned, Cold, Warm, Hot

    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        if let Ok(node) = db.nodes.get(idx) {
            if node.is_deleted() {
                continue;
            }

            let score = calculate(&node);
            let tier = BioTier::from_score(score);

            match tier {
                BioTier::Hot => tier_counts[3] += 1,
                BioTier::Warm => tier_counts[2] += 1,
                BioTier::Cold => tier_counts[1] += 1,
                BioTier::Pruned => tier_counts[0] += 1,
            }

            let labels: Vec<String> = node.labels()
                .into_iter()
                .flatten()
                .map(|lid| output::resolve_label(&mut db.label_tokens, lid.get()))
                .collect();

            entries.push(BioEntry {
                id: node.id,
                labels: if labels.is_empty() { "(none)".to_string() } else { labels.join(", ") },
                score,
                tier,
                access_count: node.access_count,
                significance: node.significance as f64 / 255.0,
            });
        }
    }

    // Sort by score descending.
    entries.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    if entries.is_empty() {
        output::print_error("no nodes found");
        return Ok(Status::Continue);
    }

    println!("\n{}", "Vitality Leaderboard".bold());
    println!("{}", "─".repeat(70));

    let rows: Vec<_> = entries.iter().map(|e| {
        vec![
            e.id.to_string(),
            e.labels.clone(),
            format!("{:.2}", e.score),
            e.tier.to_string(),
            e.access_count.to_string(),
            format!("{:.2}", e.significance),
        ]
    }).collect();

    println!("{}", table(
        &["Node", "Labels", "Score", "Tier", "Access", "Signif."],
        rows,
    ));

    println!("\n  Total: {} nodes  |  Hot: {}  |  Warm: {}  |  Cold: {}  |  Pruned: {}",
        entries.len(), tier_counts[3], tier_counts[2], tier_counts[1], tier_counts[0]);
    println!();

    Ok(Status::Continue)
}
