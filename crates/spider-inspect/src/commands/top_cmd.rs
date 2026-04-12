//! `top [n]` — show top N nodes by bio score.

use spider_core::bio::{score::calculate, tier::BioTier};
use spider_core::schema::token::TokenId;

use crate::commands::Status;
use crate::context::Context;
use crate::output::print_paged_table;

pub fn run(ctx: &mut Context, args: &[&str]) -> Status {
    let n: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(10);
    let db = &mut ctx.db;

    // Collect IDs first.
    let mut scored_ids = Vec::new();
    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        if let Ok(node) = db.nodes.get(idx) {
            if node.is_deleted() { continue; }
            let score = calculate(&node);
            scored_ids.push((score, node.id));
        }
    }

    scored_ids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<_> = scored_ids.into_iter().take(n).collect();

    // Format each separately to avoid double borrows.
    let mut rows = Vec::new();
    for (score, nid) in top {
        let node = match db.nodes.get(nid - 1) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let tier = BioTier::from_score(score);
        let labels: Vec<_> = node.labels().iter().flatten()
            .filter_map(|lid| TokenId::new(lid.get()).ok())
            .filter_map(|tid| db.label_tokens.get_name(tid).map(String::from))
            .collect();

        rows.push(vec![
            nid.to_string(),
            labels.join(", "),
            format!("{:.2}", score),
            tier.to_string(),
            node.access_count.to_string(),
        ]);
    }

    println!("\nTop {} nodes by bio score:", n);
    print_paged_table(&["ID", "Labels", "Score", "Tier", "Access"], rows);
    println!();
    Status::Continue
}
