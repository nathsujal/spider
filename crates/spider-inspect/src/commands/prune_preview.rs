//! `prune-preview` — list nodes that would be pruned (bio score ≤ 0).

use spider_core::bio::{score::calculate, tier::BioTier};
use spider_core::db::nodes::NodeId;
use spider_core::property::list_all;
use spider_core::schema::token::TokenId;

use crate::commands::Status;
use crate::context::Context;
use crate::output::print_paged_table;

pub fn run(ctx: &mut Context) -> Status {
    let db = &mut ctx.db;

    // Collect IDs first.
    let mut prunable_ids = Vec::new();
    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        if let Ok(node) = db.nodes.get(idx) {
            if node.is_deleted() { continue; }
            let score = calculate(&node);
            let tier = BioTier::from_score(score);
            if tier == BioTier::Pruned {
                prunable_ids.push(node.id);
            }
        }
    }

    // Format each separately.
    let mut rows = Vec::new();
    for nid in prunable_ids {
        let node = match db.nodes.get(nid - 1) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let score = calculate(&node);
        let labels: Vec<_> = node.labels().iter().flatten()
            .filter_map(|lid| TokenId::new(lid.get()).ok())
            .filter_map(|tid| db.label_tokens.get_name(tid).map(String::from))
            .collect();

        let props_str = match list_all(db, NodeId::new(nid).unwrap()) {
            Ok(props) => props.iter()
                .map(|p| format!("{}={}", p.key, p.value))
                .take(3)
                .collect::<Vec<_>>()
                .join(", "),
            Err(_) => String::new(),
        };

        rows.push(vec![
            nid.to_string(),
            labels.join(", "),
            format!("{:.2}", score),
            props_str,
        ]);
    }

    println!("\nNodes that would be pruned (bio score ≤ 0): {}", rows.len());
    if rows.is_empty() {
        println!("(none — all nodes are above the pruning threshold)");
    } else {
        print_paged_table(&["ID", "Labels", "Score", "Properties"], rows);
    }
    println!();
    Status::Continue
}
