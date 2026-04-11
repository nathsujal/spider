//! `why-dead <node_id>` — explain why a node has a low bio score.
use colored::Colorize;

use anyhow::{anyhow, Result};
use spider_core::bio::score::BioParams;
use spider_core::db::nodes::NodeId;

use crate::commands::Status;
use crate::context::Context;
use crate::output;

pub fn run(ctx: &mut Context, args: &[&str]) -> Result<Status> {
    if args.is_empty() {
        output::print_error("usage: why-dead <node_id>");
        return Ok(Status::Continue);
    }

    let id: u32 = args[0].parse().map_err(|_| anyhow!("invalid node ID: '{}'", args[0]))?;
    let _node_id = NodeId::new(id)?;

    let node = ctx.db.nodes.get(id - 1).map_err(|_| anyhow!("node #{} not found", id))?;
    if node.is_deleted() {
        output::print_error(&format!("node #{} is deleted", id));
        return Ok(Status::Continue);
    }

    let params = BioParams {
        w_sig: ctx.db.metadata.bio_w_sig,
        w_freq: ctx.db.metadata.bio_w_freq,
        gravity: ctx.db.metadata.bio_gravity,
    };

    let score = calculate_with_params(&node, &params);
    let significance = node.significance as f64 / 255.0;
    let freq = (1.0 + node.access_count as f64).ln() * 10.0;
    let sig_contribution = significance * params.w_sig * 100.0;
    let freq_contribution = freq * params.w_freq;
    let numerator = sig_contribution + freq_contribution;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;
    let age_secs = now.saturating_sub(node.last_accessed_at);
    let days = age_secs as f64 / 86400.0;
    let denominator = (days + 2.0).powf(params.gravity);

    println!("\n{}", format!("Why Node #{} has a low bio score", id).bold());
    println!("{}", "─".repeat(50));

    let tier = spider_core::bio::tier::BioTier::from_score(score);
    println!("Bio Score:      {:.2}  [{}]", score, tier);
    println!();

    println!("Factor breakdown:");
    println!("  Significance:  {:.2} × {:.1} × 100 = {:>8.2}", significance, params.w_sig, sig_contribution);
    println!("  Frequency:     ln(1 + {}) × 10 × {:.1} = {:>8.2}", node.access_count, params.w_freq, freq_contribution);
    println!("  {:─<49}", "");
    println!("  Numerator:                              {:>8.2}", numerator);
    println!("  Denominator:  ({:.1} days + 2)^{:.1}    {:>8.2}", days, params.gravity, denominator);
    println!("  {:─<49}", "");
    println!("  Final score:                            {:>8.2}", score);
    println!();

    // Verdict.
    let mut reasons = Vec::new();
    if significance < 0.3 {
        reasons.push(format!("very low significance ({:.2} / 1.00)", significance));
    } else if significance < 0.6 {
        reasons.push(format!("moderate significance ({:.2} / 1.00)", significance));
    }
    if node.access_count == 0 {
        reasons.push("never been accessed".to_string());
    } else if node.access_count < 5 {
        reasons.push(format!("rarely accessed ({} times)", node.access_count));
    }
    if days > 90.0 {
        reasons.push(format!("not accessed in {:.0} days", days));
    } else if days > 7.0 {
        reasons.push(format!("last accessed {:.0} days ago", days));
    }

    if !reasons.is_empty() {
        println!("Verdict: This node was {}", reasons.join(", "));
        println!();

        // Suggestions.
        println!("To improve this node's score:");
        if significance < 0.5 {
            println!("  → Increase significance (currently {}/{})", node.significance, 255);
        }
        if node.access_count < 5 {
            println!("  → Access the node more (currently {} accesses)", node.access_count);
        }
        if days > 1.0 {
            println!("  → Reduce time since last access (currently {:.0} days)", days);
        }
        println!();
    }

    Ok(Status::Continue)
}

fn calculate_with_params(node: &spider_core::schema::node::Node, params: &BioParams) -> f64 {
    let s = (node.significance as f64 / 255.0) * params.w_sig * 100.0;
    let f = (1.0 + node.access_count as f64).ln() * 10.0 * params.w_freq;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32;
    let days = (now.saturating_sub(node.last_accessed_at) as f64) / 86400.0;
    (s + f) / (days + 2.0).powf(params.gravity)
}
