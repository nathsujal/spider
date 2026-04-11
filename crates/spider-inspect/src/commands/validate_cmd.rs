//! `validate` — strict integrity check beyond `broken`.

use std::collections::HashSet;
use colored::Colorize;

use spider_core::store::record::Record;

use crate::commands::Status;
use crate::context::Context;

const MAX_ISSUES: usize = 20;

pub fn run(ctx: &mut Context) -> Status {
    let mut issues = Vec::new();

    // 1. Check every property chain terminates (no cycles, no dangling pointers).
    check_property_chains(ctx, &mut issues);

    // 2. Check string pointers in property blocks point to live DynamicStringRecord.
    check_string_pointers(ctx, &mut issues);

    // 3. Check all token IDs in label slots are within the token store's range.
    check_label_tokens(ctx, &mut issues);

    // 4. Check edge chains.
    check_edge_chains(ctx, &mut issues);

    // Report.
    println!("\n{}", "Strict Integrity Check".bold());
    println!("{}", "─".repeat(50));

    if issues.is_empty() {
        println!("{}", "All checks passed ✓".green());
    } else {
        println!("{} {} issues found:", "ERROR".red().bold(), issues.len());
        for (i, issue) in issues.iter().enumerate().take(MAX_ISSUES) {
            println!("  {}. {}", i + 1, issue);
        }
        if issues.len() > MAX_ISSUES {
            println!("  ... and {} more (stopped at {})", issues.len() - MAX_ISSUES, MAX_ISSUES);
        }
    }
    println!();
    Status::Continue
}

fn check_property_chains(ctx: &mut Context, issues: &mut Vec<String>) {
    let db = &mut ctx.db;

    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        let node = match db.nodes.get(idx) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if node.is_deleted() || node.first_prop_id == 0 {
            continue;
        }

        let mut visited = HashSet::new();
        let mut cursor = node.first_prop_id;
        let mut steps = 0;

        while cursor != 0 {
            if steps > 10_000 {
                issues.push(format!("Node #{}: property chain exceeds 10,000 steps (cycle or corruption)", node.id));
                break;
            }
            if visited.contains(&cursor) {
                issues.push(format!("Node #{}: property chain cycle detected at prop #{}", node.id, cursor));
                break;
            }
            visited.insert(cursor);

            let prop = match db.properties.get(cursor - 1) {
                Ok(p) => p,
                Err(_) => {
                    issues.push(format!("Node #{}: dangling property pointer at prop #{}", node.id, cursor));
                    break;
                }
            };
            if prop.is_deleted() {
                // Deleted mid-chain = corruption.
                if prop.next_prop_id != 0 {
                    issues.push(format!("Node #{}: deleted property #{} has non-zero next_prop_id", node.id, cursor));
                }
                break;
            }

            cursor = prop.next_prop_id;
            steps += 1;
        }

        if issues.len() >= MAX_ISSUES { return; }
    }
}

fn check_string_pointers(ctx: &mut Context, issues: &mut Vec<String>) {
    let db = &mut ctx.db;

    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        let node = match db.nodes.get(idx) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if node.is_deleted() || node.first_prop_id == 0 {
            continue;
        }

        let mut cursor = node.first_prop_id;
        let mut steps = 0;

        while cursor != 0 && steps < 10_000 {
            let prop = match db.properties.get(cursor - 1) {
                Ok(p) => p,
                Err(_) => break,
            };
            if prop.is_deleted() { break; }

            for block in &prop.blocks {
                if block.is_empty() { continue; }
                if block.value_type() == spider_core::schema::property::PropertyType::String {
                    if let Some(ptr) = block.dyn_string_ptr() {
                        // Check the string record exists and is_start=true.
                        match db.strings.get(ptr - 1) {
                            Ok(rec) if rec.is_in_use() => {
                                if !rec.is_start() {
                                    issues.push(format!(
                                        "Node #{}: property points to string block #{} which is not a start block",
                                        node.id, ptr
                                    ));
                                }
                            }
                            Ok(_) => {
                                issues.push(format!(
                                    "Node #{}: property points to deleted string block #{}",
                                    node.id, ptr
                                ));
                            }
                            Err(_) => {
                                issues.push(format!(
                                    "Node #{}: property points to out-of-range string block #{}",
                                    node.id, ptr
                                ));
                            }
                        }
                    }
                }
            }

            cursor = prop.next_prop_id;
            steps += 1;
        }

        if issues.len() >= MAX_ISSUES { return; }
    }
}

fn check_label_tokens(ctx: &mut Context, issues: &mut Vec<String>) {
    let db = &mut ctx.db;
    let max_label_id = db.label_tokens.len() as u8;

    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        let node = match db.nodes.get(idx) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if node.is_deleted() { continue; }

        for lid in node.labels().iter().flatten() {
            if lid.get() > max_label_id {
                issues.push(format!(
                    "Node #{}: label token ID {} exceeds token store range (max {})",
                    node.id, lid.get(), max_label_id
                ));
            }
        }

        if issues.len() >= MAX_ISSUES { return; }
    }
}

fn check_edge_chains(ctx: &mut Context, issues: &mut Vec<String>) {
    let db = &mut ctx.db;

    for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
        let node = match db.nodes.get(idx) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if node.is_deleted() || node.first_edge_id == 0 {
            continue;
        }

        let mut cursor = node.first_edge_id;
        let mut steps = 0;

        while cursor != 0 && steps < 10_000 {
            let edge = match db.edges.get(cursor - 1) {
                Ok(e) => e,
                Err(_) => {
                    issues.push(format!(
                        "Node #{}: dangling edge pointer at edge #{}",
                        node.id, cursor
                    ));
                    break;
                }
            };
            if edge.is_deleted() { break; }

            // Check that this edge actually references this node as source or target.
            if edge.source_id != node.id && edge.target_id != node.id {
                issues.push(format!(
                    "Node #{}: edge #{} does not reference this node (source={}, target={})",
                    node.id, cursor, edge.source_id, edge.target_id
                ));
            }

            let next = if edge.source_id == node.id {
                edge.next_edge_source
            } else {
                edge.next_edge_target
            };
            cursor = next;
            steps += 1;
        }

        if issues.len() >= MAX_ISSUES { return; }
    }
}
