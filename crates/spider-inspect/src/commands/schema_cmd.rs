//! `schema` — print on-disk record layouts for all record types.

use colored::Colorize;

use crate::commands::Status;
use crate::context::Context;
use crate::output::print_paged_table;

pub fn run(_ctx: &mut Context) -> Status {
    println!("\n{}", "On-Disk Record Layouts".bold());
    println!("{}", "─".repeat(70));

    let mut all_rows = Vec::new();

    // Node — 29 bytes
    let node_layout: &[(&str, &str, &str, &str, &str)] = &[
        ("Node (29 bytes)", "", "", "", ""),
        ("  id", "0", "4", "u32", "Node ID (0 = tombstone)"),
        ("  first_edge_id", "4", "4", "u32", "Head of edge linked list"),
        ("  first_prop_id", "8", "4", "u32", "Head of property linked list"),
        ("  labels[0..3]", "12", "4", "u8[4]", "Label token IDs (0 = empty)"),
        ("  access_count", "16", "4", "u32", "Frequency counter"),
        ("  created_at", "20", "4", "u32", "Unix timestamp"),
        ("  last_accessed_at", "24", "4", "u32", "Unix timestamp of last touch"),
        ("  significance", "28", "1", "u8", "Importance 0–255"),
    ];
    for row in node_layout {
        all_rows.push(vec![row.0.to_string(), row.1.to_string(), row.2.to_string(), row.3.to_string(), row.4.to_string()]);
    }

    // Edge — 33 bytes
    let edge_layout: &[(&str, &str, &str, &str, &str)] = &[
        ("", "", "", "", ""),
        ("Edge (33 bytes)", "", "", "", ""),
        ("  id", "0", "4", "u32", "Edge ID (0 = tombstone)"),
        ("  source_id", "4", "4", "u32", "Source node ID"),
        ("  target_id", "8", "4", "u32", "Target node ID"),
        ("  type_id", "12", "1", "u8", "Edge type token ID"),
        ("  prev_edge_source", "13", "4", "u32", "Prev edge in source chain"),
        ("  next_edge_source", "17", "4", "u32", "Next edge in source chain"),
        ("  prev_edge_target", "21", "4", "u32", "Prev edge in target chain"),
        ("  next_edge_target", "25", "4", "u32", "Next edge in target chain"),
        ("  first_prop_id", "29", "4", "u32", "Head of property linked list"),
    ];
    for row in edge_layout {
        all_rows.push(vec![row.0.to_string(), row.1.to_string(), row.2.to_string(), row.3.to_string(), row.4.to_string()]);
    }

    // PropertyRecord — 40 bytes
    let prop_layout: &[(&str, &str, &str, &str, &str)] = &[
        ("", "", "", "", ""),
        ("PropertyRecord (40 bytes)", "", "", "", ""),
        ("  prev_prop_id", "0", "4", "u32", "Previous record in chain (0 = head)"),
        ("  next_prop_id", "4", "4", "u32", "Next record in chain (0 = tail)"),
        ("  blocks[0]", "8", "8", "PropertyBlock", "Packed key+type+value"),
        ("  blocks[1]", "16", "8", "PropertyBlock", "Packed key+type+value"),
        ("  blocks[2]", "24", "8", "PropertyBlock", "Packed key+type+value"),
        ("  blocks[3]", "32", "8", "PropertyBlock", "Packed key+type+value"),
    ];
    for row in prop_layout {
        all_rows.push(vec![row.0.to_string(), row.1.to_string(), row.2.to_string(), row.3.to_string(), row.4.to_string()]);
    }

    // DynamicStringRecord — 128 bytes
    let ds_layout: &[(&str, &str, &str, &str, &str)] = &[
        ("", "", "", "", ""),
        ("DynamicStringRecord (128 bytes)", "", "", "", ""),
        ("  flags", "0", "1", "u8", "bit 0 = in_use, bit 1 = is_start"),
        ("  length", "1", "3", "u24 LE", "Total string bytes (start block)"),
        ("  next_block", "4", "4", "u32", "Next block in chain (0 = end)"),
        ("  data", "8", "120", "u8[120]", "UTF-8 payload"),
    ];
    for row in ds_layout {
        all_rows.push(vec![row.0.to_string(), row.1.to_string(), row.2.to_string(), row.3.to_string(), row.4.to_string()]);
    }

    print_paged_table(&["Field", "Offset", "Size", "Type", "Description"], all_rows);
    println!();
    Status::Continue
}
