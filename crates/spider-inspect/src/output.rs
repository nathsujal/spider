//! Shared output helpers — tables, colors, value formatting.

use colored::Colorize;
use comfy_table::{Table, Attribute, Cell, Row};
use spider_core::bio::tier::BioTier;
use spider_core::schema::token::{TokenId, TokenStore};

/// Prints an error in red.
pub fn print_error(msg: &str) {
    eprintln!("{}", msg.red().bold());
}

/// Prints a success message in green.
#[allow(dead_code)]
pub fn print_ok(msg: &str) {
    println!("{}", msg.green());
}

/// Builds a styled table with the given headers and rows.
pub fn table(headers: &[&str], rows: Vec<Vec<String>>) -> Table {
    let mut t = Table::new();
    t.load_preset(comfy_table::presets::UTF8_FULL);
    t.apply_modifier(comfy_table::modifiers::UTF8_ROUND_CORNERS);

    if !headers.is_empty() {
        let header_row = Row::from(
            headers.iter().map(|h| Cell::new(h).add_attribute(Attribute::Bold)).collect::<Vec<_>>(),
        );
        t.set_header(header_row);
    }

    for row in rows {
        t.add_row(row);
    }

    t
}

/// Resolves a label token ID to its name.
pub fn resolve_label(tokens: &mut TokenStore, id: u8) -> String {
    TokenId::new(id)
        .ok()
        .and_then(|tid| tokens.get_name(tid).map(|s| s.to_string()))
        .unwrap_or_else(|| format!("<{}>", id))
}

/// Resolves an edge type token ID to its name.
#[allow(dead_code)]
pub fn resolve_edge_type(tokens: &mut TokenStore, id: u8) -> String {
    TokenId::new(id)
        .ok()
        .and_then(|tid| tokens.get_name(tid).map(|s| s.to_string()))
        .unwrap_or_else(|| format!("<{}>", id))
}

/// Formats a bio score with tier label, color-coded.
#[allow(dead_code)]
pub fn format_bio(score: f64) -> String {
    let tier = BioTier::from_score(score);
    let colored = match tier {
        BioTier::Hot => format!("{:.2}", score).green().bold().to_string(),
        BioTier::Warm => format!("{:.2}", score).yellow().to_string(),
        BioTier::Cold => format!("{:.2}", score).dimmed().to_string(),
        BioTier::Pruned => format!("{:.2}", score).red().to_string(),
    };
    format!("{colored}  [{tier}]")
}

/// Formats significance as 0.00 with raw /255.
#[allow(dead_code)]
pub fn format_significance(sig: u8) -> String {
    format!("{:.2} ({}/255)", sig as f64 / 255.0, sig)
}

/// Formats a UNIX timestamp as a human-readable date string.
#[allow(dead_code)]
pub fn format_timestamp(unix_secs: u32) -> String {
    // Simple format without chrono dependency.
    let days = unix_secs / 86400;
    let remaining = unix_secs % 86400;
    let hours = remaining / 3600;
    let mins = (remaining % 3600) / 60;
    let secs = remaining % 60;
    // Approximate date from days since epoch (good enough for debug tool).
    let year = 1970 + days / 365;
    let day_of_year = days % 365;
    format!("{year}-DOY{day_of_year:03} {hours:02}:{mins:02}:{secs:02}")
}

/// Sets a string property on a node. Shared helper for write commands.
pub fn set_string_prop(
    spider: &mut spider_core::db::lifecycle::Spider,
    node_id: u32,
    key: &str,
    value: &str,
) -> anyhow::Result<()> {
    use spider_core::schema::property::{PropKeyId, PropertyBlock, PropertyRecord};

    let key_token = spider.prop_key_tokens.get_or_create(key)
        .map_err(|e| anyhow::anyhow!("failed to register property key '{}': {}", key, e))?;
    let key_id = PropKeyId::new(key_token.get())
        .map_err(|_| anyhow::anyhow!("property key token ID out of range"))?;

    let prop_id = spider.metadata.next_prop_id;
    spider.metadata.next_prop_id += 1;

    let node_idx = node_id - 1;
    let mut node = spider.nodes.get(node_idx)?;
    if node.is_deleted() {
        return Err(anyhow::anyhow!("node #{} is deleted", node_id));
    }

    let block = if value.len() <= PropertyBlock::MAX_SHORT_STRING {
        PropertyBlock::from_short_string(key_id, value)
            .map_err(|e| anyhow::anyhow!("property value too long: {}", e))?
    } else {
        use spider_core::schema::dynamic::DynamicStringRecord;
        let data = value.as_bytes();
        let total_len: u32 = data.len().try_into()
            .map_err(|_| anyhow::anyhow!("property value too long"))?;
        let block_count = data.len().div_ceil(DynamicStringRecord::DATA_SIZE);
        let base_id = spider.metadata.next_string_id;
        spider.metadata.next_string_id += block_count as u32;

        let mut next_block: u32 = 0;
        let mut head_string_id: u32 = 0;

        for chunk_idx in (0..block_count).rev() {
            let offset = chunk_idx * DynamicStringRecord::DATA_SIZE;
            let end = (offset + DynamicStringRecord::DATA_SIZE).min(data.len());
            let chunk = &data[offset..end];
            let this_block_id = base_id + chunk_idx as u32;

            let record = if chunk_idx == 0 {
                DynamicStringRecord::new_start(chunk, total_len, next_block)
                    .map_err(|e| anyhow::anyhow!("dynamic string error: {}", e))?
            } else {
                DynamicStringRecord::new_continuation(chunk, next_block)
            };

            spider.strings.append(&[record])?;
            if chunk_idx == 0 {
                head_string_id = this_block_id;
            }
            next_block = this_block_id;
        }

        PropertyBlock::from_dyn_string_ptr(key_id, head_string_id)
    };

    let mut prop_record = PropertyRecord::new();
    prop_record.blocks[0] = block;
    prop_record.prev_prop_id = 0;
    prop_record.next_prop_id = node.first_prop_id;

    if node.first_prop_id != 0 {
        let old_head_idx = node.first_prop_id - 1;
        let mut old_head = spider.properties.get(old_head_idx)?;
        old_head.prev_prop_id = prop_id;
        spider.properties.set(old_head_idx, &old_head)?;
    }

    node.first_prop_id = prop_id;
    spider.nodes.set(node_idx, &node)?;
    spider.properties.append(&[prop_record])?;

    Ok(())
}

/// Paginated table printer. Prints rows in pages of PAGE_SIZE, prompts "press enter for more".
pub fn print_paged_table(headers: &[&str], rows: Vec<Vec<String>>) {
    const PAGE_SIZE: usize = 20;

    if rows.is_empty() {
        println!("(no results)");
        return;
    }

    let total = rows.len();
    let mut shown = 0;

    loop {
        let end = (shown + PAGE_SIZE).min(total);
        let page = &rows[shown..end];
        println!("{}", table(headers, page.to_vec()));

        shown = end;
        if shown >= total {
            break;
        }

        print!("\n  {} rows shown, {}/{} total — press Enter for more: ",
            PAGE_SIZE, shown, total);
        use std::io::Write;
        let _ = std::io::stdout().flush();

        let mut buf = String::new();
        let _ = std::io::stdin().read_line(&mut buf);
        if buf.trim().eq_ignore_ascii_case("q") {
            println!("(cancelled)");
            break;
        }
    }
}
