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
