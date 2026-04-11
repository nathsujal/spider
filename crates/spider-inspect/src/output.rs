//! Shared output helpers — tables, colors, value formatting.

use colored::Colorize;
use comfy_table::{Table, Attribute, Cell, Row};

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
#[allow(dead_code)]
pub fn table(headers: &[&str], rows: Vec<Vec<String>>) -> Table {
    let mut t = Table::new();
    t.load_preset(comfy_table::presets::UTF8_FULL);
    t.apply_modifier(comfy_table::modifiers::UTF8_ROUND_CORNERS);

    if !headers.is_empty() {
        let header_row = Row::from(
            headers.iter().map(|h| {
                Cell::new(h).add_attribute(Attribute::Bold)
            }).collect::<Vec<_>>(),
        );
        t.set_header(header_row);
    }

    for row in rows {
        t.add_row(row);
    }

    t
}
