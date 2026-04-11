//! Global output sink — thread-local so commands don't need to carry a sink parameter.

use crate::sink::OutputSink;
use crate::output::make_table;
use colored::Colorize;
use std::io::{self, Write};

// --- StdoutSink defined here to avoid circular deps ---

/// Output sink that prints to stdout with colors.
pub struct StdoutSink;

impl OutputSink for StdoutSink {
    fn print_line(&mut self, line: &str) {
        println!("{}", line);
    }

    fn print_error(&mut self, msg: &str) {
        eprintln!("{}", msg.red().bold());
    }

    fn print_ok(&mut self, msg: &str) {
        println!("{}", msg.green());
    }

    fn print_table(&mut self, headers: &[&str], rows: Vec<Vec<String>>) {
        let t = make_table(headers, rows);
        println!("{}", t);
    }

    fn print_paged_table(&mut self, headers: &[&str], rows: Vec<Vec<String>>) {
        const PAGE_SIZE: usize = 20;

        if rows.is_empty() {
            println!("{}", "(no results)".dimmed());
            return;
        }

        let total = rows.len();
        let mut shown = 0;

        loop {
            let end = (shown + PAGE_SIZE).min(total);
            let page: Vec<Vec<String>> = rows[shown..end].iter().cloned().collect();
            println!("{}", make_table(headers, page));

            shown = end;
            if shown >= total {
                break;
            }

            if !self.confirm_continue(shown, total) {
                println!("{}", "(cancelled)".dimmed());
                break;
            }
        }
    }

    fn confirm_continue(&mut self, shown: usize, total: usize) -> bool {
        print!("\n  {} rows shown, {}/{} total — press Enter for more: ",
            20, shown, total);
        let _ = io::stdout().flush();

        let mut buf = String::new();
        let _ = io::stdin().read_line(&mut buf);
        !buf.trim().eq_ignore_ascii_case("q")
    }
}

// --- Thread-local sink management ---

thread_local! {
    static CURRENT_SINK: std::cell::RefCell<Box<dyn OutputSink>> =
        std::cell::RefCell::new(Box::new(StdoutSink));
}

/// Run code with the given output sink as the current global sink.
pub fn with_sink<T>(f: impl FnOnce(&mut dyn OutputSink) -> T) -> T {
    CURRENT_SINK.with(|cell| {
        let mut sink = cell.borrow_mut();
        f(sink.as_mut())
    })
}

/// Replace the current global sink.
pub fn set_sink(sink: Box<dyn OutputSink>) {
    CURRENT_SINK.with(|cell| {
        *cell.borrow_mut() = sink;
    });
}

/// Convenience wrappers so commands can keep using the existing API.

pub fn print_line(line: &str) {
    with_sink(|s| s.print_line(line));
}

pub fn print_error(msg: &str) {
    with_sink(|s| s.print_error(msg));
}

pub fn print_ok(msg: &str) {
    with_sink(|s| s.print_ok(msg));
}

pub fn print_table(headers: &[&str], rows: Vec<Vec<String>>) {
    with_sink(|s| s.print_table(headers, rows));
}

pub fn print_paged_table(headers: &[&str], rows: Vec<Vec<String>>) {
    with_sink(|s| s.print_paged_table(headers, rows));
}

pub fn set_tree_view(tree: String) {
    with_sink(|s| s.set_tree_view(tree));
}

pub fn set_node_id(id: u32) {
    with_sink(|s| s.set_node_id(id));
}
