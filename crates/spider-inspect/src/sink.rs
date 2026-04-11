//! Output abstraction layer — allows commands to write to either
//! stdout (REPL) or a TUI buffer.

/// Trait abstracting all output operations used by commands.
///
/// The REPL uses [`StdoutSink`], while the TUI will use [`TuiSink`]
/// (added in a later task). This lets the same command logic work
/// in both environments.
pub trait OutputSink {
    /// Print a plain line to output.
    fn print_line(&mut self, line: &str);

    /// Print an error message (typically styled red).
    fn print_error(&mut self, msg: &str);

    /// Print a success message (typically styled green).
    fn print_ok(&mut self, msg: &str);

    /// Print a table with the given headers and rows.
    fn print_table(&mut self, headers: &[&str], rows: Vec<Vec<String>>);

    /// Print paginated table output. For the REPL this prompts for
    /// more pages; for the TUI it just appends all lines.
    fn print_paged_table(&mut self, headers: &[&str], rows: Vec<Vec<String>>);

    /// Set the tree/graph view for the most recently inspected node.
    fn set_tree_view(&mut self, _tree: String) {}

    /// Record the most recently inspected node ID.
    fn set_node_id(&mut self, _id: u32) {}

    /// Prompt the user to continue (for pagination in REPL).
    /// Returns `true` to continue showing more, `false` to cancel.
    fn confirm_continue(&mut self, _shown: usize, _total: usize) -> bool {
        true
    }
}
