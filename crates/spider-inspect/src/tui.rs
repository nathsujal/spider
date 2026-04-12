//! TUI rendering layer — split-pane terminal UI using ratatui.
//!
//! Layout:
//! ┌─────────────────────────────────────────────────┐
//! │  Graph View (40%)  │  Output (60%)             │
//! │                    │                            │
//! │                    │                            │
//! ├─────────────────────────────────────────────────┤
//! │  Status bar (3 rows): db path, counts, input    │
//! └─────────────────────────────────────────────────┘

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};

use crate::context::Context;
use crate::output_globals::set_sink;
use crate::sink::OutputSink;

/// Application state shared between TUI and commands.
#[derive(Debug)]
pub struct AppState {
    /// Command output lines (scrollable).
    pub output_lines: Vec<String>,
    /// Tree view string (for graph pane).
    pub tree_view: String,
    /// Current command input buffer.
    pub input: String,
    /// Database path for display.
    pub db_path: String,
    /// Node count.
    pub node_count: u32,
    /// Edge count.
    pub edge_count: u32,
    /// Scroll offset for output pane.
    pub scroll: usize,
}

impl AppState {
    pub fn new(db_path: &str) -> Self {
        Self {
            output_lines: vec![
                "Spider Inspect — TUI mode".to_string(),
                "Type commands and press Enter. Ctrl+C to quit.".to_string(),
                "".to_string(),
            ],
            tree_view: "No node inspected yet.\n\nUse 'show <id>' or 'tree <doc_id>' to inspect.".to_string(),
            input: String::new(),
            db_path: db_path.to_string(),
            node_count: 0,
            edge_count: 0,
            scroll: 0,
        }
    }

    /// Append a line to output and auto-scroll.
    pub fn append_output(&mut self, line: &str) {
        self.output_lines.push(line.to_string());
        // Auto-scroll: keep scroll at the end.
        self.scroll = self.output_lines.len();
    }

    /// Clear output for a fresh command run.
    pub fn clear_output(&mut self) {
        self.output_lines.clear();
    }

    /// Update counts from database.
    pub fn refresh_counts(&mut self, ctx: &mut Context) {
        let db = &mut ctx.db;
        let mut nodes = 0u32;
        let mut edges = 0u32;
        for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
            if let Ok(node) = db.nodes.get(idx) {
                if !node.is_deleted() { nodes += 1; }
            }
            if idx < db.metadata.next_rel_id.saturating_sub(1) {
                if let Ok(edge) = db.edges.get(idx) {
                    if !edge.is_deleted() { edges += 1; }
                }
            }
        }
        self.node_count = nodes;
        self.edge_count = edges;
    }
}

/// TUI-specific output sink — writes to AppState instead of stdout.
struct TuiSink {
    state: Rc<RefCell<AppState>>,
}

impl OutputSink for TuiSink {
    fn print_line(&mut self, line: &str) {
        self.state.borrow_mut().append_output(line);
    }

    fn print_error(&mut self, msg: &str) {
        self.state.borrow_mut().append_output(&format!("ERROR: {}", msg));
    }

    fn print_ok(&mut self, msg: &str) {
        self.state.borrow_mut().append_output(&format!("OK: {}", msg));
    }

    fn print_table(&mut self, headers: &[&str], rows: Vec<Vec<String>>) {
        use comfy_table::Table;
        let mut t = Table::new();
        t.load_preset(comfy_table::presets::UTF8_FULL);
        if !headers.is_empty() {
            t.set_header(headers);
        }
        for row in rows {
            t.add_row(row);
        }
        let table_str = format!("{}", t);
        let mut state = self.state.borrow_mut();
        for line in table_str.lines() {
            state.append_output(line);
        }
    }

    fn print_paged_table(&mut self, headers: &[&str], rows: Vec<Vec<String>>) {
        // TUI doesn't paginate — just show all rows.
        self.print_table(headers, rows);
    }

    fn set_tree_view(&mut self, tree: String) {
        self.state.borrow_mut().tree_view = tree;
    }

    fn set_node_id(&mut self, id: u32) {
        self.state.borrow_mut().tree_view = format!("Node #{}\n(inspect with 'show {}' for details)", id, id);
    }
}

/// Run the TUI event loop.
pub fn run_tui(mut ctx: Context) -> Result<()> {
    // Setup terminal.
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let db_path = ctx.db.path().display().to_string();
    let mut app = AppState::new(&db_path);
    app.refresh_counts(&mut ctx);

    let result = run_app(&mut terminal, &mut ctx, &mut app);

    // Restore terminal.
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture,
    )?;
    terminal.show_cursor()?;

    result
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    ctx: &mut Context,
    app: &mut AppState,
) -> Result<()> {
    let app_rc = Rc::new(RefCell::new(std::mem::replace(app, AppState::new(""))));

    loop {
        terminal.draw(|frame| ui(frame, &app_rc.borrow()))?;

        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Char(c) => {
                    app_rc.borrow_mut().input.push(c);
                }
                KeyCode::Backspace => {
                    app_rc.borrow_mut().input.pop();
                }
                KeyCode::Enter => {
                    let cmd = app_rc.borrow().input.trim().to_string();
                    if !cmd.is_empty() {
                        app_rc.borrow_mut().clear_output();
                        app_rc.borrow_mut().append_output(&format!("> {}", cmd));

                        // Execute command through normal dispatch.
                        dispatch_command(ctx, &app_rc, &cmd);
                        app_rc.borrow_mut().refresh_counts(ctx);
                    }
                    app_rc.borrow_mut().input.clear();
                }
                KeyCode::Up => {
                    if app_rc.borrow().scroll > 0 {
                        app_rc.borrow_mut().scroll -= 1;
                    }
                }
                KeyCode::Down => {
                    let len = app_rc.borrow().output_lines.len();
                    if app_rc.borrow().scroll < len.saturating_sub(1) {
                        app_rc.borrow_mut().scroll += 1;
                    }
                }
                KeyCode::Esc | KeyCode::Char('q') if app_rc.borrow().input.is_empty() => {
                    app_rc.borrow_mut().append_output("bye");
                    return Ok(());
                }
                _ => {}
            }
        }
    }
}

/// Dispatch a command string through the normal command parser.
fn dispatch_command(ctx: &mut Context, app_rc: &Rc<RefCell<AppState>>, input: &str) {
    use crate::commands::Command;
    use crate::output_globals::set_sink;

    // Install TuiSink as the global output sink.
    set_sink(Box::new(TuiSink { state: Rc::clone(app_rc) }));

    if let Some((cmd, args)) = Command::parse(input) {
        if let Err(e) = cmd.execute(ctx, &args) {
            app_rc.borrow_mut().append_output(&format!("ERROR: {}", e));
        }
    } else if !input.trim().is_empty() {
        app_rc.borrow_mut().append_output(&format!("unknown command: '{}'. Type 'help' for commands.", input));
    }
}

fn ui(frame: &mut ratatui::Frame, app: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),  // top panes
            Constraint::Length(3), // bottom bar
        ])
        .split(frame.area());

    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),  // graph view
            Constraint::Percentage(60),  // output
        ])
        .split(chunks[0]);

    // Graph view pane.
    let graph_block = Block::default()
        .title(" Graph View ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::DarkGray));
    let graph_text = Paragraph::new(app.tree_view.clone())
        .block(graph_block)
        .wrap(Wrap { trim: true });
    frame.render_widget(graph_text, top_chunks[0]);

    // Output pane.
    let output_block = Block::default()
        .title(" Output ")
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::Cyan));
    let display_lines = app.output_lines.len().min(50);
    let start = app.scroll.min(app.output_lines.len().saturating_sub(display_lines));
    let end = (start + display_lines).min(app.output_lines.len());
    let visible_lines: Vec<Line> = app.output_lines[start..end]
        .iter()
        .map(|l| {
            let (style, text) = if l.starts_with("ERROR:") {
                (Style::default().fg(Color::Red).add_modifier(Modifier::BOLD), &l[7..])
            } else if l.starts_with("OK:") {
                (Style::default().fg(Color::Green), &l[4..])
            } else if l.starts_with("> ") {
                (Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD), &l[2..])
            } else {
                (Style::default(), l.as_str())
            };
            Line::from(Span::styled(text.to_string(), style))
        })
        .collect();
    let output_text = Paragraph::new(Text::from(visible_lines))
        .block(output_block)
        .scroll((0, 0));
    frame.render_widget(output_text, top_chunks[1]);

    // Bottom bar.
    let status_line = Line::from(vec![
        Span::styled(
            format!(" {} ", app.db_path),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | "),
        Span::styled(
            format!("nodes:{} edges:{} ", app.node_count, app.edge_count),
            Style::default().fg(Color::Yellow),
        ),
        Span::raw(format!("> {}", app.input)),
    ]);
    let status_block = Block::default()
        .borders(Borders::ALL)
        .style(Style::default().fg(Color::DarkGray));
    let status_text = Paragraph::new(status_line).block(status_block);
    frame.render_widget(status_text, chunks[1]);
}
