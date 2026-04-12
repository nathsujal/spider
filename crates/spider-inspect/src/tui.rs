//! TUI rendering layer — split-pane terminal UI with force-directed graph.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use ratatui::widgets::canvas::{Canvas,  Line as CanvasLine, Painter, Shape};

use crate::commands::Command;
use crate::context::Context as AppContext;
use crate::output_globals::set_sink;
use crate::sink::OutputSink;

// Color palette

mod c {
    use ratatui::style::Color;
    pub const BG: Color = Color::Rgb(12, 12, 16);
    pub const BORDER: Color = Color::Rgb(28, 28, 36);
    pub const DIM: Color = Color::Rgb(60, 60, 70);
    pub const CYAN: Color = Color::Rgb(79, 195, 247);
    pub const GREEN: Color = Color::Rgb(129, 199, 132);
    pub const PURPLE: Color = Color::Rgb(206, 147, 216);
    #[allow(dead_code)]
    pub const AMBER: Color = Color::Rgb(255, 183, 77);
    pub const RED: Color = Color::Rgb(229, 115, 115);
    pub const WHITE: Color = Color::Rgb(224, 224, 224);
}

fn dim(c: Color, f: f64) -> Color {
    if let Color::Rgb(r, g, b) = c {
        Color::Rgb(((r as f64)*f) as u8, ((g as f64)*f) as u8, ((b as f64)*f) as u8)
    } else { c }
}

// Physics config

#[derive(Debug, Clone, Copy)]
pub struct PhysicsConfig {
    pub repulsion: f64,
    pub spring_stiffness: f64,
    pub rest_length: f64,
    pub damping: f64,
    pub center_pull: f64,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self { repulsion: 2000.0, spring_stiffness: 0.04, rest_length: 80.0, damping: 0.85, center_pull: 0.003 }
    }
}

// Graph data types

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeLabel { Document, Proposition, Entity, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind { Contains, Mentions, Other }

#[derive(Debug)]
pub struct GraphNode {
    pub id: u32,
    pub label: NodeLabel,
    pub x: f64, pub y: f64,
    pub vx: f64, pub vy: f64,
}

#[derive(Debug)]
pub struct GraphEdge {
    pub source: usize,
    pub target: usize,
    pub edge_type: EdgeKind,
}

#[derive(Debug)]
pub struct GraphState {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub highlighted: Option<u32>,
}

impl GraphState {
    pub fn new() -> Self { Self { nodes: vec![], edges: vec![], highlighted: None } }
    #[allow(dead_code)]
    pub fn clear(&mut self) { self.nodes.clear(); self.edges.clear(); self.highlighted = None; }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    pub fn tick(&mut self, cfg: &PhysicsConfig) {
        let n = self.nodes.len();
        if n == 0 { return; }
        for i in 0..n {
            for j in (i+1)..n {
                let dx = self.nodes[j].x - self.nodes[i].x;
                let dy = self.nodes[j].y - self.nodes[i].y;
                let d2 = dx*dx + dy*dy; let d = d2.sqrt().max(1.0);
                let f = cfg.repulsion / d2;
                let fx = (dx/d)*f; let fy = (dy/d)*f;
                self.nodes[i].vx -= fx; self.nodes[i].vy -= fy;
                self.nodes[j].vx += fx; self.nodes[j].vy += fy;
            }
        }
        for e in &self.edges {
            if e.source >= n || e.target >= n { continue; }
            let dx = self.nodes[e.target].x - self.nodes[e.source].x;
            let dy = self.nodes[e.target].y - self.nodes[e.source].y;
            let d = (dx*dx+dy*dy).sqrt().max(1.0);
            let f = cfg.spring_stiffness * (d - cfg.rest_length);
            let fx = (dx/d)*f; let fy = (dy/d)*f;
            self.nodes[e.source].vx += fx; self.nodes[e.source].vy += fy;
            self.nodes[e.target].vx -= fx; self.nodes[e.target].vy -= fy;
        }
        let (cx, cy) = (300.0, 200.0);
        for nd in &mut self.nodes {
            nd.vx += (cx-nd.x)*cfg.center_pull;
            nd.vy += (cy-nd.y)*cfg.center_pull;
            nd.vx *= cfg.damping; nd.vy *= cfg.damping;
            nd.vx = nd.vx.clamp(-20.0, 20.0); nd.vy = nd.vy.clamp(-20.0, 20.0);
            nd.x += nd.vx; nd.y += nd.vy;
            nd.x = nd.x.clamp(10.0, 590.0); nd.y = nd.y.clamp(10.0, 390.0);
        }
    }

    fn find_idx(&self, id: u32) -> Option<usize> {
        self.nodes.iter().position(|n| n.id == id)
    }
}

/// Build GraphState from the database by loading a neighborhood.
fn load_neighborhood(ctx: &mut AppContext, center_id: u32, depth: usize) -> Result<GraphState> {
    use spider_core::db::nodes::NodeId;
    use spider_core::db::rels::Direction;
    use spider_core::query::traverse::get_relationships;

    let mut gs = GraphState::new();
    gs.highlighted = Some(center_id);
    let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut queue = vec![(center_id, 0usize)];
    let mut edge_list: Vec<(u32, u32, EdgeKind)> = vec![];

    if ctx.db.nodes.get(center_id - 1).is_err() { return Ok(gs); }

    while let Some((nid, d)) = queue.pop() {
        if d > depth || visited.contains(&nid) { continue; }
        visited.insert(nid);
        let node = match ctx.db.nodes.get(nid - 1) {
            Ok(n) => n, Err(_) => continue,
        };
        if node.is_deleted() { continue; }

        let label = node.labels().iter().flatten().find_map(|lid| {
            spider_core::schema::token::TokenId::new(lid.get()).ok()
                .and_then(|tid| ctx.db.label_tokens.get_name(tid).map(String::from))
        });
        let nl = match label.as_deref() {
            Some("DOCUMENT") => NodeLabel::Document,
            Some("PROPOSITION") => NodeLabel::Proposition,
            Some("ENTITY") => NodeLabel::Entity,
            _ => NodeLabel::Unknown,
        };
        let angle = (gs.nodes.len() as f64) * std::f64::consts::PI * 2.0 / 8.0;
        let radius = 30.0 + (d as f64) * 60.0;
        gs.nodes.push(GraphNode {
            id: nid, label: nl,
            x: 300.0 + radius*angle.cos(), y: 200.0 + radius*angle.sin(),
            vx: 0.0, vy: 0.0,
        });

        let edges = match get_relationships(&mut ctx.db, NodeId::new(nid)?, Direction::Both) {
            Ok(e) => e, Err(_) => continue,
        };
        for e in &edges {
            let (other, kind) = if e.source_id == nid {
                let tname = e.edge_type()
                    .and_then(|t| spider_core::schema::token::TokenId::new(t.get()).ok())
                    .and_then(|tid| ctx.db.edge_type_tokens.get_name(tid).map(String::from))
                    .unwrap_or_default();
                let k = match tname.as_str() { "CONTAINS" => EdgeKind::Contains, "MENTIONS" => EdgeKind::Mentions, _ => EdgeKind::Other };
                (e.target_id, k)
            } else {
                let tname = e.edge_type()
                    .and_then(|t| spider_core::schema::token::TokenId::new(t.get()).ok())
                    .and_then(|tid| ctx.db.edge_type_tokens.get_name(tid).map(String::from))
                    .unwrap_or_default();
                let k = match tname.as_str() { "CONTAINS" => EdgeKind::Contains, "MENTIONS" => EdgeKind::Mentions, _ => EdgeKind::Other };
                (e.source_id, k)
            };
            edge_list.push((nid, other, kind));
            if !visited.contains(&other) { queue.push((other, d+1)); }
        }
    }

    for (s, t, k) in edge_list {
        if let (Some(si), Some(ti)) = (gs.find_idx(s), gs.find_idx(t)) {
            gs.edges.push(GraphEdge { source: si, target: ti, edge_type: k });
        }
    }
    Ok(gs)
}

// Output line types

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind { Command, Ok, Error, Info, Dim }

#[derive(Debug)]
pub struct StyledLine { pub content: String, pub kind: LineKind }

impl StyledLine {
    pub fn to_ratatui_line(&self) -> Line<'static> {
        let (color, modifier) = match self.kind {
            LineKind::Command => (c::CYAN, Modifier::BOLD),
            LineKind::Ok => (c::GREEN, Modifier::empty()),
            LineKind::Error => (c::RED, Modifier::BOLD),
            LineKind::Info => (c::WHITE, Modifier::empty()),
            LineKind::Dim => (c::DIM, Modifier::empty()),
        };
        Line::from(Span::styled(self.content.clone(), Style::default().fg(color).add_modifier(modifier)))
    }
}

// Application state

#[derive(Debug)]
pub struct AppState {
    pub output_lines: Vec<StyledLine>,
    pub graph: GraphState,
    pub physics_config: PhysicsConfig,
    pub input: String,
    pub scroll_offset: usize,
    pub auto_scroll: bool,
    #[allow(dead_code)]
    pub db_label: String,
    pub node_count: u32,
    pub edge_count: u32,
    pub highlighted_node: Option<u32>,
}

impl AppState {
    pub fn new(db_path: &str) -> Self {
        Self {
            output_lines: vec![
                StyledLine { content: "Spider Inspect — TUI".into(), kind: LineKind::Info },
                StyledLine { content: "Type commands and press Enter. q or Esc to quit.".into(), kind: LineKind::Info },
                StyledLine { content: String::new(), kind: LineKind::Dim },
            ],
            graph: GraphState::new(),
            physics_config: PhysicsConfig::default(),
            input: String::new(),
            scroll_offset: 0, auto_scroll: true,
            db_label: db_path.into(),
            node_count: 0, edge_count: 0,
            highlighted_node: None,
        }
    }
    #[allow(dead_code)]
    pub fn append_output(&mut self, line: &str) {
        self.output_lines.push(StyledLine { content: line.into(), kind: LineKind::Info });
        self.scroll_offset = self.output_lines.len(); self.auto_scroll = true;
    }
    pub fn append_styled(&mut self, line: &str, kind: LineKind) {
        self.output_lines.push(StyledLine { content: line.into(), kind });
        if self.auto_scroll { self.scroll_offset = self.output_lines.len(); }
    }
    pub fn clear_output(&mut self) { self.output_lines.clear(); }
    pub fn refresh_counts(&mut self, ctx: &mut AppContext) {
        let db = &mut ctx.db;
        let mut nodes = 0u32; let mut edges = 0u32;
        for idx in 0..db.metadata.next_node_id.saturating_sub(1) {
            if let Ok(n) = db.nodes.get(idx) { if !n.is_deleted() { nodes += 1; } }
        }
        for idx in 0..db.metadata.next_rel_id.saturating_sub(1) {
            if let Ok(e) = db.edges.get(idx) { if !e.is_deleted() { edges += 1; } }
        }
        self.node_count = nodes; self.edge_count = edges;
    }
}

// TuiSink

struct TuiSink { state: Rc<RefCell<AppState>> }

impl OutputSink for TuiSink {
    fn print_line(&mut self, line: &str) { self.state.borrow_mut().append_output(line); }
    fn print_error(&mut self, msg: &str) { self.state.borrow_mut().append_styled(msg, LineKind::Error); }
    fn print_ok(&mut self, msg: &str) { self.state.borrow_mut().append_styled(msg, LineKind::Ok); }
    fn print_table(&mut self, headers: &[&str], rows: Vec<Vec<String>>) {
        use comfy_table::Table;
        let mut t = Table::new();
        t.load_preset(comfy_table::presets::UTF8_FULL);
        if !headers.is_empty() { t.set_header(headers); }
        for row in rows { t.add_row(row); }
        let mut st = self.state.borrow_mut();
        for line in format!("{}", t).lines() { st.append_styled(line, LineKind::Dim); }
    }
    fn print_paged_table(&mut self, headers: &[&str], rows: Vec<Vec<String>>) {
        self.print_table(headers, rows);
    }
    fn set_tree_view(&mut self, _tree: String) {}
    fn set_node_id(&mut self, id: u32) { self.state.borrow_mut().highlighted_node = Some(id); }
}

// Command dispatch

fn dispatch_command(ctx: &mut AppContext, app_rc: &Rc<RefCell<AppState>>, input: &str) -> crate::commands::Status {
    set_sink(Box::new(TuiSink { state: Rc::clone(app_rc) }));
    if let Some((cmd, args)) = Command::parse(input) {
        match cmd.execute(ctx, &args) {
            Ok(s) => s,
            Err(e) => {
                app_rc.borrow_mut().append_styled(&format!("{}", e), LineKind::Error);
                crate::commands::Status::Continue
            }
        }
    } else if !input.trim().is_empty() {
        app_rc.borrow_mut().append_styled(&format!("unknown command: '{}'. Type 'help'.", input), LineKind::Error);
        crate::commands::Status::Continue
    } else {
        crate::commands::Status::Continue
    }
}

// Canvas drawing

use ratatui::widgets::canvas::Context as CanvasCtx;

/// Draw a filled circle as a grid of points (visible at braille resolution).
fn draw_filled_circle(painter: &mut Painter<'_, '_>, cx: f64, cy: f64, radius: f64, color: ratatui::style::Color) {
    let r = radius.ceil() as i32;
    for dx in -r..=r {
        for dy in -r..=r {
            let dist = ((dx*dx + dy*dy) as f64).sqrt();
            if dist <= radius as f64 {
                let x = cx + dx as f64;
                let y = cy + dy as f64;
                if let Some((px, py)) = painter.get_point(x, y) {
                    painter.paint(px, py, color);
                }
            }
        }
    }
}

fn draw_graph(canvas_ctx: &mut CanvasCtx<'_>, gs: &GraphState) {
    if gs.is_empty() { return; }
    let highlighted = gs.highlighted;
    let mut painter = Painter::from(canvas_ctx);

    // Draw edges — much brighter opacity.
    for edge in &gs.edges {
        if edge.source >= gs.nodes.len() || edge.target >= gs.nodes.len() { continue; }
        let src = &gs.nodes[edge.source]; let tgt = &gs.nodes[edge.target];
        let is_hl = highlighted.map_or(false, |h| h == src.id || h == tgt.id);
        let (color, opacity) = match edge.edge_type {
            EdgeKind::Contains => (c::CYAN, if is_hl { 1.0 } else { 0.6 }),
            EdgeKind::Mentions => (c::PURPLE, if is_hl { 1.0 } else { 0.5 }),
            EdgeKind::Other => (c::DIM, if is_hl { 0.8 } else { 0.35 }),
        };
        let c = dim(color, opacity);
        CanvasLine { x1: src.x, y1: src.y, x2: tgt.x, y2: tgt.y, color: c }.draw(&mut painter);
        if is_hl {
            let c2 = dim(color, 0.7);
            CanvasLine { x1: src.x, y1: src.y-2.0, x2: tgt.x, y2: tgt.y-2.0, color: c2 }.draw(&mut painter);
            CanvasLine { x1: src.x, y1: src.y+2.0, x2: tgt.x, y2: tgt.y+2.0, color: c2 }.draw(&mut painter);
        }
    }

    // Draw nodes as filled circles (much more visible than thin outlines).
    for node in &gs.nodes {
        let (color, radius) = match node.label {
            NodeLabel::Document => (c::CYAN, 8.0),
            NodeLabel::Proposition => (c::GREEN, 6.0),
            NodeLabel::Entity => (c::PURPLE, 5.0),
            NodeLabel::Unknown => (c::DIM, 4.0),
        };
        draw_filled_circle(&mut painter, node.x, node.y, radius, color);

        // Highlight: bright outer ring.
        if highlighted == Some(node.id) {
            let ring_radius = radius + 3.0;
            for angle in 0..360 {
                let a = (angle as f64).to_radians();
                let rx = node.x + ring_radius * a.cos();
                let ry = node.y + ring_radius * a.sin();
                if let Some((px, py)) = painter.get_point(rx, ry) {
                    painter.paint(px, py, dim(color, 0.7));
                }
            }
        }
    }
}

// UI rendering

fn ui(frame: &mut Frame, app: &AppState) {
    let chunks = Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(3)]).split(frame.area());
    let top_chunks = Layout::default().direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)]).split(chunks[0]);

    // Graph pane.
    let graph_block = Block::default().title(" graph ")
        .title_style(Style::default().fg(c::DIM))
        .borders(Borders::ALL).border_style(Style::default().fg(c::BORDER));
    let canvas = Canvas::default().block(graph_block.clone())
        .background_color(c::BG).x_bounds([-10.0, 610.0]).y_bounds([-10.0, 410.0])
        .paint(|ctx| { draw_graph(ctx, &app.graph); });
    frame.render_widget(canvas, top_chunks[0]);

    // Overlay hint ONLY when graph is truly empty.
    if app.graph.is_empty() {
        let ov = Layout::default().direction(Direction::Vertical)
            .constraints([Constraint::Fill(1), Constraint::Length(3), Constraint::Fill(1)]).split(top_chunks[0]);
        frame.render_widget(
            Paragraph::new(Span::styled("no graph loaded · use show <id> or tree <doc_id>", Style::default().fg(c::DIM)))
                .alignment(Alignment::Center).wrap(Wrap { trim: true }), ov[1]);
    }

    // Output pane.
    let output_block = Block::default().title(" output ")
        .title_style(Style::default().fg(c::DIM))
        .borders(Borders::ALL).border_style(Style::default().fg(c::BORDER));
    let display_lines = 30;
    let start = app.scroll_offset.min(app.output_lines.len().saturating_sub(display_lines));
    let end = (start + display_lines).min(app.output_lines.len());
    let visible: Vec<Line> = app.output_lines[start..end].iter().map(|sl| sl.to_ratatui_line()).collect();
    frame.render_widget(Paragraph::new(Text::from(visible)).block(output_block), top_chunks[1]);

    // Input bar (3 rows).
    let input_chunks = Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)]).split(chunks[1]);

    frame.render_widget(Block::default().borders(Borders::TOP).border_style(Style::default().fg(c::DIM)), input_chunks[0]);

    let cursor = if app.input.is_empty() {
        Span::styled("█", Style::default().bg(c::GREEN).fg(c::BG))
    } else {
        Span::styled(&app.input, Style::default().fg(c::WHITE))
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ❯ spider › ", Style::default().fg(c::CYAN)), cursor,
        ])),
        input_chunks[1],
    );

    let hints = Line::from(vec![
        Span::styled("show <id>", Style::default().fg(c::DIM)),
        Span::styled(" · ", Style::default().fg(c::BORDER)),
        Span::styled("tree <doc>", Style::default().fg(c::DIM)),
        Span::styled(" · ", Style::default().fg(c::BORDER)),
        Span::styled("bio", Style::default().fg(c::DIM)),
        Span::styled(" · ", Style::default().fg(c::BORDER)),
        Span::styled("stats", Style::default().fg(c::DIM)),
        Span::styled(" · ", Style::default().fg(c::BORDER)),
        Span::styled("help", Style::default().fg(c::DIM)),
        Span::styled("    [↑↓ scroll]  [ctrl+L clear]  [q quit]", Style::default().fg(c::BORDER)),
    ]);
    frame.render_widget(Paragraph::new(hints), input_chunks[2]);
}

// Main loop

fn run_app<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, ctx: &mut AppContext, app: &mut AppState) -> Result<()> {
    let app_rc = Rc::new(RefCell::new(std::mem::replace(app, AppState::new(""))));
    let tick_rate = Duration::from_millis(16);
    let mut last_tick = Instant::now();

    loop {
        let cfg = app_rc.borrow().physics_config; app_rc.borrow_mut().graph.tick(&cfg);
        terminal.draw(|f| ui(f, &app_rc.borrow()))?;

        let timeout = tick_rate.checked_sub(last_tick.elapsed()).unwrap_or_default();
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press { continue; }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    *app = Rc::try_unwrap(app_rc).unwrap().into_inner(); return Ok(());
                }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('l') {
                    app_rc.borrow_mut().clear_output(); continue;
                }
                match key.code {
                    KeyCode::Esc => { *app = Rc::try_unwrap(app_rc).unwrap().into_inner(); return Ok(()); }
                    KeyCode::Char('q') if app_rc.borrow().input.is_empty() => {
                        *app = Rc::try_unwrap(app_rc).unwrap().into_inner(); return Ok(());
                    }
                    KeyCode::Char(c) => { app_rc.borrow_mut().input.push(c); app_rc.borrow_mut().auto_scroll = true; }
                    KeyCode::Backspace => { app_rc.borrow_mut().input.pop(); app_rc.borrow_mut().auto_scroll = true; }
                    KeyCode::Enter => {
                        let cmd = app_rc.borrow().input.trim().to_string();
                        if !cmd.is_empty() {
                            app_rc.borrow_mut().append_styled(&format!("> {}", cmd), LineKind::Command);
                            let status = dispatch_command(ctx, &app_rc, &cmd);

                            // Update graph if there's a highlighted node.
                            let maybe_nid = { app_rc.borrow().highlighted_node };
                            if let Some(nid) = maybe_nid {
                                match load_neighborhood(ctx, nid, 2) {
                                    Ok(gs) => { app_rc.borrow_mut().graph = gs; }
                                    Err(e) => {
                                        app_rc.borrow_mut().append_styled(
                                            &format!("graph load error: {}", e),
                                            LineKind::Error,
                                        );
                                    }
                                }
                            }
                            app_rc.borrow_mut().refresh_counts(ctx);
                            if matches!(status, crate::commands::Status::Quit) {
                                *app = Rc::try_unwrap(app_rc).unwrap().into_inner(); return Ok(());
                            }
                        }
                        app_rc.borrow_mut().input.clear();
                    }
                    KeyCode::Up => { let mut a = app_rc.borrow_mut(); if a.scroll_offset > 0 { a.scroll_offset -= 1; a.auto_scroll = false; } }
                    KeyCode::Down => {
                        let mut a = app_rc.borrow_mut();
                        let len = a.output_lines.len(); let display = 30; let mx = len.saturating_sub(display);
                        if a.scroll_offset < mx { a.scroll_offset += 1; } else { a.auto_scroll = true; a.scroll_offset = len; }
                    }
                    _ => {}
                }
            }
        }
        last_tick = Instant::now();
    }
}

/// Run the TUI event loop.
pub fn run_tui(mut ctx: AppContext) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let db_path = ctx.db.path().display().to_string();
    let mut app = AppState::new(&db_path);
    app.refresh_counts(&mut ctx);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_app(&mut terminal, &mut ctx, &mut app)
    }));

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(anyhow::anyhow!("TUI panicked")),
    }
}
