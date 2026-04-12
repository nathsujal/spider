# spider-inspect — Implementation Status

> All 5 phases complete. 18 REPL commands + TUI mode.

## Phase 1: Write Commands ✅ DONE
| Command | File | Description |
|---|---|---|
| `create node <L> [k=v]` | create_node.rs | New node with label + properties |
| `create edge <s> <T> <d>` | create_edge.rs | Directed edge between nodes |
| `set <id> <k> <v>` | cmd_set.rs | String property on node |
| `delete node <id>` | delete_node.rs | Soft-delete with edge warning |
| `delete edge <id>` | delete_edge.rs | Soft-delete edge |
| `touch <id>` | cmd_touch.rs | Bump access count |
| `sig <id> <0-255>` | cmd_sig.rs | Set significance, print bio score |

## Phase 2: Graph Visualization ✅ DONE
| Command | File | Description |
|---|---|---|
| `tree <doc_id>` | tree.rs | Unicode tree: Document → Propositions → Entities |
| `graph <id> [depth]` | graph_vis.rs | BFS subgraph with colored tier boxes |

## Phase 3: Query & Filter ✅ DONE
| Command | File | Description |
|---|---|---|
| `find label <L>` | find_cmd.rs | All nodes with label, paginated |
| `find prop <k> <v>` | find_cmd.rs | Nodes matching property, paginated |
| `find bio <min>` | find_cmd.rs | Nodes above bio score, sorted desc |
| `top [n]` | top_cmd.rs | Top N by bio score (default 10) |
| `prune-preview` | prune_preview.rs | Nodes that would be pruned (score ≤ 0) |

## Phase 4: Debug Tools ✅ DONE
| Command | File | Description |
|---|---|---|
| `schema` | schema_cmd.rs | On-disk record layouts (Node/Edge/Property/DynamicString) |
| `hex <id>` | hex_dump.rs | Hex dump of node record with field annotations |
| `validate` | validate_cmd.rs | Strict integrity: chain cycles, dangling pointers, token ranges |

## Phase 5: TUI Rewrite ✅ DONE
| Component | File | Description |
|---|---|---|
| OutputSink trait | sink.rs | Abstract output layer for REPL/TUI |
| Global sink | output_globals.rs | Thread-local sink routing |
| Feature flags | Cargo.toml | `default=["repl"]`, `tui=["ratatui","crossterm"]` |
| TUI app | tui.rs | ratatui split-pane: graph(40%) / output(60%) / status(3) |
| Graph view | show.rs, tree.rs | Auto-updates TUI graph pane |

## Complete Command List (18 + built-in)

| Command | Description |
|---|---|
| `help`, `?` | Show commands |
| `quit`, `q` | Exit |
| `stats` | Database overview |
| `show <id>` | Full node detail |
| `bio` | Vitality leaderboard |
| `why-dead <id>` | Bio score factor breakdown |
| `props <doc_id>` | List document propositions |
| `trace <doc_id>` | Replay ingestion trace |
| `broken` | Integrity check |
| `export trace <id> <f>` | Export trace as JSON |
| `create node <L> [k=v]` | Create node |
| `create edge <s> <T> <d>` | Create edge |
| `set <id> <k> <v>` | Set property |
| `delete node <id>` | Delete node |
| `delete edge <id>` | Delete edge |
| `touch <id>` | Touch node |
| `sig <id> <val>` | Set significance |
| `tree <doc_id>` | Tree view |
| `graph <id> [depth]` | Subgraph view |
| `find label <L>` | Find by label |
| `find prop <k> <v>` | Find by property |
| `find bio <min>` | Find by bio score |
| `top [n]` | Top N by bio score |
| `prune-preview` | Prunable nodes |
| `schema` | Record layouts |
| `hex <id>` | Hex dump |
| `validate` | Strict integrity check |

## Build Commands

```bash
# REPL mode (default)
cargo build -p spider-inspect

# TUI mode
cargo build -p spider-inspect --features tui

# Run
cargo run -p spider-inspect -- ./my_graph/        # REPL
cargo run -p spider-inspect -- --tui ./my_graph/  # TUI (needs --features tui)
```

## Files

```
crates/spider-inspect/
├── Cargo.toml          (feature flags: repl, tui)
├── README.md
├── TASKS.md            ← this file
└── src/
    ├── main.rs              (feature-gated entry points)
    ├── repl.rs              (rustyline REPL)
    ├── tui.rs               (ratatui TUI)
    ├── context.rs           (owns Spider handle)
    ├── output.rs            (formatting helpers + re-exports)
    ├── output_globals.rs    (thread-local sink + StdoutSink)
    ├── sink.rs              (OutputSink trait)
    └── commands/
        ├── mod.rs           (dispatch + help text)
        ├── bio.rs
        ├── broken.rs
        ├── cmd_set.rs
        ├── cmd_sig.rs
        ├── cmd_touch.rs
        ├── create_edge.rs
        ├── create_node.rs
        ├── delete_edge.rs
        ├── delete_node.rs
        ├── export_cmd.rs
        ├── find_cmd.rs
        ├── graph_vis.rs
        ├── prune_preview.rs
        ├── propositions.rs
        ├── schema_cmd.rs
        ├── show.rs
        ├── stats.rs
        ├── top_cmd.rs
        ├── trace.rs
        ├── tree.rs
        ├── validate_cmd.rs
        └── why_dead.rs
```
