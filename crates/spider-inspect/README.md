# spider-inspect

Interactive REPL for debugging and inspecting Spider databases.

Reads `.db` files directly — **no daemon required**. Always works, even when the daemon is broken (ADR-002).

## Quick Start

```bash
# Open the default database (~/.local/share/spider/default/)
cargo run -p spider-inspect

# Open a specific database
cargo run -p spider-inspect ./my_graph/
```

You'll see the prompt:

```
Spider Inspect — /path/to/my_graph
Type 'help' for commands.

spider (my_graph) >
```

## Commands

### Data Inspection

| Command | Description |
|---|---|
| `stats` | Database overview: node/edge/property counts, label frequency, bio params |
| `show <id>` | Full node detail: labels, properties, edges, bio score with tier |
| `bio` | Vitality leaderboard: all nodes ranked by bio score, sorted descending |
| `props <doc_id>` | List all propositions connected to a document |
| `trace <doc_id>` | Full ingestion trace: document → propositions → entities with edge details |

### Analysis

| Command | Description |
|---|---|
| `why-dead <id>` | Factor breakdown explaining a node's bio score (significance, frequency, decay) |
| `broken` | Integrity check: dangling edges, orphaned nodes, broken property chains |

### Export

| Command | Description |
|---|---|
| `export trace <id> <file>` | Save ingestion trace as pretty-printed JSON |

### General

| Command | Description |
|---|---|
| `help`, `?` | Show available commands |
| `quit`, `q` | Exit the REPL |

## Examples

### Check database health

```
spider (my_graph) > stats

Database: /path/to/my_graph
──────────────────────────────────────────────────
╭─────────────────┬──────────────────────────────╮
│ Nodes           ┆ 1247                         │
│ Edges           ┆ 3891                         │
│ Properties      ┆ 4502                         │
│ Dynamic strings ┆ 892                          │
│ Property keys   ┆ 5                            │
│ Bio params      ┆ w_sig=3.0  w_freq=2.0  ...   │
╰─────────────────┴──────────────────────────────╯

  Labels:
    DOCUMENT        421
    PROPOSITION     410
    ENTITY          416
```

### Inspect a specific node

```
spider (my_graph) > show 42

Node #42
─────────────────────────────────────
╭───────────────┬──────────────────────────────╮
│ Labels        ┆ DOCUMENT                     │
│ Created       ┆ 2026-04-10 14:32:01          │
│ Last accessed ┆ 2026-04-11 09:15:44          │
│ Access count  ┆ 12                           │
│ Significance  ┆ 0.50 (128/255)               │
│ Bio Score     ┆ 18.42  [Warm]                │
╰───────────────┴──────────────────────────────╯

Properties:
╭───────────────┬──────────────────────────────╮
│ Key           ┆ Value                        │
├───────────────┼──────────────────────────────┤
│ title         ┆ My Document                  │
╰───────────────┴──────────────────────────────╯

Edges (3 outgoing, 0 incoming):
╭──────┬──────────┬──────────┬───────────────╮
│ ID   ┆ Type     ┆ Node     ┆ Labels        │
├──────┼──────────┼──────────┼───────────────┤
│ 15   ┆ CONTAINS ┆ #43      ┆ PROPOSITION   │
│ 16   ┆ CONTAINS ┆ #44      ┆ PROPOSITION   │
│ 17   ┆ CONTAINS ┆ #45      ┆ PROPOSITION   │
╰──────┴──────────┴──────────┴───────────────╯
```

### See the vitality leaderboard

```
spider (my_graph) > bio

Vitality Leaderboard
──────────────────────────────────────────────────────────────
╭──────┬────────────┬───────┬─────────┬────────┬──────────╮
│ Node ┆ Labels     ┆ Score ┆ Tier    ┆ Access ┆ Signif.  │
├──────┼────────────┼───────┼─────────┼────────┼──────────┤
│ 42   ┆ DOCUMENT   ┆ 24.18 ┆ Warm    ┆     12 ┆ 0.50     │
│ 43   ┆ PROPOSITION┆ 22.05 ┆ Warm    ┆      0 ┆ 0.50     │
│ 7    ┆ ENTITY     ┆  0.00 ┆ Pruned  ┆      0 ┆ 0.10     │
╰──────┴────────────┴───────┴─────────┴────────┴──────────╯

  Total: 1247 nodes  |  Hot: 12  |  Warm: 410  |  Cold: 409  |  Pruned: 416
```

### Explain why a node is dead

```
spider (my_graph) > why-dead 7

Why Node #7 has a low bio score
─────────────────────────────────────
Bio Score:      0.02  [Pruned]

Factor breakdown:
  Significance:  0.10 × 3.0 × 100 =   30.00
  Frequency:     ln(1 + 0) × 10 × 2.0 =    0.00
  ──────────────────────────────────────
  Numerator:                                30.00
  Denominator:  (120.5 days + 2)^1.5     1286.30
  ──────────────────────────────────────
  Final score:                               0.02

Verdict: This node was not accessed in 120 days, never been accessed, and has very low significance (0.10 / 1.00)

To improve this node's score:
  → Increase significance (currently 25/255)
  → Access the node more (currently 0 accesses)
  → Reduce time since last access (currently 120 days)
```

### Check database integrity

```
spider (my_graph) > broken

Integrity Check
─────────────────────────────────────
Nodes scanned:    1247
  (deleted: 3)
Edges scanned:    3891
Properties:       4502 IDs allocated

Issues found: 0  ✓

(All references are valid)
```

### Export a trace as JSON

```
spider (my_graph) > export trace 42 report.json

Exported 3 propositions to report.json
```

## Database Location

| Mode | Path |
|---|---|
| Default (no args) | Platform-specific data directory + `/spider/default/` |
| Explicit arg | The path you provide |

Platform defaults:

| Platform | Default Path |
|---|---|
| Linux | `~/.local/share/spider/default/` |
| macOS | `~/Library/Application Support/spider/default/` |
| Windows | `%APPDATA%\spider\default\` |

## Building

```bash
cargo build -p spider-inspect --release
```

The binary is available at `target/release/spider-inspect`.

## Architecture

spider-inspect bypasses the daemon entirely and reads the `.db` files directly via spider-core. This is by design (ADR-002): when the daemon is broken, inspect still works.

```
spider-inspect ────reads────► spider-core ────owns────► .db files
                                                          meta.db
                                                          nodes.db
                                                          edges.db
                                                          properties.db
                                                          strings.db
                                                          blobs/
```

See [spider-core README](../spider-core/README.md) for database file format details.
