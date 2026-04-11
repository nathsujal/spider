# spider-inspect — Implementation Plan

> Rust REPL for debugging and inspecting Spider databases.
> Bypasses the daemon entirely — reads `.db` files directly via spider-core.

---

## Architecture

```
spider-inspect (binary)
    │
    ├── rustyline        → interactive REPL input
    ├── clap             → CLI arg parsing (--db-path)
    ├── comfy-table      → formatted tabular output
    ├── colored          → terminal colors
    └── spider-core      → direct database access
```

### Dependency Flow

```
user types "stats"
    │
    ▼
repl::run()  ── parse command
    │
    ▼
commands::stats(db)  ── calls spider-core APIs
    │
    ├── db.metadata          → ID counters, bio params
    ├── db.nodes.iter()      → count live nodes, top labels
    ├── db.edges.iter()      → count relationships
    └── print table
```

---

## Tasks (in build order)

### 1. REPL skeleton and CLI

**Goal:** `spider-inspect ./my_graph/` opens a database and enters an interactive prompt.

**Work:**
- `src/main.rs` — clap app with optional `--db-path` positional arg
- `src/repl.rs` — rustyline loop: `read_line()` → `parse()` → `dispatch()` → `print result`
- `src/context.rs` — `Context` struct wrapping `Spider` handle, provides `bio_params` from `db.metadata`
- Commands: `help`, `quit` / `exit` (built-in, not spider-core calls)

**Key design:**
- `Context` owns `Spider` so the database stays open across commands
- No `&mut Spider` juggling — `Context` has one `db: Spider` field and exposes `&mut self` methods
- Error handling: all commands return `anyhow::Result<()>`, errors print in red

**Acceptance:**
- `spider-inspect` without args → opens default database (via `Spider::open_default()`)
- `spider-inspect ./my_graph/` → opens specified database
- Prompt shows `spider> ` (or `spider (./my_graph) > ` with path)
- `help` prints command list
- `quit` exits cleanly

**Estimated:** ~150 lines

---

### 2. Output helpers

**Goal:** Shared formatting so every command doesn't repeat table/color boilerplate.

**Work:**
- `src/output.rs` — helper functions:
  - `table(headers, rows)` → builds `comfy-table` with consistent styling
  - `print_error(e)` → red text
  - `print_ok(msg)` → green text
  - `print_label(token_id, tokens)` → resolves token ID to label name
  - `print_edge_type(token_id, tokens)` → resolves edge type token to name
  - `format_prop_value(PropertyValue)` → human-readable string
  - `format_timestamp(unix_secs)` → readable date string

**Acceptance:**
- Tables have borders, headers bold, rows aligned
- Errors always red, success always green
- Label and edge type names resolved from token stores automatically

**Estimated:** ~100 lines

---

### 3. `stats` — Database overview

**Goal:** `stats` → node count, edge count, property count, blob size, top labels.

**spider-core APIs used:**
- `db.metadata` → counters (subtract 1 for current next_* values = count of live-ish IDs)
- Iterate `db.nodes` → count live (non-tombstone) nodes, tally labels
- Iterate `db.edges` → count live edges
- `db.prop_key_tokens.len()` → unique property keys
- `db.label_tokens.iter()` → label names + frequency

**Output:**
```
Database: ./my_graph/
─────────────────────────────────────
Nodes:          1,247
Edges:          3,891
Properties:     4,502
Dynamic strings: 892
  Labels:       3
    DOCUMENT    421
    PROPOSITION 410
    ENTITY      416
Bio params:     w_sig=3.0  w_freq=2.0  gravity=1.5
```

**Acceptance:**
- All counts accurate (skips tombstones)
- Labels shown with counts, sorted descending
- Bio params from metadata displayed

**Estimated:** ~80 lines

---

### 4. `show <node_id>` — Full node detail

**Goal:** `show 42` → type, labels, properties, edges, bio score, tier.

**spider-core APIs used:**
- `db.nodes.get(id - 1)` → raw `Node` record
- `property::list_all(db, NodeId::new(id)?)` → all properties
- `traverse::get_relationships(db, NodeId::new(id)?, Direction::Both)` → all edges
- `bio::calculate(node)` → vitality score
- `BioTier::from_score(score)` → tier classification
- Resolve labels from `db.label_tokens`
- Resolve edge types from `db.edge_type_tokens`

**Output:**
```
Node #42
─────────────────────────────────────
Labels:         DOCUMENT
Created:        2026-04-10 14:32:01
Last accessed:  2026-04-11 09:15:44
Access count:   12
Significance:   128 (0.50)

Bio Score:      18.42  [Warm]

Properties:
┌───────────────┬──────────────────────────────────┐
│ Key           │ Value                            │
├───────────────┼──────────────────────────────────┤
│ title         │ My Document                      │
│ ingested_at   │ 2026-04-10T14:32:01Z             │
└───────────────┴──────────────────────────────────┘

Edges (3 outgoing, 0 incoming):
┌──────┬──────────┬──────────┬──────────┐
│ ID   │ Type     │ Target   │ Labels   │
├──────┼──────────┼──────────┼──────────┤
│ 15   │ CONTAINS │ #43      │ PROPOSITION │
│ 16   │ CONTAINS │ #44      │ PROPOSITION │
│ 17   │ CONTAINS │ #45      │ PROPOSITION │
└──────┴──────────┴──────────┴──────────┘
```

**Acceptance:**
- Shows all node fields, resolved label names
- Properties table with key/value pairs
- Edges table with target node labels
- Bio score + tier displayed
- Invalid node ID → helpful error

**Estimated:** ~120 lines

---

### 5. `bio` — Vitality leaderboard

**Goal:** `bio` → all nodes ranked by bio score, descending.

**spider-core APIs used:**
- Iterate all nodes in `db.nodes` (skip tombstones)
- `bio::calculate(node)` for each
- `BioTier::from_score(score)` for tier
- Resolve labels from `db.label_tokens`
- Sort by score descending

**Output:**
```
Vitality Leaderboard
──────────────────────────────────────────────────────────────
┌──────┬────────────┬───────┬───────────┬────────┬───────────┐
│ Node │ Labels     │ Score │ Tier      │ Access │ Signif.   │
├──────┼────────────┼───────┼───────────┼────────┼───────────┤
│ 42   │ DOCUMENT   │ 24.18 │ Warm      │     12 │ 0.50      │
│ 43   │ PROPOSITION│ 22.05 │ Warm      │      0 │ 0.50      │
│  7   │ ENTITY     │  0.00 │ Pruned    │      0 │ 0.10      │
└──────┴────────────┴───────┴───────────┴────────┴───────────┘

Total: 1,247 nodes  |  Hot: 12  |  Warm: 410  |  Cold: 409  |  Pruned: 416
```

**Acceptance:**
- All live nodes listed, sorted by score descending
- Tier breakdown in footer
- Pagination or limit flag for large databases (future)

**Estimated:** ~80 lines

---

### 6. `why-dead <node_id>` — Explain low bio score

**Goal:** `why-dead 7` → shows each factor contributing to the node's score.

**spider-core APIs used:**
- `db.nodes.get(id - 1)` → node record
- `bio::calculate(node)` → total score
- Manual factor breakdown:
  - Significance contribution: `(S × Ws × 100)`
  - Frequency contribution: `ln(1 + access_count) × 10 × Wf`
  - Time decay divisor: `(days + 2)^G`
- Compare against thresholds for each tier

**Output:**
```
Why Node #7 has a low bio score
─────────────────────────────────────
Bio Score:      0.00  [Pruned]

Factor breakdown:
  Significance:  0.10 × 3.0 × 100 =  30.00
  Frequency:     ln(1 + 0) × 10 × 2.0 =   0.00
  ──────────────────────────────────────
  Numerator:                                30.00
  Denominator:  (120.5 days + 2)^1.5     1,286.30
  ──────────────────────────────────────
  Final score:                              0.02

Verdict: This node was created 120 days ago, has never been accessed,
         and has very low significance (0.10 / 1.00).

To improve this node's score:
  → Increase significance (currently 25/255)
  → Access the node more (currently 0 accesses)
  → Reduce time since last access (currently 120 days)
```

**Acceptance:**
- Shows each factor with intermediate values
- Human-readable verdict
- Actionable suggestions for improvement

**Estimated:** ~100 lines

---

### 7. `propositions <doc_id>` — Document propositions

**Goal:** `propositions 42` → list all propositions connected to a document.

**spider-core APIs used:**
- Verify node 42 has DOCUMENT label
- `traverse::get_relationships(db, NodeId::new(42)?, Direction::Outgoing)` → CONTAINS edges
- Filter edges by type "CONTAINS" (resolve from `db.edge_type_tokens`)
- For each proposition target: `property::get_string(db, prop_id, "text")`

**Output:**
```
Document #42: "My Document"
Propositions: 3
─────────────────────────────────────
┌──────┬──────────────────────────────────────────────────┐
│ Node │ Proposition Text                                 │
├──────┼──────────────────────────────────────────────────┤
│   43 │ Mumbai is the financial capital of India         │
│   44 │ India has a population of over 1.4 billion       │
│   45 │ The Indian economy grew 7.2% in 2025             │
└──────┴──────────────────────────────────────────────────┘
```

**Acceptance:**
- Only shows CONTAINS edges (not other edge types)
- Proposition text from property lookup
- Non-document node ID → helpful error

**Estimated:** ~80 lines

---

### 8. `trace <doc_id>` — Ingestion trace

**Goal:** `trace 42` → replay what happened during ingestion.

**spider-core APIs used:**
- Same as `propositions` for finding propositions
- For each proposition: `traverse::get_relationships(db, prop_id, Direction::Outgoing)` → MENTIONS edges
- For each entity: `property::list_all(db, entity_id)` → name, entity_type
- Count totals

**Output:**
```
Ingestion Trace for Document #42: "My Document"
─────────────────────────────────────
Document node:  #42  [DOCUMENT]
Propositions:   3
Entities:       4  (2 unique, 2 reused)
Edges:          7  (3 CONTAINS + 4 MENTIONS)

Proposition #43: "Mumbai is the financial capital of India"
  → MENTIONS → Entity #50: "Mumbai" [LOCATION]
  → MENTIONS → Entity #51: "India" [LOCATION]

Proposition #44: "India has a population of over 1.4 billion"
  → MENTIONS → Entity #51: "India" [LOCATION]  ← reused
  → MENTIONS → Entity #52: "population" [CONCEPT]

Proposition #45: "The Indian economy grew 7.2% in 2025"
  → MENTIONS → Entity #51: "India" [LOCATION]  ← reused
```

**Acceptance:**
- Shows full graph: document → propositions → entities
- Reused entities marked
- Edge type names resolved

**Estimated:** ~100 lines

---

### 9. `broken` — Integrity check

**Goal:** `broken` → find orphaned nodes, missing edges, dangling properties.

**spider-core APIs used:**
- Iterate all nodes → check if `first_edge_id` points to a valid edge
- Iterate all edges → check if source_id and target_id point to live nodes
- Iterate all property chains → check if `prev_prop_id` / `next_prop_id` form valid chains

**Output:**
```
Integrity Check
─────────────────────────────────────
Nodes scanned:    1,247
Edges scanned:    3,891
Properties scanned: 4,502

Issues found: 0  ✓

(All references are valid)
```

Or with issues:
```
Issues found: 3

⚠ Edge #892: source node #123 is deleted
⚠ Edge #892: target node #456 does not exist
⚠ Node #789: first_edge_id=999 but edge #999 is deleted
```

**Acceptance:**
- Checks edge→node references (both source and target)
- Checks property chain integrity
- Clean report when no issues found

**Estimated:** ~120 lines

---

### 10. `export trace <doc_id> <file>` — Export as JSON

**Goal:** `export trace 42 report.json` → save ingestion trace as JSON.

**spider-core APIs used:**
- Same as `trace` command for data gathering
- `serde_json` for serialization

**Output file:**
```json
{
  "document_id": 42,
  "title": "My Document",
  "propositions": [
    {
      "node_id": 43,
      "text": "Mumbai is the financial capital of India",
      "entities": [
        { "node_id": 50, "name": "Mumbai", "type": "LOCATION" },
        { "node_id": 51, "name": "India", "type": "LOCATION" }
      ]
    }
  ],
  "stats": {
    "total_propositions": 3,
    "total_entities": 4,
    "unique_entities": 2,
    "total_edges": 7
  }
}
```

**Acceptance:**
- Valid JSON output
- Includes all trace data
- CSV export option for leaderboard (`export bio stats.csv`)

**Estimated:** ~60 lines

---

## File Structure

```
crates/spider-inspect/
├── Cargo.toml
├── TASKS.md                 ← this file
└── src/
    ├── main.rs              → CLI args, entry point
    ├── repl.rs              → rustyline loop, command dispatch
    ├── context.rs           → owns Spider, provides db access
    ├── output.rs            → table formatting, colors, helpers
    └── commands/
        ├── mod.rs           → command enum, dispatch
        ├── help.rs          → built-in help text
        ├── stats.rs         → database overview
        ├── show.rs          → single node detail
        ├── bio.rs           → vitality leaderboard
        ├── why_dead.rs      → bio score factor breakdown
        ├── propositions.rs  → document propositions
        ├── trace.rs         → ingestion trace
        ├── broken.rs        → integrity check
        └── export.rs        → JSON/CSV export
```

---

## Dependencies (all in Cargo.toml already)

| Crate | Purpose |
|---|---|
| `spider-core` | Database engine — all data comes from here |
| `clap` | CLI argument parsing (`--db-path`) |
| `rustyline` | Interactive REPL with history |
| `comfy-table` | Formatted tabular output |
| `colored` | Terminal colors (red errors, green success) |
| `serde` + `serde_json` | Export serialization |
| `csv` | CSV export |
| `anyhow` | Error handling (`Result<()>`) |
| `directories` | Platform default DB path |

---

## Build Verification

After all tasks:
```bash
cargo build -p spider-inspect
cargo test -p spider-inspect
cargo clippy -p spider-inspect -- -D warnings
```

---

## Implementation Order

Build tasks in this order: **1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10**

Each task is independently testable. Task 1 (REPL skeleton) is the critical path — nothing else works without it. Tasks 3-8 are the core commands. Tasks 9-10 are polish.
