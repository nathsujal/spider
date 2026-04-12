//! Document ingestion — wires propositions and entities into the graph.
//!
//! Accepts pre-extracted propositions from the Python/LLM layer and creates
//! the full graph structure: Document → Proposition → Entity nodes with
//! CONTAINS and MENTIONS edges.
//!
//! ## Graph shape after ingestion
//!
//! ```text
//! [Document node] ──CONTAINS──► [Proposition node] ──MENTIONS──► [Entity node]
//!                 ──CONTAINS──► [Proposition node]
//!                 ──CONTAINS──► [Proposition node] ──MENTIONS──► [Entity node]
//! ```
//!
//! ## Design
//!
//! The core never calls an LLM (ADR-003). Propositions and entities are
//! pre-extracted by the Python layer and passed in as borrowed data. The
//! `index()` function owns all node/edge creation and returns the document
//! ID for the caller to track.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::lifecycle::Spider;
use crate::db::nodes::NodeId;
use crate::error::{DbError, SpiderResult};
use crate::schema::dynamic::DynamicStringRecord;
use crate::schema::edge::EdgeTypeId;
use crate::schema::node::{LabelId, Node};
use crate::schema::property::{PropKeyId, PropertyBlock, PropertyRecord};
use crate::schema::token::TokenId;
use crate::store::record::Record;

// --- Ingest request types ---

/// A single factual statement extracted from a document by an LLM.
///
/// The `entities` field contains pre-extracted named entities that appear in
/// this proposition. The ingest function will create/reuse Entity nodes and
/// wire MENTIONS edges between the proposition and each entity.
#[derive(Debug, Clone)]
pub struct Proposition<'a> {
    /// The text of the proposition (e.g. "Mumbai is the financial capital of India").
    pub text: &'a str,
    /// Named entities mentioned in this proposition.
    pub entities: Vec<Entity<'a>>,
}

/// A named entity extracted from proposition text.
///
/// Entities are deduplicated by name — if an Entity node with the same name
/// already exists, it is reused rather than creating a duplicate.
#[derive(Debug, Clone)]
pub struct Entity<'a> {
    /// The entity name (e.g. "Mumbai", "Albert Einstein").
    pub name: &'a str,
    /// The entity type (e.g. "PERSON", "LOCATION", "CONCEPT", "ORGANIZATION").
    pub entity_type: &'a str,
}

/// Request to index a document with its pre-extracted propositions.
///
/// All fields are borrowed — the caller retains ownership of the data.
/// This is intentional: the Python layer owns the strings and spider-core
/// just reads them (ADR-003).
#[derive(Debug)]
pub struct IngestRequest<'a> {
    /// Document title (stored as the "title" property on the Document node).
    pub title: &'a str,
    /// Pre-extracted propositions from the LLM pipeline.
    pub propositions: Vec<Proposition<'a>>,
}

/// Result of a successful ingestion.
#[derive(Debug)]
pub struct IngestResult {
    /// The ID of the created Document node.
    pub document_id: NodeId,
    /// Number of proposition nodes created.
    pub proposition_count: usize,
    /// Number of entity nodes created (excludes deduplicated/reused entities).
    pub entity_count: usize,
    /// Total edges wired (CONTAINS + MENTIONS).
    pub edge_count: usize,
}

// --- Token helpers ---

/// Gets or creates a label token, returning its TokenId.
fn get_or_create_label(spider: &mut Spider, name: &str) -> SpiderResult<TokenId> {
    spider
        .label_tokens
        .get_or_create(name)
        .map_err(DbError::TokenError)
}

/// Gets or creates an edge type token, returning its TokenId.
fn get_or_create_edge_type(spider: &mut Spider, name: &str) -> SpiderResult<TokenId> {
    spider
        .edge_type_tokens
        .get_or_create(name)
        .map_err(DbError::TokenError)
}

/// Gets or creates a property key token, returning its TokenId.
fn get_or_create_prop_key(spider: &mut Spider, name: &str) -> SpiderResult<TokenId> {
    spider
        .prop_key_tokens
        .get_or_create(name)
        .map_err(DbError::TokenError)
}

// --- Node creation helper ---

/// Creates a new node with the given labels and returns its NodeId.
///
/// Allocates a new ID from `metadata.next_node_id`, appends the node to
/// `nodes.db`, and increments the counter.
fn create_node_with_labels(spider: &mut Spider, label_ids: &[LabelId]) -> SpiderResult<NodeId> {
    let node_id = spider.metadata.next_node_id;
    spider.metadata.next_node_id += 1;

    let now = now_unix_secs();
    let node = Node::new(node_id, label_ids, now, None)
        .map_err(DbError::NodeError)?;

    spider.nodes.append(&[node])?;

    NodeId::new(node_id)
}

// --- Property helpers ---

/// Sets a string property on a node.
///
/// For short strings (≤6 bytes), stores inline in the property record.
/// For longer strings, uses the dynamic string store.
fn set_string_property(
    spider: &mut Spider,
    node_id: NodeId,
    key: &str,
    value: &str,
) -> SpiderResult<()> {
    let key_token = get_or_create_prop_key(spider, key)?;
    let key_id = PropKeyId::new(key_token.get())
        .map_err(|_| DbError::TokenError(crate::schema::token::TokenError::InvalidId))?;

    let prop_id = spider.metadata.next_prop_id;
    spider.metadata.next_prop_id += 1;

    // Read the node to get its current first_prop_id.
    let node_idx = node_id.get() - 1;
    let mut node = spider.nodes.get(node_idx)?;
    if node.is_deleted() {
        return Err(DbError::NodeDeleted(node_id.get()));
    }

    // Build the property block.
    let block = if value.len() <= PropertyBlock::MAX_SHORT_STRING {
        PropertyBlock::from_short_string(key_id, value)
            .map_err(DbError::PropertyError)?
    } else {
        // Store in dynamic string store — chain blocks for the string data.
        let data = value.as_bytes();
        let total_len: u32 = data.len().try_into().map_err(|_| {
            DbError::DynamicError(crate::schema::dynamic::DynamicError::LengthOverflow {
                value: data.len() as u32,
                max: crate::schema::dynamic::MAX_LENGTH,
            })
        })?;

        let block_count = data.len().div_ceil(DynamicStringRecord::DATA_SIZE);

        // Allocate IDs for all blocks up front.
        let base_id = spider.metadata.next_string_id;
        spider.metadata.next_string_id += block_count as u32;

        // Build blocks in reverse order (tail first) so next_block is known.
        let mut next_block: u32 = 0;
        let mut head_string_id: u32 = 0;

        for chunk_idx in (0..block_count).rev() {
            let offset = chunk_idx * DynamicStringRecord::DATA_SIZE;
            let end = (offset + DynamicStringRecord::DATA_SIZE).min(data.len());
            let chunk = &data[offset..end];

            let this_block_id = base_id + chunk_idx as u32;

            let record = if chunk_idx == 0 {
                DynamicStringRecord::new_start(chunk, total_len, next_block)
                    .map_err(DbError::DynamicError)?
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

    // Build the property record.
    let mut prop_record = PropertyRecord::new();
    prop_record.blocks[0] = block;
    prop_record.prev_prop_id = 0;
    prop_record.next_prop_id = node.first_prop_id;

    // If there was an existing head property, update its prev pointer.
    if node.first_prop_id != 0 {
        let old_head_idx = node.first_prop_id - 1;
        let mut old_head = spider.properties.get(old_head_idx)?;
        old_head.prev_prop_id = prop_id;
        spider.properties.set(old_head_idx, &old_head)?;
    }

    // Update node's first_prop_id.
    node.first_prop_id = prop_id;
    spider.nodes.set(node_idx, &node)?;

    // Persist the new property record.
    spider.properties.append(&[prop_record])?;

    Ok(())
}

/// Sets an integer property on a node.
fn set_int_property(
    spider: &mut Spider,
    node_id: NodeId,
    key: &str,
    value: i32,
) -> SpiderResult<()> {
    let key_token = get_or_create_prop_key(spider, key)?;
    let key_id = PropKeyId::new(key_token.get())
        .map_err(|_| DbError::TokenError(crate::schema::token::TokenError::InvalidId))?;

    let prop_id = spider.metadata.next_prop_id;
    spider.metadata.next_prop_id += 1;

    let node_idx = node_id.get() - 1;
    let mut node = spider.nodes.get(node_idx)?;
    if node.is_deleted() {
        return Err(DbError::NodeDeleted(node_id.get()));
    }

    let block = PropertyBlock::from_int(key_id, value as i64)
        .map_err(DbError::PropertyError)?;

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

// --- Entity lookup ---

/// Finds an existing Entity node by name, or returns None.
///
/// Walks all ENTITY-labelled nodes and compares the "name" property.
/// This is O(n) but acceptable for ingestion-scale graphs. A proper
/// index will be added later.
fn find_entity_by_name(spider: &mut Spider, name: &str) -> SpiderResult<Option<NodeId>> {
    let entity_tid = match spider.label_tokens.get_id("ENTITY") {
        Some(tid) => tid,
        None => return Ok(None), // No ENTITY nodes exist yet.
    };

    // Scan all nodes looking for one with label ENTITY and matching name.
    let max_id = spider.metadata.next_node_id;

    for nid in 1..max_id {
        let node = match spider.nodes.get(nid - 1) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if node.is_deleted() {
            continue;
        }

        // Check if this node has the ENTITY label.
        let labels = node.labels();
        let has_entity_label = labels.iter().any(|opt| opt.is_some_and(|lid| lid.get() == entity_tid.get()));
        if !has_entity_label {
            continue;
        }

        // Check the "name" property.
        if node_has_property_value(spider, &node, "name", name)? {
            return Ok(Some(NodeId::new(node.id)?));
        }
    }

    Ok(None)
}

/// Checks if a node has a property with the given key and string value.
fn node_has_property_value(
    spider: &mut Spider,
    node: &Node,
    key: &str,
    value: &str,
) -> SpiderResult<bool> {
    let key_token = match spider.prop_key_tokens.get_id(key) {
        Some(tid) => tid,
        None => return Ok(false),
    };

    let mut cursor = node.first_prop_id;
    let max_steps = 10_000;
    let mut steps = 0;

    while cursor != 0 {
        steps += 1;
        if steps > max_steps {
            return Err(DbError::TraversalDepthExceeded { limit: max_steps });
        }

        let prop = spider.properties.get(cursor - 1)?;
        if prop.is_deleted() {
            break;
        }

        let next = prop.next_prop_id;

        // Check each block in the property record.
        for block in &prop.blocks {
            if block.is_empty() {
                continue;
            }
            if block.key_id().map_or(true, |k| k.get() != key_token.get()) {
                continue;
            }

            // Try to read as short string.
            if let Some(s) = block.as_short_string() {
                if s == value {
                    return Ok(true);
                }
            }
        }

        cursor = next;
    }

    Ok(false)
}

// --- Public API ---

/// Indexes a document with pre-extracted propositions and entities.
///
/// Creates the following graph structure:
///
/// 1. A **Document** node (label: `DOCUMENT`) with a `title` property.
/// 2. For each proposition, a **Proposition** node (label: `PROPOSITION`)
///    with a `text` property, connected via a `CONTAINS` edge.
/// 3. For each entity in a proposition, an **Entity** node (label: `ENTITY`)
///    with `name` and `entity_type` properties, connected via a `MENTIONS` edge.
///
/// Entities are deduplicated by name — if an Entity node with the same name
/// already exists, it is reused.
///
/// # Errors
/// - [`DbError::Io`] — if file writes fail
/// - [`DbError::NodeError`] — if node creation fails
/// - [`DbError::EdgeError`] — if edge creation fails
/// - [`DbError::TokenError`] — if token store is full
/// - [`DbError::PropertyError`] — if property storage fails
///
/// # Example
///
/// ```no_run
/// # use std::path::Path;
/// # use spider_core::db::lifecycle::Spider;
/// # use spider_core::db::ingest::{index, IngestRequest, Proposition, Entity};
/// let mut db = Spider::open(Path::new("./test_db")).unwrap();
///
/// let req = IngestRequest {
///     title: "My Document",
///     propositions: vec![
///         Proposition {
///             text: "Mumbai is the financial capital of India",
///             entities: vec![
///                 Entity { name: "Mumbai", entity_type: "LOCATION" },
///                 Entity { name: "India", entity_type: "LOCATION" },
///             ],
///         },
///     ],
/// };
///
/// let result = index(&mut db, &req).unwrap();
/// println!("Document ID: {:?}", result.document_id);
/// println!("Propositions: {}", result.proposition_count);
/// println!("New entities: {}", result.entity_count);
/// ```
pub fn index(spider: &mut Spider, req: &IngestRequest<'_>) -> SpiderResult<IngestResult> {
    // Ensure token types exist.
    let document_label = get_or_create_label(spider, "DOCUMENT")?;
    let proposition_label = get_or_create_label(spider, "PROPOSITION")?;
    let entity_label = get_or_create_label(spider, "ENTITY")?;
    let contains_edge = get_or_create_edge_type(spider, "CONTAINS")?;
    let mentions_edge = get_or_create_edge_type(spider, "MENTIONS")?;

    let document_label_id = LabelId::new(document_label.get())
        .map_err(|_| DbError::TokenError(crate::schema::token::TokenError::InvalidId))?;
    let proposition_label_id = LabelId::new(proposition_label.get())
        .map_err(|_| DbError::TokenError(crate::schema::token::TokenError::InvalidId))?;
    let entity_label_id = LabelId::new(entity_label.get())
        .map_err(|_| DbError::TokenError(crate::schema::token::TokenError::InvalidId))?;
    let contains_edge_id = EdgeTypeId::new(contains_edge.get())
        .map_err(|_| DbError::TokenError(crate::schema::token::TokenError::InvalidId))?;
    let mentions_edge_id = EdgeTypeId::new(mentions_edge.get())
        .map_err(|_| DbError::TokenError(crate::schema::token::TokenError::InvalidId))?;

    // 1. Create Document node.
    let doc_id = create_node_with_labels(spider, &[document_label_id])?;

    // Set title property on document.
    set_string_property(spider, doc_id, "title", req.title)?;

    // 2. Create Proposition nodes and wire CONTAINS edges.
    let mut new_entity_count = 0usize;
    let mut total_edges = 0usize;

    for (chunk_idx, prop) in req.propositions.iter().enumerate() {
        // Create Proposition node.
        let prop_id = create_node_with_labels(spider, &[proposition_label_id])?;

        // Set text property.
        set_string_property(spider, prop_id, "text", prop.text)?;

        // Set chunk_index property.
        set_int_property(spider, prop_id, "chunk_index", chunk_idx as i32)?;

        // Wire CONTAINS edge: Document → Proposition.
        spider.edge_ops().create(doc_id, prop_id, contains_edge_id)?;
        total_edges += 1;

        // 3. Create/reuse Entity nodes and wire MENTIONS edges.
        for entity in &prop.entities {
            // Try to find existing entity by name.
            let ent_id = match find_entity_by_name(spider, entity.name)? {
                Some(existing_id) => existing_id,
                None => {
                    // Create new Entity node.
                    let new_id = create_node_with_labels(spider, &[entity_label_id])?;
                    new_entity_count += 1;

                    // Set name and entity_type properties.
                    set_string_property(spider, new_id, "name", entity.name)?;
                    set_string_property(spider, new_id, "entity_type", entity.entity_type)?;

                    new_id
                }
            };

            // Wire MENTIONS edge: Proposition → Entity.
            spider.edge_ops().create(prop_id, ent_id, mentions_edge_id)?;
            total_edges += 1;
        }
    }

    Ok(IngestResult {
        document_id: doc_id,
        proposition_count: req.propositions.len(),
        entity_count: new_entity_count,
        edge_count: total_edges,
    })
}

/// Returns current Unix timestamp in seconds.
fn now_unix_secs() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as u32
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, Spider) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_ingest_db");
        let db = Spider::open(&db_path).unwrap();
        (dir, db)
    }

    #[test]
    fn ingest_empty_propositions() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Empty Doc",
            propositions: vec![],
        };

        let result = index(&mut db, &req).unwrap();
        assert_eq!(result.proposition_count, 0);
        assert_eq!(result.entity_count, 0);
        assert_eq!(result.edge_count, 0);
        assert!(result.document_id.get() >= 1);
    }

    #[test]
    fn ingest_single_proposition_no_entities() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test Document",
            propositions: vec![
                Proposition {
                    text: "The sky is blue",
                    entities: vec![],
                },
            ],
        };

        let result = index(&mut db, &req).unwrap();
        assert_eq!(result.proposition_count, 1);
        assert_eq!(result.entity_count, 0);
        assert_eq!(result.edge_count, 1); // 1 CONTAINS edge
    }

    #[test]
    fn ingest_short_string_property() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Hi",
            propositions: vec![],
        };

        let result = index(&mut db, &req).unwrap();
        assert_eq!(result.proposition_count, 0);
    }

    #[test]
    fn ingest_long_proposition_text() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "This is a very long proposition that exceeds six bytes",
                    entities: vec![],
                },
            ],
        };

        let result = index(&mut db, &req).unwrap();
        assert_eq!(result.proposition_count, 1);
        assert_eq!(result.entity_count, 0);
        assert_eq!(result.edge_count, 1);
    }

    #[test]
    fn ingest_entity_with_long_type() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Short",
                    entities: vec![
                        Entity { name: "X", entity_type: "LOCATION" },
                    ],
                },
            ],
        };

        let result = index(&mut db, &req).unwrap();
        assert_eq!(result.proposition_count, 1);
        assert_eq!(result.entity_count, 1);
        assert_eq!(result.edge_count, 2);
    }

    #[test]
    fn ingest_two_entities_with_long_types() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Mumbai is in India",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                        Entity { name: "India", entity_type: "LOCATION" },
                    ],
                },
            ],
        };

        let result = index(&mut db, &req).unwrap();
        assert_eq!(result.proposition_count, 1);
        assert_eq!(result.entity_count, 2);
        assert_eq!(result.edge_count, 3);
    }

    #[test]
    fn ingest_proposition_with_entities() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Geography Facts",
            propositions: vec![
                Proposition {
                    text: "Mumbai is the financial capital of India",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                        Entity { name: "India", entity_type: "LOCATION" },
                    ],
                },
            ],
        };

        let result = index(&mut db, &req).unwrap();
        assert_eq!(result.proposition_count, 1);
        assert_eq!(result.entity_count, 2);
        assert_eq!(result.edge_count, 3); // 1 CONTAINS + 2 MENTIONS
    }

    #[test]
    fn ingest_multiple_propositions() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Multiple Facts",
            propositions: vec![
                Proposition {
                    text: "Paris is the capital of France",
                    entities: vec![
                        Entity { name: "Paris", entity_type: "LOCATION" },
                        Entity { name: "France", entity_type: "LOCATION" },
                    ],
                },
                Proposition {
                    text: "Einstein developed the theory of relativity",
                    entities: vec![
                        Entity { name: "Einstein", entity_type: "PERSON" },
                        Entity { name: "relativity", entity_type: "CONCEPT" },
                    ],
                },
            ],
        };

        let result = index(&mut db, &req).unwrap();
        assert_eq!(result.proposition_count, 2);
        assert_eq!(result.entity_count, 4);
        assert_eq!(result.edge_count, 6); // 2 CONTAINS + 4 MENTIONS
    }

    #[test]
    fn ingest_deduplicates_entities_by_name() {
        let (_dir, mut db) = setup();

        // First ingestion with "Mumbai".
        let req1 = IngestRequest {
            title: "Doc 1",
            propositions: vec![
                Proposition {
                    text: "Mumbai is a big city",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                    ],
                },
            ],
        };
        let result1 = index(&mut db, &req1).unwrap();
        assert_eq!(result1.entity_count, 1);

        // Second ingestion with same "Mumbai" — should be deduplicated.
        let req2 = IngestRequest {
            title: "Doc 2",
            propositions: vec![
                Proposition {
                    text: "Mumbai has great food",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                    ],
                },
            ],
        };
        let result2 = index(&mut db, &req2).unwrap();
        assert_eq!(result2.entity_count, 0); // No new entities.
        assert_eq!(result2.proposition_count, 1);
        assert_eq!(result2.edge_count, 2); // 1 CONTAINS + 1 MENTIONS (to existing entity)
    }

    #[test]
    fn ingest_creates_correct_labels() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Labels Test",
            propositions: vec![
                Proposition {
                    text: "Test proposition",
                    entities: vec![
                        Entity { name: "TestEntity", entity_type: "CONCEPT" },
                    ],
                },
            ],
        };

        index(&mut db, &req).unwrap();

        // Verify token stores have the expected labels.
        assert!(db.label_tokens.contains("DOCUMENT"));
        assert!(db.label_tokens.contains("PROPOSITION"));
        assert!(db.label_tokens.contains("ENTITY"));
        assert!(db.edge_type_tokens.contains("CONTAINS"));
        assert!(db.edge_type_tokens.contains("MENTIONS"));
    }

    #[test]
    fn ingest_stores_document_title() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "My Important Document",
            propositions: vec![],
        };

        let result = index(&mut db, &req).unwrap();

        // Verify the document node has the title property.
        let doc_node = db.nodes.get(result.document_id.get() - 1).unwrap();
        assert!(!doc_node.is_deleted());
        assert!(doc_node.has_properties());
    }

    #[test]
    fn ingest_advances_metadata_counters() {
        let (_dir, mut db) = setup();

        let initial_node_id = db.metadata.next_node_id;
        let initial_prop_id = db.metadata.next_prop_id;
        let initial_rel_id = db.metadata.next_rel_id;

        let req = IngestRequest {
            title: "Counter Test",
            propositions: vec![
                Proposition {
                    text: "One proposition with one entity",
                    entities: vec![
                        Entity { name: "TestEnt", entity_type: "CONCEPT" },
                    ],
                },
            ],
        };

        index(&mut db, &req).unwrap();

        // 1 Document + 1 Proposition + 1 Entity = 3 nodes.
        assert_eq!(
            db.metadata.next_node_id,
            initial_node_id + 3
        );

        // 1 CONTAINS + 1 MENTIONS = 2 edges.
        assert_eq!(
            db.metadata.next_rel_id,
            initial_rel_id + 2
        );

        // 1 title + 1 text + 1 chunk_index + 1 name + 1 entity_type = 5 properties.
        assert_eq!(
            db.metadata.next_prop_id,
            initial_prop_id + 5
        );
    }

    #[test]
    fn ingest_short_and_long_strings() {
        let (_dir, mut db) = setup();

        // Test with a very long proposition text.
        let long_text = "This is a very long proposition text. ".repeat(50);

        let req = IngestRequest {
            title: "Long Text Test",
            propositions: vec![
                Proposition {
                    text: &long_text,
                    entities: vec![],
                },
            ],
        };

        let result = index(&mut db, &req).unwrap();
        assert_eq!(result.proposition_count, 1);
    }
}
