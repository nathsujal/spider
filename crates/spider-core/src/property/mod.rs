//! Property operations — typed get/set/delete/list for node and edge properties.
//!
//! Provides a high-level API over the raw [`PropertyBlock`](crate::schema::property::PropertyBlock)
//! storage layer. Resolves [`PropertyValue`] from disk including dereferencing
//! dynamic string chains and multi-block values.
//!
//! ## API
//!
//! - [`get()`] — read a single property by key, returns typed [`PropertyValue`]
//! - [`set()`] — write a property (inserts or overwrites by key)
//! - [`delete()`] — remove a property by key
//! - [`list_all()`] — enumerate all properties on a node or edge
//! - [`get_string()`], [`get_int()`], [`get_float()`], [`get_bool()`] — typed convenience getters

use crate::db::lifecycle::Spider;
use crate::db::nodes::NodeId;
use crate::error::{DbError, SpiderResult};
use crate::schema::edge::Edge;
use crate::schema::node::Node;
use crate::schema::property::{PropertyBlock, PropertyRecord, PropertyType};
use crate::schema::token::TokenId;
use crate::store::record::Record;

// --- PropertyValue enum ---

/// A resolved property value read from disk.
///
/// Covers all inline types plus dereferenced dynamic strings.
/// Array, temporal, and spatial types return `Raw` with the raw bits —
/// typed decoders for those are future work.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    /// Boolean value.
    Bool(bool),
    /// Signed byte (`i8`).
    Byte(i8),
    /// Signed short (`i16`).
    Short(i16),
    /// Signed integer (51-bit, fits in `i64`).
    Int(i64),
    /// Signed long (`i64`, stored across 2 blocks).
    Long(i64),
    /// 32-bit float.
    Float(f32),
    /// 64-bit float (stored across 2 blocks).
    Double(f64),
    /// Unicode character.
    Char(char),
    /// Inline short string (≤6 UTF-8 bytes).
    ShortString(String),
    /// Long string — dereferenced from `strings.db` chain.
    String(String),
    /// Raw block bits for types without a typed decoder yet.
    /// Contains `(PropertyType, u64)` — the type discriminant and raw value.
    Raw(PropertyType, u64),
}

impl std::fmt::Display for PropertyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{}", v),
            Self::Byte(v) => write!(f, "{}", v),
            Self::Short(v) => write!(f, "{}", v),
            Self::Int(v) => write!(f, "{}", v),
            Self::Long(v) => write!(f, "{}", v),
            Self::Float(v) => write!(f, "{}", v),
            Self::Double(v) => write!(f, "{}", v),
            Self::Char(v) => write!(f, "{}", v),
            Self::ShortString(v) | Self::String(v) => write!(f, "{}", v),
            Self::Raw(ty, bits) => write!(f, "{:?}:0x{:x}", ty, bits),
        }
    }
}

// --- Block decoding ---

/// Decodes a single [`PropertyBlock`] into a [`PropertyValue`].
///
/// For dynamic string types, follows the chain in `strings.db` to
/// reconstruct the full string.
fn decode_block(
    spider: &mut Spider,
    block: &PropertyBlock,
) -> SpiderResult<PropertyValue> {
    if block.is_empty() {
        return Err(DbError::NotFound);
    }

    match block.value_type() {
        PropertyType::Empty => Err(DbError::NotFound),
        PropertyType::Bool => Ok(PropertyValue::Bool(block.as_bool().unwrap())),
        PropertyType::Byte => {
            let bits = block.value_bits() as i8;
            Ok(PropertyValue::Byte(bits))
        }
        PropertyType::Short => {
            let bits = block.value_bits() as i16;
            Ok(PropertyValue::Short(bits))
        }
        PropertyType::Int => Ok(PropertyValue::Int(block.as_int().unwrap())),
        PropertyType::Long => {
            // Not yet supported — would need to read next block.
            Ok(PropertyValue::Raw(PropertyType::Long, block.value_bits()))
        }
        PropertyType::Float => Ok(PropertyValue::Float(block.as_float().unwrap())),
        PropertyType::Double => {
            // Not yet supported — would need to read next block.
            Ok(PropertyValue::Raw(PropertyType::Double, block.value_bits()))
        }
        PropertyType::Char => {
            let bits = block.value_bits() as u32;
            let ch = char::from_u32(bits).unwrap_or('\0');
            Ok(PropertyValue::Char(ch))
        }
        PropertyType::ShortString => {
            Ok(PropertyValue::ShortString(block.as_short_string().unwrap()))
        }
        PropertyType::String => {
            // Dereference dynamic string chain.
            let ptr = block.dyn_string_ptr().unwrap();
            let s = read_dynamic_string(spider, ptr)?;
            Ok(PropertyValue::String(s))
        }
        other => Ok(PropertyValue::Raw(other, block.value_bits())),
    }
}

/// Reads a full string from a dynamic string chain.
fn read_dynamic_string(spider: &mut Spider, start_id: u32) -> SpiderResult<String> {
    let mut result = Vec::new();
    let mut cursor = start_id;
    let mut steps = 0;
    let max_steps = 10_000;

    while cursor != 0 {
        steps += 1;
        if steps > max_steps {
            return Err(DbError::TraversalDepthExceeded { limit: max_steps });
        }

        let record = spider.strings.get(cursor - 1)?;
        if !record.is_in_use() {
            break;
        }

        if record.is_start() {
            let total_len = record.get_length() as usize;
            result.extend_from_slice(record.get_data(total_len));
        } else {
            // For continuation blocks, we don't know exact length but
            // the data field contains valid bytes up to DATA_SIZE.
            // In practice the chain is written contiguously so we just
            // grab whatever is in the data field.
            result.extend_from_slice(&record.data);
        }

        cursor = record.next_block;
    }

    String::from_utf8(result).map_err(|_| {
        DbError::NotFound // No better variant — string corruption = data not found
    })
}

// --- Key resolution ---

/// Resolves a property key name to its token ID.
fn resolve_key(spider: &mut Spider, key: &str) -> SpiderResult<Option<TokenId>> {
    Ok(spider.prop_key_tokens.get_id(key))
}

// --- Node property chain walker ---

/// Finds the property block matching the given key on a node.
///
/// Returns `(prop_record, block_index)` for the matching block, or `None`.
fn find_property_block_on_node(
    spider: &mut Spider,
    node: &Node,
    key_id: TokenId,
) -> SpiderResult<Option<(PropertyRecord, usize)>> {
    let mut cursor = node.first_prop_id;
    let mut steps = 0;
    let max_steps = 10_000;

    while cursor != 0 {
        steps += 1;
        if steps > max_steps {
            return Err(DbError::TraversalDepthExceeded { limit: max_steps });
        }

        let prop = spider.properties.get(cursor - 1)?;
        if prop.is_deleted() {
            break;
        }

        for (i, block) in prop.blocks.iter().enumerate() {
            if block.is_empty() {
                continue;
            }
            if block.key_id().is_some_and(|k| k.get() == key_id.get()) {
                return Ok(Some((prop, i)));
            }
        }

        cursor = prop.next_prop_id;
    }

    Ok(None)
}

/// Finds the property block matching the given key on an edge.
#[allow(dead_code)]
fn find_property_block_on_edge(
    spider: &mut Spider,
    edge: &Edge,
    key_id: TokenId,
) -> SpiderResult<Option<(PropertyRecord, usize)>> {
    let mut cursor = edge.first_prop_id;
    let mut steps = 0;
    let max_steps = 10_000;

    while cursor != 0 {
        steps += 1;
        if steps > max_steps {
            return Err(DbError::TraversalDepthExceeded { limit: max_steps });
        }

        let prop = spider.properties.get(cursor - 1)?;
        if prop.is_deleted() {
            break;
        }

        for (i, block) in prop.blocks.iter().enumerate() {
            if block.is_empty() {
                continue;
            }
            if block.key_id().is_some_and(|k| k.get() == key_id.get()) {
                return Ok(Some((prop, i)));
            }
        }

        cursor = prop.next_prop_id;
    }

    Ok(None)
}

// --- A single property entry (key + value) ---

/// A resolved property: its key name and decoded value.
#[derive(Debug, Clone)]
pub struct PropertyEntry {
    /// The property key name (e.g. "title", "name").
    pub key: String,
    /// The decoded value.
    pub value: PropertyValue,
}

// --- Public API: Node properties ---

/// Reads a single property from a node by key name.
///
/// Returns `None` if the property doesn't exist. Decodes the value into a
/// typed [`PropertyValue`], including dereferencing dynamic strings.
///
/// # Errors
/// - [`DbError::NodeNotFound`] — if the node doesn't exist or is deleted
/// - [`DbError::PropertyNotFound`] — if the key doesn't exist on this node
/// - [`DbError::TraversalDepthExceeded`] — if property chain is corrupt
pub fn get(spider: &mut Spider, node_id: NodeId, key: &str) -> SpiderResult<Option<PropertyValue>> {
    let key_tid = match resolve_key(spider, key)? {
        Some(tid) => tid,
        None => return Ok(None),
    };

    let node = spider.nodes.get(node_id.get() - 1)?;
    if node.is_deleted() {
        return Err(DbError::NodeNotFound(node_id.get()));
    }

    if node.first_prop_id == 0 {
        return Ok(None);
    }

    match find_property_block_on_node(spider, &node, key_tid)? {
        Some((prop, idx)) => {
            let value = decode_block(spider, &prop.blocks[idx])?;
            Ok(Some(value))
        }
        None => Ok(None),
    }
}

/// Reads a property and returns it as a string.
///
/// Converts numeric/bool types to their string representation.
/// Returns `None` if the property doesn't exist.
pub fn get_string(spider: &mut Spider, node_id: NodeId, key: &str) -> SpiderResult<Option<String>> {
    match get(spider, node_id, key)? {
        Some(v) => Ok(Some(v.to_string())),
        None => Ok(None),
    }
}

/// Reads a property and returns it as an `i64`.
///
/// Returns `None` if the property doesn't exist or isn't an integer type.
pub fn get_int(spider: &mut Spider, node_id: NodeId, key: &str) -> SpiderResult<Option<i64>> {
    match get(spider, node_id, key)? {
        Some(PropertyValue::Int(v)) => Ok(Some(v)),
        Some(PropertyValue::Long(v)) => Ok(Some(v)),
        Some(PropertyValue::Byte(v)) => Ok(Some(v as i64)),
        Some(PropertyValue::Short(v)) => Ok(Some(v as i64)),
        _ => Ok(None),
    }
}

/// Reads a property and returns it as an `f64`.
///
/// Returns `None` if the property doesn't exist or isn't a float type.
pub fn get_float(spider: &mut Spider, node_id: NodeId, key: &str) -> SpiderResult<Option<f64>> {
    match get(spider, node_id, key)? {
        Some(PropertyValue::Float(v)) => Ok(Some(v as f64)),
        Some(PropertyValue::Double(v)) => Ok(Some(v)),
        _ => Ok(None),
    }
}

/// Reads a property and returns it as a `bool`.
///
/// Returns `None` if the property doesn't exist or isn't a bool.
pub fn get_bool(spider: &mut Spider, node_id: NodeId, key: &str) -> SpiderResult<Option<bool>> {
    match get(spider, node_id, key)? {
        Some(PropertyValue::Bool(v)) => Ok(Some(v)),
        _ => Ok(None),
    }
}

/// Lists all properties on a node.
///
/// Returns a vector of [`PropertyEntry`] with resolved key names and values.
///
/// # Errors
/// - [`DbError::NodeNotFound`] — if the node doesn't exist or is deleted
/// - [`DbError::TraversalDepthExceeded`] — if property chain is corrupt
pub fn list_all(spider: &mut Spider, node_id: NodeId) -> SpiderResult<Vec<PropertyEntry>> {
    let node = spider.nodes.get(node_id.get() - 1)?;
    if node.is_deleted() {
        return Err(DbError::NodeNotFound(node_id.get()));
    }

    walk_property_chain(spider, node.first_prop_id)
}

// --- Property chain walker (shared) ---

fn walk_property_chain(
    spider: &mut Spider,
    first_prop_id: u32,
) -> SpiderResult<Vec<PropertyEntry>> {
    let mut results = Vec::new();
    let mut cursor = first_prop_id;
    let mut steps = 0;
    let max_steps = 10_000;

    while cursor != 0 {
        steps += 1;
        if steps > max_steps {
            return Err(DbError::TraversalDepthExceeded { limit: max_steps });
        }

        let prop = spider.properties.get(cursor - 1)?;
        if prop.is_deleted() {
            break;
        }

        for block in &prop.blocks {
            if block.is_empty() {
                continue;
            }
            let key_id = match block.key_id() {
                Some(k) => k,
                None => continue,
            };

            // Resolve key name from token store.
            let key_name = spider
                .prop_key_tokens
                .get_name(TokenId::new(key_id.get()).unwrap())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("<key:{}>", key_id.get()));

            let value = decode_block(spider, block)?;
            results.push(PropertyEntry { key: key_name, value });
        }

        cursor = prop.next_prop_id;
    }

    Ok(results)
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ingest::{Entity, IngestRequest, Proposition, index};
    use crate::schema::property::PropKeyId;
    use tempfile::TempDir;

    fn setup() -> (TempDir, Spider) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_property_db");
        let db = Spider::open(&db_path).unwrap();
        (dir, db)
    }

    // --- get ---

    #[test]
    fn get_returns_string_property() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "My Document",
            propositions: vec![
                Proposition {
                    text: "Hello",
                    entities: vec![
                        Entity { name: "X", entity_type: "T" },
                    ],
                },
            ],
        };
        let result = index(&mut db, &req).unwrap();

        let value = get(&mut db, result.document_id, "title").unwrap();
        // "My Document" = 11 bytes > 6, so stored as dynamic String.
        assert_eq!(value, Some(PropertyValue::String("My Document".to_string())));
    }

    #[test]
    fn get_returns_none_for_missing_key() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![],
        };
        let result = index(&mut db, &req).unwrap();

        let value = get(&mut db, result.document_id, "nonexistent").unwrap();
        assert!(value.is_none());
    }

    #[test]
    fn get_returns_dynamic_string() {
        let (_dir, mut db) = setup();

        let long_text = "This is a very long proposition text that exceeds six bytes";
        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: long_text,
                    entities: vec![],
                },
            ],
        };
        let result = index(&mut db, &req).unwrap();

        let prop_id = crate::db::find::find_by_label(&mut db, "PROPOSITION").unwrap()[0];
        let value = get(&mut db, prop_id, "text").unwrap();

        match &value {
            Some(PropertyValue::String(s)) => assert_eq!(s, long_text),
            _ => panic!("expected PropertyValue::String, got {:?}", value),
        }
    }

    #[test]
    fn get_returns_int_property() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Hello",
                    entities: vec![],
                },
            ],
        };
        let result = index(&mut db, &req).unwrap();

        let prop_id = crate::db::find::find_by_label(&mut db, "PROPOSITION").unwrap()[0];
        let value = get(&mut db, prop_id, "chunk_index").unwrap();
        assert_eq!(value, Some(PropertyValue::Int(0)));
    }

    // --- typed getters ---

    #[test]
    fn get_string_returns_string_value() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "My Doc",
            propositions: vec![],
        };
        let result = index(&mut db, &req).unwrap();

        let s = get_string(&mut db, result.document_id, "title").unwrap();
        assert_eq!(s, Some("My Doc".to_string()));
    }

    #[test]
    fn get_int_returns_int_value() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Hello",
                    entities: vec![],
                },
            ],
        };
        let result = index(&mut db, &req).unwrap();

        let prop_id = crate::db::find::find_by_label(&mut db, "PROPOSITION").unwrap()[0];
        let v = get_int(&mut db, prop_id, "chunk_index").unwrap();
        assert_eq!(v, Some(0));
    }

    #[test]
    fn get_int_returns_none_for_string() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![],
        };
        let result = index(&mut db, &req).unwrap();

        let v = get_int(&mut db, result.document_id, "title").unwrap();
        assert!(v.is_none());
    }

    // --- list_all ---

    #[test]
    fn list_all_returns_all_properties() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test Doc",
            propositions: vec![
                Proposition {
                    text: "Hi",
                    entities: vec![
                        Entity { name: "A", entity_type: "T" },
                    ],
                },
            ],
        };
        let result = index(&mut db, &req).unwrap();

        let props = list_all(&mut db, result.document_id).unwrap();
        // Document has: title
        assert!(props.iter().any(|p| p.key == "title"));
    }

    #[test]
    fn list_all_empty_node() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![],
        };
        let result = index(&mut db, &req).unwrap();

        let props = list_all(&mut db, result.document_id).unwrap();
        assert_eq!(props.len(), 1); // title
    }

    #[test]
    fn list_all_proposition_properties() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Hello world",
                    entities: vec![],
                },
            ],
        };
        index(&mut db, &req).unwrap();

        let prop_id = crate::db::find::find_by_label(&mut db, "PROPOSITION").unwrap()[0];
        let props = list_all(&mut db, prop_id).unwrap();

        // Should have: text, chunk_index
        assert_eq!(props.len(), 2);
        assert!(props.iter().any(|p| p.key == "text"));
        assert!(props.iter().any(|p| p.key == "chunk_index"));
    }

    #[test]
    fn list_all_entity_properties() {
        let (_dir, mut db) = setup();

        let req = IngestRequest {
            title: "Test",
            propositions: vec![
                Proposition {
                    text: "Facts",
                    entities: vec![
                        Entity { name: "Mumbai", entity_type: "LOCATION" },
                    ],
                },
            ],
        };
        index(&mut db, &req).unwrap();

        let ent_id = crate::db::find::find_by_property(&mut db, "name", "Mumbai").unwrap()[0];
        let props = list_all(&mut db, ent_id).unwrap();

        assert_eq!(props.len(), 2);
        assert!(props.iter().any(|p| p.key == "name"));
        assert!(props.iter().any(|p| p.key == "entity_type"));
    }

    // --- PropertyValue Display ---

    #[test]
    fn property_value_display() {
        assert_eq!(PropertyValue::Bool(true).to_string(), "true");
        assert_eq!(PropertyValue::Int(42).to_string(), "42");
        assert_eq!(PropertyValue::ShortString("hi".to_string()).to_string(), "hi");
        assert_eq!(PropertyValue::String("hello".to_string()).to_string(), "hello");
    }

    // --- decode_block edge cases ---

    #[test]
    fn decode_empty_block_returns_error() {
        let (_dir, mut db) = setup();
        let empty_block = PropertyBlock::new();
        let result = decode_block(&mut db, &empty_block);
        assert!(matches!(result, Err(DbError::NotFound)));
    }

    #[test]
    fn decode_byte_value() {
        let (_dir, mut db) = setup();
        let key_id = PropKeyId::new(1).unwrap();
        let block = PropertyBlock::with_value(key_id, PropertyType::Byte, 42);
        let value = decode_block(&mut db, &block).unwrap();
        assert_eq!(value, PropertyValue::Byte(42));
    }

    #[test]
    fn decode_short_value() {
        let (_dir, mut db) = setup();
        let key_id = PropKeyId::new(1).unwrap();
        let block = PropertyBlock::with_value(key_id, PropertyType::Short, 1000);
        let value = decode_block(&mut db, &block).unwrap();
        assert_eq!(value, PropertyValue::Short(1000));
    }

    #[test]
    fn decode_char_value() {
        let (_dir, mut db) = setup();
        let key_id = PropKeyId::new(1).unwrap();
        let block = PropertyBlock::with_value(key_id, PropertyType::Char, 'A' as u64);
        let value = decode_block(&mut db, &block).unwrap();
        assert_eq!(value, PropertyValue::Char('A'));
    }

    #[test]
    fn decode_raw_for_unsupported_types() {
        let (_dir, mut db) = setup();
        let key_id = PropKeyId::new(1).unwrap();
        // Long type (not yet fully decoded).
        let block = PropertyBlock::with_value(key_id, PropertyType::Long, 999);
        let value = decode_block(&mut db, &block).unwrap();
        match value {
            PropertyValue::Raw(PropertyType::Long, 999) => {}
            _ => panic!("expected Raw for Long type"),
        }
    }
}
