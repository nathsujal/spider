//! Property operations — typed get/list for node properties.
//!
//! Provides a high-level API over the raw [`PropertyBlock`](crate::schema::property::PropertyBlock)
//! storage layer. Resolves [`PropertyValue`] from disk including dereferencing
//! dynamic string chains and multi-block values.
//!
//! ## API
//!
//! - [`get()`] — read a single property by key, returns typed [`PropertyValue`]
//! - [`get_string()`], [`get_int()`], [`get_float()`], [`get_bool()`] — typed convenience getters
//! - [`list_all()`] — enumerate all properties on a node

use crate::db::lifecycle::Spider;
use crate::db::nodes::NodeId;
use crate::error::{DbError, SpiderResult};
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
    Bool(bool),
    Byte(i8),
    Short(i16),
    Int(i64),
    Long(i64),       // stored across 2 blocks, raw for now
    Float(f32),
    Double(f64),     // stored across 2 blocks, raw for now
    Char(char),
    ShortString(String),    // ≤6 bytes inline
    String(String),         // dereferenced from strings.db chain
    Raw(PropertyType, u64), // fallback for unimplemented types
}

impl std::fmt::Display for PropertyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{v}"),
            Self::Byte(v) => write!(f, "{v}"),
            Self::Short(v) => write!(f, "{v}"),
            Self::Int(v) | Self::Long(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Double(v) => write!(f, "{v}"),
            Self::Char(v) => write!(f, "{v}"),
            Self::ShortString(v) | Self::String(v) => write!(f, "{v}"),
            Self::Raw(ty, bits) => write!(f, "{ty:?}:0x{bits:x}"),
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
        PropertyType::Byte => Ok(PropertyValue::Byte(block.value_bits() as i8)),
        PropertyType::Short => Ok(PropertyValue::Short(block.value_bits() as i16)),
        PropertyType::Int => Ok(PropertyValue::Int(block.as_int().unwrap())),
        PropertyType::Float => Ok(PropertyValue::Float(block.as_float().unwrap())),
        PropertyType::Char => {
            let ch = char::from_u32(block.value_bits() as u32).unwrap_or('\0');
            Ok(PropertyValue::Char(ch))
        }
        PropertyType::ShortString => {
            Ok(PropertyValue::ShortString(block.as_short_string().unwrap()))
        }
        PropertyType::String => {
            let ptr = block.dyn_string_ptr().unwrap();
            Ok(PropertyValue::String(read_dynamic_string(spider, ptr)?))
        }
        // Multi-block and unimplemented types → raw
        other => Ok(PropertyValue::Raw(other, block.value_bits())),
    }
}

/// Reads a full string from a dynamic string chain.
fn read_dynamic_string(spider: &mut Spider, start_id: u32) -> SpiderResult<String> {
    let mut result = Vec::new();
    let mut cursor = start_id;
    let mut steps = 0;

    while cursor != 0 {
        steps += 1;
        if steps > 10_000 {
            return Err(DbError::TraversalDepthExceeded { limit: 10_000 });
        }

        let record = spider.strings.get(cursor - 1)?;
        if !record.is_in_use() {
            break;
        }

        if record.is_start() {
            result.extend_from_slice(record.get_data(record.get_length() as usize));
        } else {
            result.extend_from_slice(&record.data);
        }

        cursor = record.next_block;
    }

    String::from_utf8(result).map_err(|_| DbError::NotFound)
}

// --- Shared property chain walker ---

/// Yields `(property_record, block_index)` for every non-empty block in the chain.
///
/// Returns records by value (they are `Copy`), so the caller can decode
/// without holding a borrow on `spider`.
fn walk_property_blocks(
    spider: &mut Spider,
    first_prop_id: u32,
) -> SpiderResult<Vec<(PropertyRecord, usize)>> {
    let mut results = Vec::new();
    let mut cursor = first_prop_id;
    let mut steps = 0;

    while cursor != 0 {
        steps += 1;
        if steps > 10_000 {
            return Err(DbError::TraversalDepthExceeded { limit: 10_000 });
        }

        let prop = spider.properties.get(cursor - 1)?;
        if prop.is_deleted() {
            break;
        }

        for (i, block) in prop.blocks.iter().enumerate() {
            if !block.is_empty() {
                results.push((prop, i));
            }
        }

        cursor = prop.next_prop_id;
    }

    Ok(results)
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

// --- Public API ---

/// Reads a single property from a node by key name.
///
/// Returns `None` if the property doesn't exist. Decodes the value into a
/// typed [`PropertyValue`], including dereferencing dynamic strings.
///
/// # Errors
/// - [`DbError::NodeNotFound`] — if the node doesn't exist or is deleted
/// - [`DbError::TraversalDepthExceeded`] — if property chain is corrupt
pub fn get(spider: &mut Spider, node_id: NodeId, key: &str) -> SpiderResult<Option<PropertyValue>> {
    let key_tid = match spider.prop_key_tokens.get_id(key) {
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

    let blocks = walk_property_blocks(spider, node.first_prop_id)?;
    for (prop, idx) in blocks {
        let block = &prop.blocks[idx];
        if block.key_id().is_some_and(|k| k.get() == key_tid.get()) {
            return Ok(Some(decode_block(spider, block)?));
        }
    }

    Ok(None)
}

/// Reads a property and returns it as a string.
///
/// Converts numeric/bool types to their string representation.
/// Returns `None` if the property doesn't exist.
pub fn get_string(spider: &mut Spider, node_id: NodeId, key: &str) -> SpiderResult<Option<String>> {
    get(spider, node_id, key).map(|v| v.map(|v| v.to_string()))
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

    let blocks = walk_property_blocks(spider, node.first_prop_id)?;
    blocks
        .into_iter()
        .map(|(prop, idx)| {
            let block = &prop.blocks[idx];
            let key_id = match block.key_id() {
                Some(k) => k,
                None => return Err(DbError::NotFound),
            };

            let key_name = spider
                .prop_key_tokens
                .get_name(TokenId::new(key_id.get()).unwrap())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("<key:{}>", key_id.get()));

            let value = decode_block(spider, block)?;
            Ok(PropertyEntry { key: key_name, value })
        })
        .collect()
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
        let db = Spider::open(&dir.path().join("test_prop_db")).unwrap();
        (dir, db)
    }

    // --- get ---

    #[test]
    fn get_returns_string_property() {
        let (_dir, mut db) = setup();
        let result = index(&mut db, &IngestRequest {
            title: "My Document",
            propositions: vec![Proposition {
                text: "Hello",
                entities: vec![Entity { name: "X", entity_type: "T" }],
            }],
        }).unwrap();

        let value = get(&mut db, result.document_id, "title").unwrap();
        assert_eq!(value, Some(PropertyValue::String("My Document".to_string())));
    }

    #[test]
    fn get_returns_none_for_missing_key() {
        let (_dir, mut db) = setup();
        let result = index(&mut db, &IngestRequest {
            title: "Test", propositions: vec![],
        }).unwrap();

        assert!(get(&mut db, result.document_id, "nonexistent").unwrap().is_none());
    }

    #[test]
    fn get_returns_dynamic_string() {
        let (_dir, mut db) = setup();
        let long_text = "This is a very long proposition text that exceeds six bytes";
        index(&mut db, &IngestRequest {
            title: "Test",
            propositions: vec![Proposition { text: long_text, entities: vec![] }],
        }).unwrap();

        let prop_id = crate::db::find::find_by_label(&mut db, "PROPOSITION").unwrap()[0];
        let value = get(&mut db, prop_id, "text").unwrap();

        match value {
            Some(PropertyValue::String(s)) => assert_eq!(s, long_text),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn get_returns_int_property() {
        let (_dir, mut db) = setup();
        index(&mut db, &IngestRequest {
            title: "Test",
            propositions: vec![Proposition { text: "Hello", entities: vec![] }],
        }).unwrap();

        let prop_id = crate::db::find::find_by_label(&mut db, "PROPOSITION").unwrap()[0];
        assert_eq!(get(&mut db, prop_id, "chunk_index").unwrap(), Some(PropertyValue::Int(0)));
    }

    // --- typed getters ---

    #[test]
    fn get_string_returns_value() {
        let (_dir, mut db) = setup();
        let result = index(&mut db, &IngestRequest {
            title: "My Doc", propositions: vec![],
        }).unwrap();

        assert_eq!(get_string(&mut db, result.document_id, "title").unwrap(), Some("My Doc".to_string()));
    }

    #[test]
    fn get_int_returns_value() {
        let (_dir, mut db) = setup();
        index(&mut db, &IngestRequest {
            title: "Test",
            propositions: vec![Proposition { text: "Hello", entities: vec![] }],
        }).unwrap();

        let prop_id = crate::db::find::find_by_label(&mut db, "PROPOSITION").unwrap()[0];
        assert_eq!(get_int(&mut db, prop_id, "chunk_index").unwrap(), Some(0));
    }

    #[test]
    fn get_int_returns_none_for_string() {
        let (_dir, mut db) = setup();
        let result = index(&mut db, &IngestRequest {
            title: "Test", propositions: vec![],
        }).unwrap();

        assert!(get_int(&mut db, result.document_id, "title").unwrap().is_none());
    }

    // --- list_all ---

    #[test]
    fn list_all_document_properties() {
        let (_dir, mut db) = setup();
        let result = index(&mut db, &IngestRequest {
            title: "Test Doc",
            propositions: vec![Proposition {
                text: "Hi",
                entities: vec![Entity { name: "A", entity_type: "T" }],
            }],
        }).unwrap();

        let props = list_all(&mut db, result.document_id).unwrap();
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].key, "title");
    }

    #[test]
    fn list_all_proposition_properties() {
        let (_dir, mut db) = setup();
        index(&mut db, &IngestRequest {
            title: "Test",
            propositions: vec![Proposition { text: "Hello world", entities: vec![] }],
        }).unwrap();

        let prop_id = crate::db::find::find_by_label(&mut db, "PROPOSITION").unwrap()[0];
        let props = list_all(&mut db, prop_id).unwrap();

        assert_eq!(props.len(), 2);
        assert!(props.iter().any(|p| p.key == "text"));
        assert!(props.iter().any(|p| p.key == "chunk_index"));
    }

    #[test]
    fn list_all_entity_properties() {
        let (_dir, mut db) = setup();
        index(&mut db, &IngestRequest {
            title: "Test",
            propositions: vec![Proposition {
                text: "Facts",
                entities: vec![Entity { name: "Mumbai", entity_type: "LOCATION" }],
            }],
        }).unwrap();

        let ent_id = crate::db::find::find_by_property(&mut db, "name", "Mumbai").unwrap()[0];
        let props = list_all(&mut db, ent_id).unwrap();

        assert_eq!(props.len(), 2);
        assert!(props.iter().any(|p| p.key == "name"));
        assert!(props.iter().any(|p| p.key == "entity_type"));
    }

    // --- Display ---

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
        assert!(matches!(decode_block(&mut db, &PropertyBlock::new()), Err(DbError::NotFound)));
    }

    #[test]
    fn decode_byte_value() {
        let (_dir, mut db) = setup();
        let block = PropertyBlock::with_value(PropKeyId::new(1).unwrap(), PropertyType::Byte, 42);
        assert_eq!(decode_block(&mut db, &block).unwrap(), PropertyValue::Byte(42));
    }

    #[test]
    fn decode_short_value() {
        let (_dir, mut db) = setup();
        let block = PropertyBlock::with_value(PropKeyId::new(1).unwrap(), PropertyType::Short, 1000);
        assert_eq!(decode_block(&mut db, &block).unwrap(), PropertyValue::Short(1000));
    }

    #[test]
    fn decode_char_value() {
        let (_dir, mut db) = setup();
        let block = PropertyBlock::with_value(PropKeyId::new(1).unwrap(), PropertyType::Char, 'A' as u64);
        assert_eq!(decode_block(&mut db, &block).unwrap(), PropertyValue::Char('A'));
    }

    #[test]
    fn decode_raw_for_unsupported_types() {
        let (_dir, mut db) = setup();
        let block = PropertyBlock::with_value(PropKeyId::new(1).unwrap(), PropertyType::Long, 999);
        match decode_block(&mut db, &block).unwrap() {
            PropertyValue::Raw(PropertyType::Long, 999) => {}
            other => panic!("expected Raw for Long, got {other:?}"),
        }
    }
}
