//! Property operations for Spider.
//!
//! Provides `PropertyValue` enum and methods to set/get/delete
//! properties on nodes and relationships.

use crate::db::{DbError, Result, Spider};
use crate::schema::{DynamicStringRecord, PropertyBlock, PropertyRecord, PropertyType};

// PropertyValue

/// High-level property value that maps to storage types.
///
/// - `Bool`   -> inline (1 bit)
/// - `Int`    -> inline (51-bit signed)
/// - `Float`  -> inline (f32)
/// - `String` -> inline if ≤6 bytes, otherwise stored in strings.db
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    Bool(bool),
    Int(i64),
    Float(f32),
    String(String),
}

impl std::fmt::Display for PropertyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PropertyValue::Bool(v) => write!(f, "{}", v),
            PropertyValue::Int(v) => write!(f, "{}", v),
            PropertyValue::Float(v) => write!(f, "{}", v),
            PropertyValue::String(v) => write!(f, "{}", v),
        }
    }
}

// Spider Property Methods

impl Spider {
    // Node Properties

    /// Set a property on a node.
    ///
    /// If the key already exists, the value is updated.
    /// If the key is new, it is inserted into the first available slot.
    pub fn set_node_property(
        &mut self,
        node_id: u32,
        key: &str,
        value: PropertyValue,
    ) -> Result<()> {
        // 1. Validate node exists
        let mut node = self
            .nodes
            .read(node_id)
            .ok_or(DbError::NodeNotFound(node_id))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(node_id));
        }

        // 2. Resolve key name -> key_id (u8)
        let key_id = self
            .prop_keys
            .get_or_create(key)
            .ok_or(DbError::TokenStoreExhausted { store: "prop_keys" })?;

        // 3. Encode value into a PropertyBlock
        let block = self.encode_block(key_id, &value)?;

        // 4. Insert or update
        if node.first_prop_id == 0 {
            // No properties yet — create first record
            let new_id = self.prop_free.allocate(&mut self.meta.next_prop_id);
            let mut record = PropertyRecord::new();
            record.blocks[0] = block;
            self.props.write(new_id, &record)?;

            node.first_prop_id = new_id;
            self.nodes.write(node_id, &node)?;
        } else {
            // Traverse existing chain
            self.insert_or_update_property(node.first_prop_id, key_id, block)?;
        }

        Ok(())
    }

    /// Get a property value from a node by key name.
    ///
    /// Returns `Ok(None)` if the key does not exist on this node.
    pub fn get_node_property(
        &self,
        node_id: u32,
        key: &str,
    ) -> Result<Option<PropertyValue>> {
        let node = self
            .nodes
            .read(node_id)
            .ok_or(DbError::NodeNotFound(node_id))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(node_id));
        }

        let key_id = match self.prop_keys.get_id(key) {
            Some(id) => id,
            None => return Ok(None), // Key name was never used
        };

        self.find_property(node.first_prop_id, key_id)
    }

    /// Delete a property from a node by key name.
    ///
    /// No-op if the key does not exist.
    pub fn delete_node_property(&mut self, node_id: u32, key: &str) -> Result<()> {
        let node = self
            .nodes
            .read(node_id)
            .ok_or(DbError::NodeNotFound(node_id))?;
        if node.is_deleted() {
            return Err(DbError::NodeNotFound(node_id));
        }

        let key_id = match self.prop_keys.get_id(key) {
            Some(id) => id,
            None => return Ok(()),
        };

        self.clear_property(node.first_prop_id, key_id)
    }

    // Relationship Properties

    /// Set a property on a relationship.
    pub fn set_rel_property(
        &mut self,
        rel_id: u32,
        key: &str,
        value: PropertyValue,
    ) -> Result<()> {
        let mut rel = self
            .rels
            .read(rel_id)
            .ok_or(DbError::RelNotFound(rel_id))?;
        if rel.is_deleted() {
            return Err(DbError::RelNotFound(rel_id));
        }

        let key_id = self
            .prop_keys
            .get_or_create(key)
            .ok_or(DbError::TokenStoreExhausted { store: "prop_keys" })?;

        let block = self.encode_block(key_id, &value)?;

        if rel.first_prop_id == 0 {
            let new_id = self.prop_free.allocate(&mut self.meta.next_prop_id);
            let mut record = PropertyRecord::new();
            record.blocks[0] = block;
            self.props.write(new_id, &record)?;

            rel.first_prop_id = new_id;
            self.rels.write(rel_id, &rel)?;
        } else {
            self.insert_or_update_property(rel.first_prop_id, key_id, block)?;
        }

        Ok(())
    }

    /// Get a property value from a relationship.
    pub fn get_rel_property(
        &self,
        rel_id: u32,
        key: &str,
    ) -> Result<Option<PropertyValue>> {
        let rel = self
            .rels
            .read(rel_id)
            .ok_or(DbError::RelNotFound(rel_id))?;
        if rel.is_deleted() {
            return Err(DbError::RelNotFound(rel_id));
        }

        let key_id = match self.prop_keys.get_id(key) {
            Some(id) => id,
            None => return Ok(None),
        };

        self.find_property(rel.first_prop_id, key_id)
    }

    /// Delete a property from a relationship.
    pub fn delete_rel_property(&mut self, rel_id: u32, key: &str) -> Result<()> {
        let rel = self
            .rels
            .read(rel_id)
            .ok_or(DbError::RelNotFound(rel_id))?;
        if rel.is_deleted() {
            return Err(DbError::RelNotFound(rel_id));
        }

        let key_id = match self.prop_keys.get_id(key) {
            Some(id) => id,
            None => return Ok(()),
        };

        self.clear_property(rel.first_prop_id, key_id)
    }

    // Internal Helpers

    /// Encode a PropertyValue into a PropertyBlock.
    ///
    /// For strings: ≤6 bytes → ShortString (inline), >6 bytes → DynString (write to strings.db).
    fn encode_block(&mut self, key_id: u8, value: &PropertyValue) -> Result<PropertyBlock> {
        match value {
            PropertyValue::Bool(v) => Ok(PropertyBlock::from_bool(key_id, *v)),
            PropertyValue::Int(v) => PropertyBlock::from_int(key_id, *v)
                .ok_or(DbError::ValueTooLarge { max_bytes: 6 }),
            PropertyValue::Float(v) => Ok(PropertyBlock::from_float(key_id, *v)),
            PropertyValue::String(s) => {
                // Try inline first (≤6 bytes)
                if let Some(block) = PropertyBlock::from_short_string(key_id, s) {
                    Ok(block)
                } else {
                    // Write to strings.db
                    let string_id = self.write_dynamic_string(s)?;
                    Ok(PropertyBlock::from_dyn_string_ptr(key_id, string_id))
                }
            }
        }
    }

    /// Decode a PropertyBlock into a PropertyValue.
    pub(crate) fn decode_block(&self, block: &PropertyBlock) -> Option<PropertyValue> {
        match block.value_type() {
            PropertyType::Bool => block.as_bool().map(PropertyValue::Bool),
            PropertyType::Int => block.as_int().map(PropertyValue::Int),
            PropertyType::Float => block.as_float().map(PropertyValue::Float),
            PropertyType::ShortString => block.as_short_string().map(PropertyValue::String),
            PropertyType::String => {
                let ptr = block.dyn_string_ptr()?;
                self.read_dynamic_string(ptr).ok().map(PropertyValue::String)
            }
            _ => None,
        }
    }

    /// Traverse property chain to find a value by key_id.
    pub(crate) fn find_property(
        &self,
        first_prop_id: u32,
        key_id: u8,
    ) -> Result<Option<PropertyValue>> {
        let mut prop_id = first_prop_id;
        while prop_id != 0 {
            let record = self.props.read(prop_id).ok_or(DbError::Corrupted(
                format!("Property record {} missing", prop_id),
            ))?;

            for block in &record.blocks {
                if !block.is_empty() && block.key_id() == key_id {
                    return Ok(self.decode_block(block));
                }
            }
            prop_id = record.next_prop_id;
        }
        Ok(None)
    }

    /// Insert a new block or update existing one in the property chain.
    fn insert_or_update_property(
        &mut self,
        first_prop_id: u32,
        key_id: u8,
        block: PropertyBlock,
    ) -> Result<()> {
        let mut prop_id = first_prop_id;

        loop {
            let mut record = self.props.read(prop_id).ok_or(DbError::Corrupted(
                format!("Property record {} missing", prop_id),
            ))?;

            // Check if key already exists in this record (Update)
            for i in 0..4 {
                if !record.blocks[i].is_empty() && record.blocks[i].key_id() == key_id {
                    // Free old DynString if it was one
                    self.free_block_string(&record.blocks[i])?;
                    record.blocks[i] = block;
                    self.props.write(prop_id, &record)?;
                    return Ok(());
                }
            }

            // If we're at the tail, insert here
            if record.next_prop_id == 0 {
                if let Some(idx) = record.first_empty_slot() {
                    record.blocks[idx] = block;
                    self.props.write(prop_id, &record)?;
                } else {
                    let new_id = self.prop_free.allocate(&mut self.meta.next_prop_id);
                    let mut new_record = PropertyRecord::new();
                    new_record.prev_prop_id = prop_id;
                    new_record.blocks[0] = block;
                    self.props.write(new_id, &new_record)?;

                    record.next_prop_id = new_id;
                    self.props.write(prop_id, &record)?;
                }
                return Ok(());
            }

            prop_id = record.next_prop_id;
        }
    }

    /// Clear (delete) a property block by key_id, freeing any DynString data.
    fn clear_property(&mut self, first_prop_id: u32, key_id: u8) -> Result<()> {
        let mut prop_id = first_prop_id;
        while prop_id != 0 {
            let mut record = self.props.read(prop_id).ok_or(DbError::Corrupted(
                format!("Property record {} missing", prop_id),
            ))?;

            for i in 0..4 {
                if !record.blocks[i].is_empty() && record.blocks[i].key_id() == key_id {
                    self.free_block_string(&record.blocks[i])?;
                    record.blocks[i] = PropertyBlock::new();
                    self.props.write(prop_id, &record)?;
                    return Ok(());
                }
            }

            prop_id = record.next_prop_id;
        }
        Ok(())
    }

    // Dynamic String Helpers

    /// Write a string to `strings.db`, returning the first record ID.
    fn write_dynamic_string(&mut self, s: &str) -> Result<u32> {
        let bytes = s.as_bytes();
        let total_len = bytes.len() as u32;

        // Split into 120-byte chunks
        let chunks: Vec<&[u8]> = bytes.chunks(DynamicStringRecord::DATA_SIZE).collect();

        // Allocate all IDs first (so we can set next_block pointers)
        let mut ids: Vec<u32> = Vec::with_capacity(chunks.len());
        for _ in 0..chunks.len() {
            ids.push(self.string_free.allocate(&mut self.meta.next_string_id));
        }

        // Write records in reverse (tail first) so next_block is known
        for (i, chunk) in chunks.iter().enumerate() {
            let next = if i + 1 < ids.len() { ids[i + 1] } else { 0 };
            let record = if i == 0 {
                DynamicStringRecord::new_start(chunk, total_len, next)
            } else {
                DynamicStringRecord::new_continuation(chunk, next)
            };
            self.strings.write(ids[i], &record)?;
        }

        Ok(ids[0])
    }

    /// Read a full string from `strings.db` by following the chain.
    fn read_dynamic_string(&self, start_id: u32) -> Result<String> {
        let first = self.strings.read(start_id).ok_or(DbError::Corrupted(
            format!("String record {} missing", start_id),
        ))?;

        let total_len = first.get_length() as usize;
        let mut result = Vec::with_capacity(total_len);

        // Read first block
        let take = total_len.min(DynamicStringRecord::DATA_SIZE);
        result.extend_from_slice(&first.data[..take]);

        // Follow chain
        let mut next_id = first.next_block;
        while next_id != 0 && result.len() < total_len {
            let block = self.strings.read(next_id).ok_or(DbError::Corrupted(
                format!("String chain broken at {}", next_id),
            ))?;
            let remaining = total_len - result.len();
            let take = remaining.min(DynamicStringRecord::DATA_SIZE);
            result.extend_from_slice(&block.data[..take]);
            next_id = block.next_block;
        }

        String::from_utf8(result)
            .map_err(|_| DbError::Corrupted("Invalid UTF-8 in string record".into()))
    }

    /// Free a DynString chain if the block points to one.
    fn free_block_string(&mut self, block: &PropertyBlock) -> Result<()> {
        if block.value_type() == PropertyType::String {
            if let Some(ptr) = block.dyn_string_ptr() {
                self.free_dynamic_string(ptr)?;
            }
        }
        Ok(())
    }

    /// Free an entire property chain: all DynStrings + all PropertyRecords.
    ///
    /// Called during cascade delete when a node or relationship is removed.
    /// Walks `first_prop_id → next → next → ...` freeing everything.
    pub(crate) fn free_all_properties(&mut self, first_prop_id: u32) -> Result<()> {
        let mut prop_id = first_prop_id;

        while prop_id != 0 {
            let record = self.props.read(prop_id).ok_or(DbError::Corrupted(
                format!("Property record {} missing during cascade delete", prop_id),
            ))?;

            // Free DynStrings in all 4 blocks
            for block in &record.blocks {
                if !block.is_empty() {
                    self.free_block_string(block)?;
                }
            }

            let next = record.next_prop_id;

            // Clear and return this PropertyRecord to free list
            self.props.write(prop_id, &PropertyRecord::new())?;
            self.prop_free.free(prop_id);

            prop_id = next;
        }

        Ok(())
    }

    /// Walk a DynamicString chain and free all blocks.
    fn free_dynamic_string(&mut self, start_id: u32) -> Result<()> {
        let mut id = start_id;
        while id != 0 {
            let record = match self.strings.read(id) {
                Some(r) => r,
                None => break,
            };
            let next = record.next_block;

            // Mark deleted and return to free list
            let mut deleted = record;
            deleted.delete();
            self.strings.write(id, &deleted)?;
            self.string_free.free(id);

            id = next;
        }
        Ok(())
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn set_and_get_bool() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id, "active", PropertyValue::Bool(true)).unwrap();
        assert_eq!(db.get_node_property(id, "active").unwrap(), Some(PropertyValue::Bool(true)));
    }

    #[test]
    fn set_and_get_int() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id, "age", PropertyValue::Int(30)).unwrap();
        assert_eq!(db.get_node_property(id, "age").unwrap(), Some(PropertyValue::Int(30)));
    }

    #[test]
    fn set_and_get_negative_int() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Account"]).unwrap();
        db.set_node_property(id, "balance", PropertyValue::Int(-500)).unwrap();
        assert_eq!(db.get_node_property(id, "balance").unwrap(), Some(PropertyValue::Int(-500)));
    }

    #[test]
    fn set_and_get_float() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Sensor"]).unwrap();
        db.set_node_property(id, "temp", PropertyValue::Float(36.5)).unwrap();
        assert_eq!(db.get_node_property(id, "temp").unwrap(), Some(PropertyValue::Float(36.5)));
    }

    #[test]
    fn update_existing_property() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id, "score", PropertyValue::Int(10)).unwrap();
        db.set_node_property(id, "score", PropertyValue::Int(99)).unwrap();
        assert_eq!(db.get_node_property(id, "score").unwrap(), Some(PropertyValue::Int(99)));
    }

    #[test]
    fn delete_property() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id, "age", PropertyValue::Int(25)).unwrap();
        db.delete_node_property(id, "age").unwrap();
        assert_eq!(db.get_node_property(id, "age").unwrap(), None);
    }

    #[test]
    fn get_nonexistent_property() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();
        assert_eq!(db.get_node_property(id, "missing").unwrap(), None);
    }

    #[test]
    fn multiple_properties_on_node() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();
        db.set_node_property(id, "active", PropertyValue::Bool(true)).unwrap();
        db.set_node_property(id, "age", PropertyValue::Int(30)).unwrap();
        db.set_node_property(id, "score", PropertyValue::Float(9.5)).unwrap();
        assert_eq!(db.get_node_property(id, "active").unwrap(), Some(PropertyValue::Bool(true)));
        assert_eq!(db.get_node_property(id, "age").unwrap(), Some(PropertyValue::Int(30)));
        assert_eq!(db.get_node_property(id, "score").unwrap(), Some(PropertyValue::Float(9.5)));
    }

    #[test]
    fn overflow_to_second_record() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Thing"]).unwrap();
        db.set_node_property(id, "a", PropertyValue::Int(1)).unwrap();
        db.set_node_property(id, "b", PropertyValue::Int(2)).unwrap();
        db.set_node_property(id, "c", PropertyValue::Int(3)).unwrap();
        db.set_node_property(id, "d", PropertyValue::Int(4)).unwrap();
        db.set_node_property(id, "e", PropertyValue::Int(5)).unwrap();
        assert_eq!(db.get_node_property(id, "e").unwrap(), Some(PropertyValue::Int(5)));
        assert_eq!(db.get_node_property(id, "a").unwrap(), Some(PropertyValue::Int(1)));
    }

    #[test]
    fn rel_properties() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let a = db.create_node(&["Person"]).unwrap();
        let b = db.create_node(&["Person"]).unwrap();
        let rel = db.create_rel(a, b, "KNOWS").unwrap();
        db.set_rel_property(rel, "since", PropertyValue::Int(2020)).unwrap();
        db.set_rel_property(rel, "weight", PropertyValue::Float(0.8)).unwrap();
        assert_eq!(db.get_rel_property(rel, "since").unwrap(), Some(PropertyValue::Int(2020)));
        assert_eq!(db.get_rel_property(rel, "weight").unwrap(), Some(PropertyValue::Float(0.8)));
        db.delete_rel_property(rel, "since").unwrap();
        assert_eq!(db.get_rel_property(rel, "since").unwrap(), None);
    }

    #[test]
    fn int_too_large_error() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Big"]).unwrap();
        let result = db.set_node_property(id, "big", PropertyValue::Int(i64::MAX));
        assert!(result.is_err());
    }

    #[test]
    fn persist_properties() {
        let dir = tempdir().unwrap();
        {
            let mut db = Spider::open(dir.path()).unwrap();
            let id = db.create_node(&["Person"]).unwrap();
            db.set_node_property(id, "age", PropertyValue::Int(42)).unwrap();
            db.close().unwrap();
        }
        {
            let db = Spider::open(dir.path()).unwrap();
            assert_eq!(db.get_node_property(1, "age").unwrap(), Some(PropertyValue::Int(42)));
        }
    }

    // String Tests

    #[test]
    fn short_string_inline() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();

        // "en" is 2 bytes → stored inline as ShortString
        db.set_node_property(id, "lang", PropertyValue::String("en".into())).unwrap();
        assert_eq!(
            db.get_node_property(id, "lang").unwrap(),
            Some(PropertyValue::String("en".into()))
        );
    }

    #[test]
    fn short_string_max() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();

        // Exactly 6 bytes → still inline
        db.set_node_property(id, "code", PropertyValue::String("abcdef".into())).unwrap();
        assert_eq!(
            db.get_node_property(id, "code").unwrap(),
            Some(PropertyValue::String("abcdef".into()))
        );
    }

    #[test]
    fn long_string_dynamic() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();

        // 13 bytes → stored in strings.db
        let name = "Alice Johnson".to_string();
        db.set_node_property(id, "name", PropertyValue::String(name.clone())).unwrap();
        assert_eq!(
            db.get_node_property(id, "name").unwrap(),
            Some(PropertyValue::String(name))
        );
    }

    #[test]
    fn very_long_string_multi_block() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Doc"]).unwrap();

        // 300 bytes → spans 3 DynamicStringRecords (120 bytes each)
        let long = "x".repeat(300);
        db.set_node_property(id, "body", PropertyValue::String(long.clone())).unwrap();
        assert_eq!(
            db.get_node_property(id, "body").unwrap(),
            Some(PropertyValue::String(long))
        );
    }

    #[test]
    fn update_string_property() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();

        db.set_node_property(id, "name", PropertyValue::String("Old Name Here".into())).unwrap();
        db.set_node_property(id, "name", PropertyValue::String("New Name Here".into())).unwrap();
        assert_eq!(
            db.get_node_property(id, "name").unwrap(),
            Some(PropertyValue::String("New Name Here".into()))
        );
    }

    #[test]
    fn delete_dynamic_string() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();

        db.set_node_property(id, "bio", PropertyValue::String("A long biography text".into())).unwrap();
        db.delete_node_property(id, "bio").unwrap();
        assert_eq!(db.get_node_property(id, "bio").unwrap(), None);
    }

    #[test]
    fn empty_string() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();

        db.set_node_property(id, "note", PropertyValue::String("".into())).unwrap();
        assert_eq!(
            db.get_node_property(id, "note").unwrap(),
            Some(PropertyValue::String("".into()))
        );
    }

    #[test]
    fn mixed_property_types() {
        let dir = tempdir().unwrap();
        let mut db = Spider::open(dir.path()).unwrap();
        let id = db.create_node(&["Person"]).unwrap();

        db.set_node_property(id, "name", PropertyValue::String("Alice Wonderland".into())).unwrap();
        db.set_node_property(id, "age", PropertyValue::Int(30)).unwrap();
        db.set_node_property(id, "active", PropertyValue::Bool(true)).unwrap();
        db.set_node_property(id, "lang", PropertyValue::String("en".into())).unwrap();

        assert_eq!(db.get_node_property(id, "name").unwrap(), Some(PropertyValue::String("Alice Wonderland".into())));
        assert_eq!(db.get_node_property(id, "age").unwrap(), Some(PropertyValue::Int(30)));
        assert_eq!(db.get_node_property(id, "active").unwrap(), Some(PropertyValue::Bool(true)));
        assert_eq!(db.get_node_property(id, "lang").unwrap(), Some(PropertyValue::String("en".into())));
    }

    #[test]
    fn persist_string_properties() {
        let dir = tempdir().unwrap();
        {
            let mut db = Spider::open(dir.path()).unwrap();
            let id = db.create_node(&["Person"]).unwrap();
            db.set_node_property(id, "name", PropertyValue::String("Alice Wonderland".into())).unwrap();
            db.set_node_property(id, "tag", PropertyValue::String("hi".into())).unwrap();
            db.close().unwrap();
        }
        {
            let db = Spider::open(dir.path()).unwrap();
            assert_eq!(db.get_node_property(1, "name").unwrap(), Some(PropertyValue::String("Alice Wonderland".into())));
            assert_eq!(db.get_node_property(1, "tag").unwrap(), Some(PropertyValue::String("hi".into())));
        }
    }
}
