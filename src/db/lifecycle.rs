//! Database lifecycle: open, close, free list helpers.

use super::*;
use std::path::{Path, PathBuf};

impl Spider {
    /// Open or create a new Spider DB.
    pub fn open<P: AsRef<Path>>(
        path: Option<P>,
        w_sig: Option<f64>,
        w_freq: Option<f64>,
        gravity: Option<f64>,
    ) -> Result<Self> {
        let path_buf = if let Some(p) = path {
            let p_ref = p.as_ref();
            if p_ref.as_os_str().is_empty() {
                Self::get_default_db_path()?
            } else {
                p_ref.to_path_buf()
            }
        } else {
            Self::get_default_db_path()?
        };

        std::fs::create_dir_all(&path_buf)?;

        // Open record files
        let nodes = RecordFile::open(&path_buf.join("nodes.db"))?;
        let rels = RecordFile::open(&path_buf.join("rels.db"))?;
        let props = RecordFile::open(&path_buf.join("props.db"))?;
        let strings = RecordFile::open(&path_buf.join("strings.db"))?;
        let arrays = RecordFile::open(&path_buf.join("arrays.db"))?;

        // Load metadata
        let mut meta = Metadata::load(&path_buf.join("meta.db"))?;
        // Override persisted bio params with any values passed by the caller.
        if let Some(v) = w_sig { meta.bio_w_sig = v; }
        if let Some(v) = w_freq { meta.bio_w_freq = v; }
        if let Some(v) = gravity { meta.bio_gravity = v; }

        // Load token stores
        let labels = TokenStore::load(&path_buf.join("labels.tok")).unwrap_or_default();
        let rel_types = TokenStore::load(&path_buf.join("rel_types.tok")).unwrap_or_default();
        let prop_keys = TokenStore::load(&path_buf.join("prop_keys.tok")).unwrap_or_default();

        // Load free lists (or create empty)
        let node_free = Self::load_free_list(&path_buf.join("node_free.bin"));
        let rel_free = Self::load_free_list(&path_buf.join("rel_free.bin"));
        let prop_free = Self::load_free_list(&path_buf.join("prop_free.bin"));
        let string_free = Self::load_free_list(&path_buf.join("string_free.bin"));
        let array_free = Self::load_free_list(&path_buf.join("array_free.bin"));

        // Open content store
        let content = ContentStore::open(&path_buf)?;

        Ok(Self {
            path: path_buf,
            nodes,
            rels,
            props,
            strings,
            arrays,
            node_free,
            rel_free,
            prop_free,
            string_free,
            array_free,
            meta,
            labels,
            rel_types,
            prop_keys,
            content,
        })
    }

    /// Resolve the default global directory for Spider.
    fn get_default_db_path() -> Result<PathBuf> {
        if let Some(proj_dirs) = directories::ProjectDirs::from("", "", "spider") {
            Ok(proj_dirs.data_local_dir().join("db"))
        } else {
            Err(DbError::Corrupted("Could not resolve default OS data directory".to_string()))
        }
    }

    /// Close the database, flushing all data to disk.
    pub fn close(&mut self) -> Result<()> {
        // Sync record files
        self.nodes.sync()?;
        self.rels.sync()?;
        self.props.sync()?;
        self.strings.sync()?;
        self.arrays.sync()?;

        // Save metadata
        self.meta.save(&self.path.join("meta.db"))?;

        // Save token stores
        self.labels.save(&self.path.join("labels.tok"))?;
        self.rel_types.save(&self.path.join("rel_types.tok"))?;
        self.prop_keys.save(&self.path.join("prop_keys.tok"))?;

        // Save free lists
        Self::save_free_list(&self.node_free, &self.path.join("node_free.bin"))?;
        Self::save_free_list(&self.rel_free, &self.path.join("rel_free.bin"))?;
        Self::save_free_list(&self.prop_free, &self.path.join("prop_free.bin"))?;
        Self::save_free_list(&self.string_free, &self.path.join("string_free.bin"))?;
        Self::save_free_list(&self.array_free, &self.path.join("array_free.bin"))?;

        // Flush content store manifest
        self.content.flush()?;

        Ok(())
    }

    fn load_free_list(path: &Path) -> FreeList {
        if let Ok(bytes) = std::fs::read(path) {
            FreeList::from_bytes(&bytes).unwrap_or_default()
        } else {
            FreeList::new()
        }
    }

    fn save_free_list(list: &FreeList, path: &Path) -> Result<()> {
        std::fs::write(path, list.to_bytes())?;
        Ok(())
    }
}

impl Drop for Spider {
    fn drop(&mut self) {
        let _ = self.close();
    }
}