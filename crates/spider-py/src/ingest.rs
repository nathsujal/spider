// Ingestion types: Entity, Proposition, IngestRequest, IngestResult

use pyo3::prelude::*;

use spider_core::db::ingest::{Entity as IngestEntity, Proposition as IngestProposition, IngestRequest as IngestRequestInner, IngestResult as IngestResultInner};

use crate::types::PyNodeId;

// ============================================================================
// PyEntity
// ============================================================================

/// A named entity mentioned in a proposition.
///
/// Entities represent real-world concepts like people, places, or organizations.
/// During ingestion, entities with the same name are automatically deduplicated.
///
/// Example:
/// ```python
/// import spider
///
/// entity = spider.Entity("Mumbai", "LOCATION")
/// print(entity.name)         # "Mumbai"
/// print(entity.entity_type)  # "LOCATION"
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PyEntity {
    /// The entity name (e.g. "Mumbai", "Albert Einstein").
    #[pyo3(get)]
    name: String,
    /// The entity type (e.g. "PERSON", "LOCATION", "CONCEPT", "ORGANIZATION").
    #[pyo3(get)]
    entity_type: String,
}

#[pymethods]
impl PyEntity {
    /// Create a new Entity.
    ///
    /// Args:
    ///     name: The entity name (e.g. "Mumbai").
    ///     entity_type: The entity type (e.g. "LOCATION").
    ///
    /// Returns:
    ///     Entity: A new Entity instance.
    #[new]
    fn new(name: &str, entity_type: &str) -> Self {
        PyEntity {
            name: name.to_string(),
            entity_type: entity_type.to_string(),
        }
    }

    /// Return a debug representation.
    fn __repr__(&self) -> String {
        format!("Entity(name='{}', type='{}')", self.name, self.entity_type)
    }
}

// Conversion: &PyEntity -> IngestEntity<'_>
impl PyEntity {
    /// Convert to a borrowed spider-core Entity.
    ///
    /// The returned Entity borrows &str slices pointing into this PyEntity's
    /// owned String fields. The borrow is valid for the duration of the
    /// method call since the PyEntity is held alive by the caller.
    pub fn to_rust(&self) -> IngestEntity<'_> {
        IngestEntity {
            name: &self.name,
            entity_type: &self.entity_type,
        }
    }
}

// ============================================================================
// PyProposition
// ============================================================================

/// A factual statement with associated entities.
///
/// Propositions represent atomic facts extracted from documents.
/// Each proposition contains text and a list of entities mentioned in it.
///
/// Example:
/// ```python
/// import spider
///
/// prop = spider.Proposition(
///     "Mumbai is the financial capital of India",
///     [
///         spider.Entity("Mumbai", "LOCATION"),
///         spider.Entity("India", "LOCATION"),
///     ],
/// )
/// print(prop.text)      # "Mumbai is the financial capital of India"
/// print(prop.entities)  # [Entity(...), Entity(...)]
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PyProposition {
    /// The text of the proposition.
    #[pyo3(get)]
    text: String,
    /// Named entities mentioned in this proposition.
    #[pyo3(get)]
    entities: Vec<PyEntity>,
}

#[pymethods]
impl PyProposition {
    /// Create a new Proposition.
    ///
    /// Args:
    ///     text: The proposition text (e.g. "Mumbai is the financial capital...").
    ///     entities: Optional list of Entity objects mentioned in this proposition.
    ///               Defaults to empty list.
    ///
    /// Returns:
    ///     Proposition: A new Proposition instance.
    #[new]
    #[pyo3(signature = (text, entities=None))]
    fn new(text: &str, entities: Option<Vec<PyEntity>>) -> Self {
        PyProposition {
            text: text.to_string(),
            entities: entities.unwrap_or_default(),
        }
    }

    /// Return a debug representation.
    fn __repr__(&self) -> String {
        format!("Proposition(text='{}', entities={})", self.text, self.entities.len())
    }
}

// Conversion: &PyProposition -> IngestProposition<'_>
impl PyProposition {
    /// Convert to a borrowed spider-core Proposition.
    ///
    /// The returned Proposition borrows &str slices pointing into this
    /// PyProposition's owned String fields and PyEntity fields. The borrow
    /// is valid for the duration of the method call.
    pub fn to_rust(&self) -> IngestProposition<'_> {
        IngestProposition {
            text: &self.text,
            entities: self.entities.iter().map(|e| e.to_rust()).collect(),
        }
    }
}

// ============================================================================
// PyIngestRequest
// ============================================================================

/// A request to ingest a document with propositions into the Spider database.
///
/// This is the primary input to `Spider.index()`. It contains a document
/// title and a list of propositions to ingest.
///
/// Example:
/// ```python
/// import spider
///
/// request = spider.IngestRequest(
///     title="My Document",
///     propositions=[
///         spider.Proposition(
///             "Mumbai is the financial capital of India",
///             [spider.Entity("Mumbai", "LOCATION")],
///         ),
///     ],
/// )
///
/// result = db.index(request)
/// print(result.document_id)
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PyIngestRequest {
    /// Document title (stored as the "title" property on the Document node).
    #[pyo3(get)]
    title: String,
    /// Pre-extracted propositions from the LLM pipeline.
    #[pyo3(get)]
    propositions: Vec<PyProposition>,
}

#[pymethods]
impl PyIngestRequest {
    /// Create a new IngestRequest.
    ///
    /// Args:
    ///     title: The document title.
    ///     propositions: Optional list of Proposition objects. Defaults to empty list.
    ///
    /// Returns:
    ///     IngestRequest: A new IngestRequest instance.
    #[new]
    #[pyo3(signature = (title, propositions=None))]
    fn new(title: &str, propositions: Option<Vec<PyProposition>>) -> Self {
        PyIngestRequest {
            title: title.to_string(),
            propositions: propositions.unwrap_or_default(),
        }
    }

    /// Return a debug representation.
    fn __repr__(&self) -> String {
        format!(
            "IngestRequest(title='{}', propositions={})",
            self.title,
            self.propositions.len()
        )
    }
}

impl PyIngestRequest {
    /// Convert to a borrowed spider-core IngestRequest<'_>.
    ///
    /// **Critical lifetime note**: The returned `IngestRequest<'_>` borrows
    /// `&str` slices pointing into this `PyIngestRequest`'s owned `String`
    /// fields and `PyProposition`/`PyEntity` fields. The borrow is valid
    /// for the duration of the method call because the Python object is
    /// held alive by the caller during the `index()` invocation.
    ///
    /// This method must be called **BEFORE** releasing the GIL via
    /// `py.allow_threads()`, since it accesses Python-owned string data.
    pub fn to_rust(&self) -> IngestRequestInner<'_> {
        IngestRequestInner {
            title: &self.title,
            propositions: self.propositions.iter().map(|p| p.to_rust()).collect(),
        }
    }
}

// ============================================================================
// PyIngestResult
// ============================================================================

/// The result of an ingestion operation.
///
/// Returned by `Spider.index()`. Contains the ID of the created document
/// node and counts of nodes/edges created.
///
/// Example:
/// ```python
/// result = db.index(request)
/// print(f"Document ID: {result.document_id}")
/// print(f"Propositions: {result.proposition_count}")
/// print(f"Entities: {result.entity_count}")
/// print(f"Edges: {result.edge_count}")
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PyIngestResult {
    /// The ID of the created Document node.
    #[pyo3(get)]
    document_id: PyNodeId,
    /// Number of proposition nodes created.
    #[pyo3(get)]
    proposition_count: usize,
    /// Number of entity nodes created (excludes deduplicated/reused entities).
    #[pyo3(get)]
    entity_count: usize,
    /// Total edges wired (CONTAINS + MENTIONS).
    #[pyo3(get)]
    edge_count: usize,
}

impl PyIngestResult {
    /// Create a PyIngestResult from a spider-core IngestResult.
    pub fn from_rust(result: IngestResultInner) -> Self {
        PyIngestResult {
            document_id: PyNodeId::from(result.document_id.get()),
            proposition_count: result.proposition_count,
            entity_count: result.entity_count,
            edge_count: result.edge_count,
        }
    }
}

impl From<IngestResultInner> for PyIngestResult {
    fn from(result: IngestResultInner) -> Self {
        Self::from_rust(result)
    }
}

#[pymethods]
impl PyIngestResult {
    /// Return a debug representation.
    fn __repr__(&self) -> String {
        format!(
            "IngestResult(document_id=NodeId({}), propositions={}, entities={}, edges={})",
            self.document_id.inner(),
            self.proposition_count,
            self.entity_count,
            self.edge_count
        )
    }
}
