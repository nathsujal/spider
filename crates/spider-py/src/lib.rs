use pyo3::prelude::*;
use pyo3::exceptions::PyException;

mod error;
mod spider_handle;
mod types;
mod ingest;

// Define Python exception classes
pyo3::create_exception!(spider, SpiderError, PyException);
pyo3::create_exception!(spider, SpiderNotFoundError, SpiderError);
pyo3::create_exception!(spider, SpiderCorruptError, SpiderError);
pyo3::create_exception!(spider, SpiderIOError, SpiderError);
pyo3::create_exception!(spider, SpiderIngestionError, SpiderError);
pyo3::create_exception!(spider, SpiderTraversalError, SpiderError);

#[pymodule]
fn spider(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register exception classes
    m.add("SpiderError", m.py().get_type_bound::<SpiderError>())?;
    m.add("SpiderNotFoundError", m.py().get_type_bound::<SpiderNotFoundError>())?;
    m.add("SpiderCorruptError", m.py().get_type_bound::<SpiderCorruptError>())?;
    m.add("SpiderIOError", m.py().get_type_bound::<SpiderIOError>())?;
    m.add("SpiderIngestionError", m.py().get_type_bound::<SpiderIngestionError>())?;
    m.add("SpiderTraversalError", m.py().get_type_bound::<SpiderTraversalError>())?;

    // Register classes
    m.add_class::<spider_handle::PySpider>()?;
    m.add_class::<types::PyNodeId>()?;
    m.add_class::<types::PyEdgeId>()?;
    m.add_class::<types::PyNeighbor>()?;
    m.add_class::<types::PyDirection>()?;
    m.add_class::<types::PyBioTier>()?;
    m.add_class::<ingest::PyEntity>()?;
    m.add_class::<ingest::PyProposition>()?;
    m.add_class::<ingest::PyIngestRequest>()?;
    m.add_class::<ingest::PyIngestResult>()?;

    Ok(())
}
