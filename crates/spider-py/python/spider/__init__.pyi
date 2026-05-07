"""
Python bindings for Spider — bio-inspired AI agent memory graph.

This module provides a Pythonic interface to the spider-core database engine
with automatic GIL release during I/O operations.
"""

from typing import overload


# ============================================================================
# Exception Hierarchy
# ============================================================================

class SpiderError(Exception):
    """Base exception for all Spider database errors."""
    pass


class SpiderNotFoundError(SpiderError):
    """Raised when a node, edge, or blob is not found."""
    pass


class SpiderCorruptError(SpiderError):
    """Raised when database files are corrupt."""
    pass


class SpiderIOError(SpiderError):
    """Raised on file I/O errors."""
    pass


class SpiderIngestionError(SpiderError):
    """Raised on ingestion failures."""
    pass


class SpiderTraversalError(SpiderError):
    """Raised when traversal depth limits are exceeded."""
    pass


# ============================================================================
# Value Types
# ============================================================================

class NodeId:
    """A unique identifier for a node in the Spider graph database.

    NodeIds are positive integers (1-based). The value 0 is reserved
    as a sentinel and is not a valid NodeId.
    """

    def __init__(self, raw: int) -> None: ...
    def __int__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def as_int(self) -> int: ...


class EdgeId:
    """A unique identifier for an edge in the Spider graph database.

    EdgeIds are positive integers (1-based). The value 0 is reserved
    as a sentinel and is not a valid EdgeId.
    """

    def __init__(self, raw: int) -> None: ...
    def __int__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def as_int(self) -> int: ...


class Neighbor:
    """A neighbor node returned by graph traversal.

    Contains the neighbor's NodeId and the EdgeId that connects to it.
    """

    @property
    def node_id(self) -> NodeId: ...
    @property
    def edge_id(self) -> EdgeId: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...


# ============================================================================
# Enums
# ============================================================================

class Direction:
    """Traversal direction for edge queries.

    Variants:
        OUTGOING: Edges where the queried node is the source.
        INCOMING: Edges where the queried node is the target.
        BOTH: All edges connected to the queried node.
    """

    OUTGOING: Direction
    INCOMING: Direction
    BOTH: Direction

    @staticmethod
    def from_str(value: str) -> Direction: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


class BioTier:
    """Bio-inspired storage tier classification for nodes.

    Tiers classify nodes by their vitality score:
        HOT: score > 20.0 (in RAM, instant access)
        WARM: score > 5.0 (on SSD, fast I/O)
        COLD: score > 0.0 (archived, slow access)
        PRUNED: score <= 0.0 (eligible for deletion)
    """

    HOT: BioTier
    WARM: BioTier
    COLD: BioTier
    PRUNED: BioTier

    @staticmethod
    def from_score(score: float) -> BioTier: ...
    def is_prunable(self) -> bool: ...
    def is_active(self) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: BioTier) -> bool: ...
    def __le__(self, other: BioTier) -> bool: ...
    def __gt__(self, other: BioTier) -> bool: ...
    def __ge__(self, other: BioTier) -> bool: ...
    def __hash__(self) -> int: ...


# ============================================================================
# Ingestion Types
# ============================================================================

class Entity:
    """A named entity mentioned in a proposition.

    Entities represent real-world concepts like people, places, or organizations.
    During ingestion, entities with the same name are automatically deduplicated.
    """

    def __init__(self, name: str, entity_type: str) -> None: ...

    @property
    def name(self) -> str: ...
    @property
    def entity_type(self) -> str: ...
    def __repr__(self) -> str: ...


class Proposition:
    """A factual statement with associated entities.

    Propositions represent atomic facts extracted from documents.
    """

    def __init__(
        self,
        text: str,
        entities: list[Entity] | None = None,
    ) -> None: ...

    @property
    def text(self) -> str: ...
    @property
    def entities(self) -> list[Entity]: ...
    def __repr__(self) -> str: ...


class IngestRequest:
    """A request to ingest a document with propositions into the Spider database.

    This is the primary input to `Spider.index()`.
    """

    def __init__(
        self,
        title: str,
        propositions: list[Proposition] | None = None,
    ) -> None: ...

    @property
    def title(self) -> str: ...
    @property
    def propositions(self) -> list[Proposition]: ...
    def __repr__(self) -> str: ...


class IngestResult:
    """The result of an ingestion operation.

    Returned by `Spider.index()`. Contains the ID of the created document
    node and counts of nodes/edges created.
    """

    @property
    def document_id(self) -> NodeId: ...
    @property
    def proposition_count(self) -> int: ...
    @property
    def entity_count(self) -> int: ...
    @property
    def edge_count(self) -> int: ...
    def __repr__(self) -> str: ...


# ============================================================================
# Spider Database Handle
# ============================================================================

class Spider:
    """Python wrapper for the Spider database handle.

    Provides a Pythonic interface to the spider-core database engine
    with automatic GIL release during I/O operations and context manager support.

    Example:
        >>> import spider
        >>> with spider.Spider.open("/tmp/my_db") as db:
        ...     result = db.index(spider.IngestRequest("My Doc"))
        ...     print(result.document_id)
    """

    @classmethod
    def open(cls, path: str) -> Spider:
        """Open or create a Spider database at the given path.

        Args:
            path: Filesystem path to the database directory.
                  The directory will be created if it doesn't exist.

        Returns:
            A new Spider database handle.

        Raises:
            SpiderIOError: If the database cannot be opened or created.
            SpiderCorruptError: If the database metadata is corrupt.
        """
        ...

    @classmethod
    def open_default(cls) -> Spider:
        """Open or create a Spider database at the platform-default location.

        Default paths:
            Linux: ~/.local/share/spider/default/
            macOS: ~/Library/Application Support/spider/default/
            Windows: %APPDATA%\\spider\\default\\

        Returns:
            A new Spider database handle.

        Raises:
            SpiderIOError: If the database cannot be opened or created.
            SpiderCorruptError: If the database metadata is corrupt.
        """
        ...

    def close(self) -> None:
        """Gracefully close the database, flushing all data to disk.

        This method is idempotent -- safe to call multiple times.
        """
        ...

    @property
    def path(self) -> str:
        """The filesystem path to the database directory."""
        ...

    def __enter__(self) -> Spider: ...
    def __exit__(
        self,
        exc_type: object | None,
        exc_value: object | None,
        traceback: object | None,
    ) -> None: ...

    def __repr__(self) -> str: ...

    # =========================================================================
    # Ingestion
    # =========================================================================

    def index(self, request: IngestRequest) -> IngestResult:
        """Ingest a document with propositions into the database.

        Creates a Document node with the given title, proposition nodes for each
        proposition, entity nodes (deduplicated by name), and wires CONTAINS +
        MENTIONS edges between them.

        Args:
            request: An IngestRequest containing the document title and propositions.

        Returns:
            IngestResult with the created document ID and counts of nodes/edges.

        Raises:
            SpiderIngestionError: If ingestion produces zero propositions.
            SpiderNotFoundError: If a referenced node is not found.
            SpiderIOError: If a file I/O error occurs.
        """
        ...

    # =========================================================================
    # Find Queries
    # =========================================================================

    def find_by_label(self, label: str) -> list[NodeId]:
        """Find all nodes with a given label.

        Performs a sequential scan over all nodes, checking if each live node
        has the given label. Returns an empty list if the label has never been used.

        Args:
            label: The label string to search for (e.g. "DOCUMENT", "ENTITY").

        Returns:
            A list of NodeId objects with the given label.

        Raises:
            SpiderIOError: If a file I/O error occurs during the scan.
            SpiderTraversalError: If a property chain exceeds the depth limit.
        """
        ...

    def find_by_property(self, key: str, value: str) -> list[NodeId]:
        """Find all nodes with a property matching the given key and value.

        Performs a sequential scan over all nodes, checking each node's property
        chain for a matching key/value pair. Only matches inline short strings (<=6 bytes).

        Args:
            key: The property key (e.g. "name").
            value: The property value to match (e.g. "Mumbai").

        Returns:
            A list of NodeId objects with the matching property.
            Returns an empty list if the key has never been used or no matches are found.

        Raises:
            SpiderIOError: If a file I/O error occurs during the scan.
            SpiderTraversalError: If a property chain exceeds the depth limit.
        """
        ...

    def find_one_by_property(self, key: str, value: str) -> NodeId | None:
        """Find the first node with a property matching the given key and value.

        Like `find_by_property`, but short-circuits on the first match.

        Args:
            key: The property key (e.g. "name").
            value: The property value to match (e.g. "Mumbai").

        Returns:
            The first matching NodeId, or None if no match found.

        Raises:
            SpiderIOError: If a file I/O error occurs during the scan.
            SpiderTraversalError: If a property chain exceeds the depth limit.
        """
        ...

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    @overload
    def get_neighbors(
        self,
        node_id: NodeId,
        direction: Direction = ...,
    ) -> list[Neighbor]: ...

    @overload
    def get_neighbors(
        self,
        node_id: NodeId,
        direction: str,
    ) -> list[Neighbor]: ...

    def get_neighbors(
        self,
        node_id: NodeId,
        direction: Direction | str = ...,
    ) -> list[Neighbor]:
        """Get all neighbor nodes connected to the given node.

        Walks the edge chain from the specified node and returns all neighbors
        in the given direction.

        Args:
            node_id: The NodeId to find neighbors for.
            direction: Traversal direction (OUTGOING, INCOMING, or BOTH).
                       Can also be a string: "outgoing", "incoming", "both".

        Returns:
            A list of Neighbor objects, each with node_id and edge_id.

        Raises:
            SpiderNotFoundError: If the node does not exist.
            SpiderTraversalError: If the edge chain exceeds the depth limit.
        """
        ...

    @overload
    def get_relationships(
        self,
        node_id: NodeId,
        direction: Direction = ...,
    ) -> list[dict]: ...

    @overload
    def get_relationships(
        self,
        node_id: NodeId,
        direction: str,
    ) -> list[dict]: ...

    def get_relationships(
        self,
        node_id: NodeId,
        direction: Direction | str = ...,
    ) -> list[dict]:
        """Get all relationships (edges) connected to the given node.

        Args:
            node_id: The NodeId to find relationships for.
            direction: Traversal direction (OUTGOING, INCOMING, or BOTH).
                       Can also be a string: "outgoing", "incoming", "both".

        Returns:
            A list of dicts with keys: source_id, target_id.

        Raises:
            SpiderNotFoundError: If the node does not exist.
            SpiderTraversalError: If the edge chain exceeds the depth limit.
        """
        ...

    @overload
    def count_relationships(
        self,
        node_id: NodeId,
        direction: Direction = ...,
    ) -> int: ...

    @overload
    def count_relationships(
        self,
        node_id: NodeId,
        direction: str,
    ) -> int: ...

    def count_relationships(
        self,
        node_id: NodeId,
        direction: Direction | str = ...,
    ) -> int:
        """Count the number of relationships connected to the given node.

        More efficient than `get_relationships` when you only need the count.

        Args:
            node_id: The NodeId to count relationships for.
            direction: Traversal direction (OUTGOING, INCOMING, or BOTH).
                       Can also be a string: "outgoing", "incoming", "both".

        Returns:
            The number of relationships.

        Raises:
            SpiderNotFoundError: If the node does not exist.
            SpiderTraversalError: If the edge chain exceeds the depth limit.
        """
        ...

    # =========================================================================
    # Bio Scoring
    # =========================================================================

    def get_bio_score(self, node_id: NodeId) -> float:
        """Calculate the bio-inspired vitality score for a node.

        The bio score reflects a node's "memory strength" based on:
        - Significance: Higher significance increases the score.
        - Access frequency: More accesses increase the score (logarithmic).
        - Recency: Older nodes have decaying scores (gravitational decay).

        Args:
            node_id: The NodeId to calculate the score for.

        Returns:
            The bio vitality score (positive number for live nodes).

        Raises:
            SpiderNotFoundError: If the node does not exist.
            SpiderIOError: If a file I/O error occurs.
        """
        ...

    def get_bio_tier(self, node_id: NodeId) -> BioTier:
        """Get the bio storage tier for a node.

        Tiers classify nodes by their vitality score:
            HOT: score > 20.0 (in RAM, instant access)
            WARM: score > 5.0 (on SSD, fast I/O)
            COLD: score > 0.0 (archived, slow access)
            PRUNED: score <= 0.0 (eligible for deletion)

        Args:
            node_id: The NodeId to classify.

        Returns:
            The storage tier (Hot, Warm, Cold, or Pruned).

        Raises:
            SpiderNotFoundError: If the node does not exist.
            SpiderIOError: If a file I/O error occurs.
        """
        ...

    # =========================================================================
    # Node Operations
    # =========================================================================

    def node_count(self) -> int:
        """Get the total number of node slots in the database.

        Returns `metadata.next_node_id - 1`, which is the count of all node
        slots ever created (including deleted ones).

        Returns:
            The number of node slots (live + deleted).
        """
        ...

    def node_touch(self, node_id: NodeId) -> int:
        """Touch a node, incrementing its access count and updating its last
        accessed timestamp.

        This increases the node's bio vitality score by refreshing its
        `last_accessed_at` to the current time and incrementing `access_count`.

        Args:
            node_id: The NodeId to touch.

        Returns:
            The new access count after incrementing.

        Raises:
            SpiderNotFoundError: If the node does not exist.
            SpiderIOError: If a file I/O error occurs.
        """
        ...

    def set_significance(self, node_id: NodeId, significance: int) -> None:
        """Set the significance value for a node.

        Significance affects the bio vitality score — higher significance
        means a higher score. Valid range is 0-255.

        Args:
            node_id: The NodeId to update.
            significance: The new significance value (0-255).

        Raises:
            SpiderNotFoundError: If the node does not exist.
            SpiderIOError: If a file I/O error occurs.
        """
        ...
