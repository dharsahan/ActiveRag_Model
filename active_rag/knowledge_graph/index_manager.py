"""Neo4j index and constraint optimization manager.

Creates composite and full-text indexes for common query patterns
to improve graph query performance.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


# Index definitions: (name, cypher_statement)
_INDEXES = [
    # Composite indexes for common lookups
    ("idx_person_name", "CREATE INDEX idx_person_name IF NOT EXISTS FOR (p:Person) ON (p.name)"),
    ("idx_org_name", "CREATE INDEX idx_org_name IF NOT EXISTS FOR (o:Organization) ON (o.name)"),
    ("idx_concept_name", "CREATE INDEX idx_concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)"),
    ("idx_component_name", "CREATE INDEX idx_component_name IF NOT EXISTS FOR (c:Component) ON (c.name)"),
    ("idx_document_title", "CREATE INDEX idx_document_title IF NOT EXISTS FOR (d:Document) ON (d.title)"),
    ("idx_process_name", "CREATE INDEX idx_process_name IF NOT EXISTS FOR (p:Process) ON (p.name)"),
    # Domain-based lookups
    ("idx_document_domain", "CREATE INDEX idx_document_domain IF NOT EXISTS FOR (d:Document) ON (d.domain)"),
    ("idx_org_type", "CREATE INDEX idx_org_type IF NOT EXISTS FOR (o:Organization) ON (o.type)"),
]

_FULLTEXT_INDEXES = [
    # Full-text search indexes for entity name search
    (
        "ft_entity_names",
        "CREATE FULLTEXT INDEX ft_entity_names IF NOT EXISTS "
        "FOR (n:Person|Organization|Concept|Component|Process) ON EACH [n.name]",
    ),
]


class IndexManager:
    """Manages Neo4j indexes and constraints for query optimization."""

    def __init__(self, neo4j_client) -> None:
        self._client = neo4j_client

    def ensure_indexes(self) -> dict:
        """Create all indexes if they don't exist.

        Returns:
            Dict with counts of created and skipped indexes.
        """
        results = {"created": 0, "skipped": 0, "errors": []}

        for name, cypher in _INDEXES + _FULLTEXT_INDEXES:
            try:
                with self._client._driver.session() as session:
                    session.run(cypher)
                results["created"] += 1
                logger.info(f"Index ensured: {name}")
            except Exception as e:
                error_str = str(e)
                if "already exists" in error_str or "An equivalent" in error_str:
                    results["skipped"] += 1
                else:
                    results["errors"].append(f"{name}: {error_str}")
                    logger.warning(f"Failed to create index {name}: {e}")

        logger.info(
            f"Index setup complete: {results['created']} created, "
            f"{results['skipped']} skipped, {len(results['errors'])} errors"
        )
        return results

    def list_indexes(self) -> List[dict]:
        """List all existing indexes in the database.

        Returns:
            List of index info dicts.
        """
        query = "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state"
        indexes = []
        try:
            with self._client._driver.session() as session:
                result = session.run(query)
                for record in result:
                    indexes.append({
                        "name": record["name"],
                        "type": record["type"],
                        "labels": record["labelsOrTypes"],
                        "properties": record["properties"],
                        "state": record["state"],
                    })
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
        return indexes

    def drop_index(self, name: str) -> bool:
        """Drop an index by name.

        Args:
            name: Index name to drop

        Returns:
            True if dropped successfully
        """
        try:
            with self._client._driver.session() as session:
                session.run(f"DROP INDEX {name} IF EXISTS")
            logger.info(f"Dropped index: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop index {name}: {e}")
            return False
