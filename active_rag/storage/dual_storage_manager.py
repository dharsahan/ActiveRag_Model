"""
Dual storage manager for synchronizing writes between ChromaDB (vector) and Neo4j (graph) databases.

This module provides the core integration that enables hybrid vector-graph retrieval by:
1. Storing documents in ChromaDB for vector similarity search
2. Extracting entities using the NLP pipeline
3. Storing entities and relationships in Neo4j for graph traversal
4. Creating MENTIONS relationships between documents and extracted entities
"""

import logging
from typing import Dict, List, Any, Optional
import hashlib
import chromadb
from chromadb.config import Settings

from ..config import Config
from ..knowledge_graph.neo4j_client import Neo4jClient
from ..knowledge_graph.schema_manager import SchemaManager
from ..nlp_pipeline.entity_extractor import EntityExtractor
from ..nlp_pipeline.document_classifier import DocumentClassifier
from ..schemas.entities import ContentDomain

logger = logging.getLogger(__name__)


class DualStorageManager:
    """Manages storage operations across both ChromaDB and Neo4j with entity extraction"""

    def __init__(self, config: Config):
        """Initialize dual storage manager with ChromaDB and optional Neo4j support

        Args:
            config: System configuration with database settings
        """
        self.config = config

        # Initialize ChromaDB client (existing vector storage)
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=config.chroma_persist_dir,
                settings=Settings(allow_reset=True)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=config.collection_name
            )
            logger.info(f"ChromaDB initialized at {config.chroma_persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

        # Initialize Neo4j client and related components (new graph storage)
        if config.enable_graph_features:
            try:
                self.neo4j_client = Neo4jClient(
                    config.neo4j_uri,
                    config.neo4j_username,
                    config.neo4j_password
                )
                self.schema_manager = SchemaManager(self.neo4j_client)
                self.schema_manager.create_base_constraints()
                logger.info(f"Neo4j client initialized at {config.neo4j_uri}")
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j client: {e}")
                self.neo4j_client = None
                self.schema_manager = None
        else:
            self.neo4j_client = None
            self.schema_manager = None
            logger.info("Graph features disabled - ChromaDB only mode")

        # Initialize NLP components for entity extraction
        self.entity_extractor = EntityExtractor()
        self.document_classifier = DocumentClassifier()

        logger.info(f"DualStorageManager initialized - Graph features: {config.enable_graph_features}")

    def store_document(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store document in both ChromaDB and Neo4j with entity extraction

        Args:
            doc_data: Document data with keys: title, content, url (optional), domain (optional)

        Returns:
            Dictionary with storage results and extracted entities
        """
        doc_id = self._generate_doc_id(doc_data)
        result = {
            "doc_id": doc_id,
            "chroma_stored": False,
            "graph_stored": False,
            "entities_extracted": [],
            "relationships_created": []
        }

        # Classify document content domain if not provided
        content = doc_data.get("content", "")
        domain = doc_data.get("domain")
        if not domain:
            domain = self.document_classifier.classify_document(content)

        # Store in ChromaDB (existing vector storage)
        try:
            self.collection.add(
                documents=[content],
                metadatas=[{
                    "title": doc_data.get("title", ""),
                    "url": doc_data.get("url", ""),
                    "domain": domain.value if hasattr(domain, 'value') else str(domain),
                    "doc_id": doc_id
                }],
                ids=[doc_id]
            )
            result["chroma_stored"] = True
            logger.info(f"Document {doc_id} stored in ChromaDB")
        except Exception as e:
            logger.error(f"Failed to store document in ChromaDB: {e}")
            result["chroma_stored"] = False

        # Store in Neo4j if graph features enabled
        if self.config.enable_graph_features and self.neo4j_client and self.schema_manager:
            try:
                # Extract entities from document content
                entities = self.entity_extractor.extract_entities(content, domain)
                result["entities_extracted"] = entities

                # Create document node
                doc_node = self._create_document_node(doc_data, doc_id, domain)

                # Store entities and create relationships
                for entity in entities:
                    if self.schema_manager.validate_entity(entity["label"], entity["properties"]):
                        # Create or merge entity
                        entity_node = self._create_or_merge_entity(entity)

                        # Create relationship from document to entity (MENTIONS)
                        rel_created = self._create_document_entity_relationship(
                            doc_node, entity_node, entity["label"]
                        )
                        if rel_created:
                            result["relationships_created"].append({
                                "from": doc_id,
                                "to": entity["properties"]["id"],
                                "type": "MENTIONS"
                            })
                    else:
                        logger.warning(f"Entity validation failed for {entity['label']}: {entity['properties']}")

                result["graph_stored"] = True
                logger.info(f"Document {doc_id} stored in Neo4j with {len(entities)} entities")

            except Exception as e:
                logger.error(f"Failed to store document in Neo4j: {e}")
                result["graph_stored"] = False

        return result

    def _generate_doc_id(self, doc_data: Dict[str, Any]) -> str:
        """Generate consistent document ID based on URL or content hash

        Args:
            doc_data: Document data dictionary

        Returns:
            Consistent document identifier
        """
        content = doc_data.get("content", "")
        url = doc_data.get("url", "")
        title = doc_data.get("title", "")

        # Use URL if available for consistency, otherwise hash content + title
        if url:
            return hashlib.md5(url.encode()).hexdigest()
        else:
            # Use title + first 200 chars of content for consistent hashing
            combined = f"{title}_{content[:200]}"
            return hashlib.md5(combined.encode()).hexdigest()

    def _create_document_node(self, doc_data: Dict[str, Any], doc_id: str, domain: ContentDomain) -> Dict[str, Any]:
        """Create document node in Neo4j graph

        Args:
            doc_data: Document data
            doc_id: Generated document ID
            domain: Content domain classification

        Returns:
            Created document node properties
        """
        doc_properties = {
            "id": doc_id,
            "title": doc_data.get("title", ""),
            "url": doc_data.get("url", ""),
            "domain": domain.value if hasattr(domain, 'value') else str(domain),
            "content_hash": doc_id,
            "created_at": doc_data.get("created_at", "")
        }

        return self.neo4j_client.create_entity("Document", doc_properties)

    def _create_or_merge_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Create entity or merge if it already exists

        Args:
            entity: Entity data with label and properties

        Returns:
            Entity node properties (existing or newly created)
        """
        entity_id = entity["properties"]["id"]
        label = entity["label"]

        # Check if entity already exists
        query = f"MATCH (n:{label} {{id: $id}}) RETURN n"
        with self.neo4j_client._driver.session() as session:
            result = session.run(query, id=entity_id)
            existing = result.single()

            if existing:
                # Entity exists, return its properties
                return dict(existing["n"])
            else:
                # Create new entity
                return self.neo4j_client.create_entity(label, entity["properties"])

    def _create_document_entity_relationship(self, doc_node: Dict[str, Any], entity_node: Dict[str, Any], entity_label: str) -> bool:
        """Create MENTIONS relationship from document to entity

        Args:
            doc_node: Document node properties
            entity_node: Entity node properties
            entity_label: Entity type label

        Returns:
            True if relationship created successfully
        """
        try:
            # Use parameterized query to avoid Cypher injection
            query = f"""
            MATCH (d:Document {{id: $doc_id}})
            MATCH (e:{entity_label} {{id: $entity_id}})
            MERGE (d)-[:MENTIONS]->(e)
            RETURN d, e
            """

            with self.neo4j_client._driver.session() as session:
                result = session.run(query,
                    doc_id=doc_node["id"],
                    entity_id=entity_node["id"]
                )
                return result.single() is not None

        except Exception as e:
            logger.error(f"Failed to create document-entity relationship: {e}")
            return False

    def get_document_entities(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all entities mentioned in a specific document

        Args:
            doc_id: Document identifier

        Returns:
            List of entity data with labels
        """
        if not self.config.enable_graph_features or not self.neo4j_client:
            return []

        query = """
        MATCH (d:Document {id: $doc_id})-[:MENTIONS]->(e)
        RETURN e, labels(e) as entity_labels
        """

        entities = []
        try:
            with self.neo4j_client._driver.session() as session:
                result = session.run(query, doc_id=doc_id)
                for record in result:
                    entity_data = dict(record["e"])
                    entity_data["labels"] = record["entity_labels"]
                    entities.append(entity_data)
        except Exception as e:
            logger.error(f"Failed to retrieve document entities for {doc_id}: {e}")

        return entities

    def close(self):
        """Close database connections gracefully"""
        if self.neo4j_client:
            try:
                self.neo4j_client.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {e}")

        # ChromaDB client doesn't require explicit closing
        logger.info("DualStorageManager connections closed")