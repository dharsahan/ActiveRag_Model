from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, DatabaseError, AuthError
import logging
import re
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self._driver = None
        self._connect()

    def _connect(self):
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self._driver.session() as session:
                session.run("RETURN 1")
            logging.info("Connected to Neo4j database")
        except (ServiceUnavailable, AuthError, DatabaseError) as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error connecting to Neo4j: {e}")
            raise

    def is_connected(self) -> bool:
        """Check if the Neo4j database connection is active."""
        if not self._driver:
            return False
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            return True
        except (ServiceUnavailable, DatabaseError, AuthError) as e:
            logging.warning(f"Neo4j connection check failed: {e}")
            return False
        except Exception as e:
            logging.warning(f"Unexpected error during connection check: {e}")
            return False

    def create_entity(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new entity in the Neo4j database.

        Args:
            label: The node label (must be alphanumeric with underscores only)
            properties: Dictionary of properties to set on the node

        Returns:
            Dictionary representation of the created node

        Raises:
            ValueError: If label name is invalid
            RuntimeError: If entity creation fails
            ServiceUnavailable: If Neo4j is not available
        """
        # Validate label name to prevent injection (alphanumeric + underscore only, must start with letter/underscore)
        if not label or not isinstance(label, str):
            raise ValueError("Label must be a non-empty string")

        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', label):
            raise ValueError(f"Invalid label name: '{label}'. Label must start with a letter or underscore and contain only alphanumeric characters and underscores.")

        # Validate properties
        if not isinstance(properties, dict):
            raise ValueError("Properties must be a dictionary")

        # Use MERGE to avoid duplicates and update existing nodes
        # We assume the properties dict contains an 'id' key for matching
        if "id" in properties:
            query = f"""
            MERGE (n:{label} {{id: $props.id}})
            SET n += $props
            RETURN n
            """
        else:
            query = f"CREATE (n:{label} $props) RETURN n"

        try:
            with self._driver.session() as session:
                result = session.run(query, props=properties)
                record = result.single()
                if record is None:
                    raise RuntimeError("Failed to create/merge entity - no record returned")
                return dict(record["n"])
        except (ServiceUnavailable, DatabaseError, AuthError) as e:
            logging.error(f"Database error creating entity '{label}': {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error creating entity '{label}': {e}")
            raise RuntimeError(f"Failed to create entity '{label}': {e}") from e

    def create_relationship(
        self, 
        subject_id: str, 
        subject_label: str, 
        predicate: str, 
        object_id: str, 
        object_label: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a relationship between two nodes.
        
        Args:
            subject_id: ID of the starting node
            subject_label: Label of the starting node
            predicate: Type of the relationship (e.g., 'AUTHORED')
            object_id: ID of the ending node
            object_label: Label of the ending node
            properties: Optional relationship properties
            
        Returns:
            True if relationship created or already existed
        """
        # Validate inputs to prevent injection
        for label in [subject_label, object_label, predicate]:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', label):
                raise ValueError(f"Invalid label or predicate: '{label}'")

        query = f"""
        MATCH (s:{subject_label} {{id: $s_id}})
        MATCH (o:{object_label} {{id: $o_id}})
        MERGE (s)-[r:{predicate}]->(o)
        SET r += $props
        RETURN type(r)
        """
        
        try:
            with self._driver.session() as session:
                result = session.run(query, s_id=subject_id, o_id=object_id, props=properties or {})
                return result.single() is not None
        except Exception as e:
            logging.error(f"Failed to create relationship: {e}")
            return False

    def clear_all_data(self) -> bool:
        """
        Delete ALL nodes and relationships from the database.
        USE WITH EXTREME CAUTION.
        """
        query = "MATCH (n) DETACH DELETE n"
        try:
            with self._driver.session() as session:
                session.run(query)
                logging.info("Neo4j database cleared successfully.")
                return True
        except Exception as e:
            logging.error(f"Failed to clear Neo4j database: {e}")
            return False

    def close(self):
        """Close the Neo4j database connection."""
        if self._driver:
            try:
                self._driver.close()
                logging.info("Neo4j connection closed")
            except Exception as e:
                logging.warning(f"Error closing Neo4j connection: {e}")