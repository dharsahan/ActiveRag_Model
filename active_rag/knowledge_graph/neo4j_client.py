from neo4j import GraphDatabase
import logging
from typing import Dict, List, Any, Optional

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
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            raise

    def is_connected(self) -> bool:
        if not self._driver:
            return False
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            return True
        except:
            return False

    def create_entity(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        query = f"CREATE (n:{label} $props) RETURN n"
        with self._driver.session() as session:
            result = session.run(query, props=properties)
            record = result.single()
            return dict(record["n"])

    def close(self):
        if self._driver:
            self._driver.close()