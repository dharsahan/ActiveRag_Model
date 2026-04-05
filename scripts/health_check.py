#!/usr/bin/env python3
"""Quick health check for Phase 1 system"""

import os
import sys
import importlib.util
from pathlib import Path

def check_environment():
    """Check basic environment setup"""
    print("🔍 Environment Check")

    # Check Python packages
    packages = ['spacy', 'neo4j', 'chromadb']
    for package in packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec:
                print(f"  ✅ {package}: Available")
            else:
                print(f"  ❌ {package}: Missing")
        except ImportError:
            print(f"  ❌ {package}: Import error")

    # Check spaCy model
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print(f"  ✅ spaCy model: Available")
    except OSError:
        print(f"  ❌ spaCy model: Missing (run: python -m spacy download en_core_web_sm)")
    except Exception as e:
        print(f"  ⚠️ spaCy model: Error - {e}")

    # Check environment file
    env_file = Path('.env')
    if env_file.exists():
        print(f"  ✅ .env file: Present")
    else:
        print(f"  ⚠️ .env file: Missing")

    # Check active_rag module
    try:
        spec = importlib.util.find_spec('active_rag.config')
        if spec:
            print(f"  ✅ active_rag.config: Available")
        else:
            print(f"  ❌ active_rag.config: Missing")
    except Exception as e:
        print(f"  ❌ active_rag.config: Error - {e}")

    print()

def check_neo4j():
    """Check Neo4j connectivity"""
    print("🔍 Neo4j Check")

    # Check HTTP endpoint
    try:
        import requests
        response = requests.get('http://localhost:7474', timeout=5)
        if response.status_code == 200:
            print(f"  ✅ Neo4j HTTP: Available (http://localhost:7474)")
        else:
            print(f"  ❌ Neo4j HTTP: Status {response.status_code}")
    except ImportError:
        print(f"  ⚠️ Neo4j HTTP: requests module not available")
    except Exception as e:
        print(f"  ❌ Neo4j HTTP: Not reachable - {e}")

    # Check Bolt connection
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687",
                                     auth=("neo4j", "activerag123"))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            if result.single():
                print(f"  ✅ Neo4j Bolt: Connected")
            else:
                print(f"  ❌ Neo4j Bolt: Query failed")
        driver.close()
    except ImportError:
        print(f"  ❌ Neo4j Bolt: neo4j module not available")
    except Exception as e:
        print(f"  ❌ Neo4j Bolt: {str(e)}")

    print()

def check_chromadb():
    """Check ChromaDB setup"""
    print("🔍 ChromaDB Check")

    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("test_collection")
        print(f"  ✅ ChromaDB: Available")

        # Clean up test collection
        client.delete_collection("test_collection")
    except ImportError:
        print(f"  ❌ ChromaDB: Module not available")
    except Exception as e:
        print(f"  ❌ ChromaDB: {str(e)}")

    print()

def check_docker():
    """Check Docker availability"""
    print("🔍 Docker Check")

    try:
        import subprocess
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Docker: Available")

            # Check if Neo4j container is running
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            if 'neo4j' in result.stdout:
                print(f"  ✅ Neo4j container: Running")
            else:
                print(f"  ⚠️ Neo4j container: Not running")
        else:
            print(f"  ❌ Docker: Not available")
    except Exception as e:
        print(f"  ❌ Docker: {str(e)}")

    print()

if __name__ == "__main__":
    print("🚀 Phase 1 System Health Check\n")
    check_environment()
    check_docker()
    check_neo4j()
    check_chromadb()
    print("Health check complete!")