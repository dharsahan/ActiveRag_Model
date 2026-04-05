# Phase 1 Troubleshooting Guide

This guide provides solutions for common issues encountered when setting up and using the Phase 1 hybrid vector-graph RAG system.

## Table of Contents
- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Neo4j Connection Problems](#neo4j-connection-problems)
- [ChromaDB Issues](#chromadb-issues)
- [NLP Pipeline Problems](#nlp-pipeline-problems)
- [Document Processing Errors](#document-processing-errors)
- [Graph Operations Issues](#graph-operations-issues)
- [Performance Problems](#performance-problems)
- [Integration Test Failures](#integration-test-failures)
- [Environment Configuration](#environment-configuration)
- [Common Error Messages](#common-error-messages)

## Quick Diagnostics

### Health Check Script

Run this quick diagnostic to identify the most common issues:

```python
#!/usr/bin/env python3
"""Quick health check for Phase 1 system"""

import os
import requests
import importlib.util
from pathlib import Path

def check_environment():
    """Check basic environment setup"""
    print("🔍 Environment Check")
    
    # Check Python packages
    packages = ['spacy', 'neo4j', 'chromadb', 'active_rag']
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
    
    # Check environment file
    env_file = Path('.env')
    if env_file.exists():
        print(f"  ✅ .env file: Present")
    else:
        print(f"  ⚠️ .env file: Missing")
    
    print()

def check_neo4j():
    """Check Neo4j connectivity"""
    print("🔍 Neo4j Check")
    
    # Check HTTP endpoint
    try:
        response = requests.get('http://localhost:7474', timeout=5)
        if response.status_code == 200:
            print(f"  ✅ Neo4j HTTP: Available (http://localhost:7474)")
        else:
            print(f"  ❌ Neo4j HTTP: Status {response.status_code}")
    except requests.RequestException:
        print(f"  ❌ Neo4j HTTP: Not reachable")
    
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
    except Exception as e:
        print(f"  ❌ ChromaDB: {str(e)}")
    
    print()

if __name__ == "__main__":
    print("🚀 Phase 1 System Health Check\n")
    check_environment()
    check_neo4j()
    check_chromadb()
    print("Health check complete!")
```

Save this as `scripts/health_check.py` and run: `python scripts/health_check.py`

## Installation Issues

### Python Dependencies

**Problem**: Package installation fails
```bash
ERROR: Could not find a version that satisfies the requirement X
```

**Solution**:
```bash
# Update pip first
python -m pip install --upgrade pip

# Install with verbose output to see specific errors
python -m pip install -r requirements.txt -v

# Try installing packages individually
python -m pip install chromadb
python -m pip install neo4j
python -m pip install spacy
```

**Problem**: spaCy model download fails
```bash
Can't find model 'en_core_web_sm'
```

**Solutions**:
```bash
# Method 1: Direct download
python -m spacy download en_core_web_sm

# Method 2: Manual installation
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Method 3: Alternative model
python -m spacy download en_core_web_md  # Larger but more accurate
```

**Problem**: Permission denied during installation
```bash
PermissionError: [Errno 13] Permission denied
```

**Solution**:
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Or install for user only
pip install --user -r requirements.txt
```

### Docker Issues

**Problem**: Docker not found
```bash
docker: command not found
```

**Solution**:
- Install Docker Desktop for your operating system
- Linux: `sudo apt-get install docker.io docker-compose` (Ubuntu/Debian)
- macOS: Download Docker Desktop from docker.com
- Windows: Download Docker Desktop from docker.com

**Problem**: Docker daemon not running
```bash
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Solution**:
```bash
# Start Docker service (Linux)
sudo systemctl start docker

# macOS/Windows: Start Docker Desktop application
# Then verify: docker ps
```

## Neo4j Connection Problems

### Container Issues

**Problem**: Neo4j container won't start
```bash
docker: Error response from daemon: port is already allocated
```

**Solution**:
```bash
# Check what's using the ports
sudo lsof -i :7474
sudo lsof -i :7687

# Stop existing Neo4j containers
docker stop $(docker ps -aq --filter "ancestor=neo4j")
docker rm $(docker ps -aq --filter "ancestor=neo4j")

# Restart with fresh container
docker-compose -f docker-compose.neo4j.yml up -d
```

**Problem**: Neo4j container starts but not accessible
```bash
curl: (7) Failed to connect to localhost port 7474
```

**Diagnosis**:
```bash
# Check container status
docker ps
docker logs active_rag_neo4j

# Check network connectivity
docker exec active_rag_neo4j netstat -tlnp
```

**Solutions**:
```bash
# Wait for full startup (can take 60-120 seconds)
sleep 120

# Check container health
docker exec active_rag_neo4j cypher-shell -u neo4j -p activerag123 "RETURN 1;"

# Restart with proper configuration
docker-compose -f docker-compose.neo4j.yml down
docker-compose -f docker-compose.neo4j.yml up -d
```

### Authentication Issues

**Problem**: Authentication failed
```bash
Neo.ClientError.Security.Unauthorized: The client is unauthorized due to authentication failure
```

**Solution**:
```bash
# Check password in environment
echo $NEO4J_PASSWORD

# Reset password using cypher-shell
docker exec -it active_rag_neo4j cypher-shell
# At prompt: ALTER USER neo4j SET PASSWORD 'activerag123';

# Or recreate container with correct credentials
docker-compose -f docker-compose.neo4j.yml down
export NEO4J_PASSWORD=activerag123
docker-compose -f docker-compose.neo4j.yml up -d
```

**Problem**: Password change required
```bash
Neo.ClientError.Security.CredentialsExpired: The credential have expired and must be changed
```

**Solution**:
```bash
# Connect and change password
docker exec -it active_rag_neo4j cypher-shell -u neo4j -p neo4j
# At prompt: ALTER USER neo4j SET PASSWORD 'activerag123';

# Update .env file
echo "NEO4J_PASSWORD=activerag123" >> .env
```

### Connection Timeout Issues

**Problem**: Connection timeouts
```bash
neo4j.exceptions.ServiceUnavailable: Failed to establish connection
```

**Solution**:
```python
# Increase timeout in configuration
from active_rag.config import Config

config = Config()
# Add custom timeout handling in neo4j_client.py initialization

# Or check system resources
docker stats active_rag_neo4j
```

**Problem**: Memory issues with large graphs
```bash
java.lang.OutOfMemoryError: Java heap space
```

**Solution**:
```yaml
# Add to docker-compose.neo4j.yml
environment:
  - NEO4J_dbms_memory_heap_initial__size=2G
  - NEO4J_dbms_memory_heap_max__size=4G
  - NEO4J_dbms_memory_pagecache_size=1G
```

## ChromaDB Issues

### Storage Problems

**Problem**: ChromaDB persistence issues
```bash
chromadb.errors.InvalidDatabaseException: Database does not exist
```

**Solution**:
```bash
# Check directory permissions
ls -la ./chroma_db/
chmod 755 ./chroma_db/

# Recreate database
rm -rf ./chroma_db/
mkdir ./chroma_db/
```

**Problem**: Collection already exists error
```bash
chromadb.errors.UniqueConstraintError: Collection already exists
```

**Solution**:
```python
# Use get_or_create instead of create
collection = client.get_or_create_collection("active_rag")

# Or delete existing collection first
try:
    client.delete_collection("active_rag")
except:
    pass
collection = client.create_collection("active_rag")
```

### Performance Issues

**Problem**: Slow embedding generation
```bash
# Taking too long to process documents
```

**Solution**:
```bash
# Check system resources
top
free -h

# Reduce batch size in processing
# Or switch to lighter embedding model in future versions
```

## NLP Pipeline Problems

### spaCy Issues

**Problem**: Model not found after installation
```python
OSError: [E050] Can't find model 'en_core_web_sm'
```

**Solutions**:
```bash
# Verify installation
python -c "import spacy; spacy.load('en_core_web_sm')"

# Reinstall model
python -m spacy download en_core_web_sm --force

# Link model if needed
python -m spacy link en_core_web_sm en
```

**Problem**: Memory errors with large documents
```python
MemoryError: Unable to allocate array
```

**Solution**:
```python
# Process documents in chunks
def process_large_document(text, chunk_size=5000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    all_entities = []
    
    for chunk in chunks:
        entities = extractor.extract_entities(chunk, domain)
        all_entities.extend(entities)
    
    return all_entities
```

### Entity Extraction Issues

**Problem**: No entities extracted from document
```python
entities = []  # Empty result
```

**Diagnosis**:
```python
# Check document classification
classifier = DocumentClassifier()
domain = classifier.classify_document(text)
print(f"Classified as: {domain}")

# Check text preprocessing
print(f"Text length: {len(text)}")
print(f"First 200 chars: {text[:200]}")

# Check spaCy processing
doc = extractor.nlp(text[:1000])
print(f"spaCy entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
```

**Solutions**:
```python
# Ensure minimum text length
if len(text.strip()) < 50:
    print("Document too short for entity extraction")

# Check domain classification
if domain == ContentDomain.MIXED_WEB:
    # May need manual domain specification
    entities = extractor.extract_entities(text, ContentDomain.RESEARCH)

# Add more context to text
text_with_context = f"Research paper about {title}. {text}"
```

## Document Processing Errors

### File Format Issues

**Problem**: Unsupported file format
```python
ValueError: Unsupported file type: .xyz
```

**Solution**:
```python
# Convert to supported format first
# Supported: .txt, .md, .pdf, .docx

# For other formats, extract text manually:
with open('document.xyz', 'r', encoding='utf-8') as f:
    content = f.read()

# Then process as string
doc_data = {
    "title": "Document Title",
    "content": content,
    "url": "file://document.xyz"
}
result = dual_storage.store_document(doc_data)
```

**Problem**: PDF extraction fails
```python
PdfReadError: PDF could not be read
```

**Solutions**:
```bash
# Install additional dependencies
pip install PyPDF2 pdfplumber

# Try alternative extraction
import pdfplumber
with pdfplumber.open("document.pdf") as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)
```

**Problem**: Encoding issues
```python
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solution**:
```python
# Handle encoding detection
import chardet

def load_with_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        encoding = chardet.detect(raw_data)['encoding']
    
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        return f.read()
```

### Large Document Issues

**Problem**: Memory errors with large files
```python
MemoryError: Unable to allocate memory
```

**Solution**:
```python
# Process large documents in chunks
def process_large_document(file_path, chunk_size=1000000):
    results = []
    
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            # Process each chunk separately
            result = dual_storage.store_document({
                "title": f"{Path(file_path).stem}_chunk_{len(results)}",
                "content": chunk,
                "url": f"file://{file_path}#chunk={len(results)}"
            })
            results.append(result)
    
    return results
```

## Graph Operations Issues

### Query Performance Problems

**Problem**: Graph queries taking too long
```cypher
MATCH (n)-[*1..5]-(m) RETURN count(*)  # Takes minutes
```

**Solutions**:
```cypher
-- Add limits to queries
MATCH (n:Person)-[*1..2]-(m) 
RETURN n, m 
LIMIT 100

-- Use more specific patterns
MATCH (p:Person {name: $name})-[:AUTHORED]->(d:Document)
RETURN d.title

-- Create indexes for common queries
CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);
CREATE INDEX document_title IF NOT EXISTS FOR (d:Document) ON (d.title);
```

**Problem**: Graph traversal depth too high
```python
ValueError: Invalid radius: 15. Must be integer between 1 and 10
```

**Solution**:
```python
# Use appropriate depth limits
neighbors = graph_ops.get_entity_neighborhood(entity_id, radius=3)

# For deeper analysis, use iterative approach
def deep_analysis(entity_id, max_depth=10):
    visited = set()
    current_level = [entity_id]
    
    for depth in range(max_depth):
        next_level = []
        for entity in current_level:
            if entity not in visited:
                visited.add(entity)
                neighbors = graph_ops.get_entity_neighborhood(entity, radius=1)
                next_level.extend([n['id'] for n in neighbors])
        
        current_level = next_level
        if not current_level:
            break
    
    return list(visited)
```

### Schema Validation Errors

**Problem**: Entity validation failures
```python
WARNING: Entity validation failed for Person: {'id': 'person_123'}
```

**Solution**:
```python
# Check required properties
from active_rag.schemas.entities import PERSON_SCHEMA

# Ensure all required properties are present
entity_data = {
    "label": "Person",
    "properties": {
        "id": "person_123",
        "name": "John Doe",  # Required property was missing
        "affiliation": "University"  # Optional
    }
}

# Validate before creation
schema = PERSON_SCHEMA
required_props = schema.required_properties
entity_props = entity_data["properties"]

for prop in required_props:
    if prop not in entity_props:
        print(f"Missing required property: {prop}")
```

## Performance Problems

### Slow Document Processing

**Problem**: Document ingestion taking too long
```bash
Processing 100 documents takes 30+ minutes
```

**Diagnosis**:
```python
import time

def timed_processing():
    start_time = time.time()
    
    # Time each component
    classification_time = time.time()
    domain = classifier.classify_document(text)
    classification_time = time.time() - classification_time
    
    extraction_time = time.time()
    entities = extractor.extract_entities(text, domain)
    extraction_time = time.time() - extraction_time
    
    storage_time = time.time()
    result = dual_storage.store_document(doc_data)
    storage_time = time.time() - storage_time
    
    total_time = time.time() - start_time
    
    print(f"Classification: {classification_time:.2f}s")
    print(f"Extraction: {extraction_time:.2f}s") 
    print(f"Storage: {storage_time:.2f}s")
    print(f"Total: {total_time:.2f}s")
```

**Solutions**:
```python
# 1. Process in batches
def batch_processing(documents, batch_size=10):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        for doc in batch:
            process_document(doc)
        # Small delay between batches
        time.sleep(0.1)

# 2. Optimize text preprocessing  
def preprocess_text(text, max_length=10000):
    # Limit text length for classification/extraction
    return text[:max_length]

# 3. Skip processing for very short documents
def should_process(text):
    return len(text.strip()) > 100  # Skip very short documents
```

### Memory Usage Issues

**Problem**: High memory consumption
```bash
Process using 8GB+ RAM
```

**Solution**:
```python
# Monitor memory usage
import psutil
import gc

def memory_check():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Force garbage collection
def cleanup_memory():
    gc.collect()
    
# Process documents with memory cleanup
def memory_efficient_processing(documents):
    for i, doc in enumerate(documents):
        process_document(doc)
        
        # Cleanup every 10 documents
        if i % 10 == 0:
            cleanup_memory()
            memory_check()
```

## Integration Test Failures

### Test Environment Issues

**Problem**: Tests fail due to missing setup
```bash
FAILED tests/integration/test_phase1_integration.py::test_integration_environment_setup
```

**Solution**:
```bash
# Run setup script first
python scripts/setup_phase1_environment.py

# Verify Neo4j is running
docker ps | grep neo4j

# Run tests with verbose output
pytest tests/integration/ -v -s

# Run specific test
pytest tests/integration/test_phase1_integration.py::test_integration_environment_setup -v
```

**Problem**: Neo4j connection fails in tests
```python
pytest.fail(f"Neo4j connection failed: {e}")
```

**Solution**:
```bash
# Check Neo4j container logs
docker logs active_rag_neo4j

# Wait for full startup before running tests
sleep 60
pytest tests/integration/

# Run tests with custom timeout
pytest tests/integration/ --timeout=300
```

### Data Consistency Issues

**Problem**: Graph and vector storage inconsistent
```python
AssertionError: ChromaDB should contain documents
```

**Diagnosis**:
```python
# Check both storage systems
def debug_storage():
    # ChromaDB
    collection = dual_storage.collection
    chroma_count = collection.count()
    print(f"ChromaDB documents: {chroma_count}")
    
    # Neo4j
    with neo4j_client._driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        neo4j_count = result.single()["count"]
        print(f"Neo4j nodes: {neo4j_count}")
```

**Solution**:
```python
# Clear both storage systems before tests
def clean_test_environment():
    # Clear ChromaDB
    try:
        client.delete_collection("active_rag")
    except:
        pass
    
    # Clear Neo4j  
    with neo4j_client._driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
```

## Environment Configuration

### Environment Variables

**Problem**: Configuration not loaded
```python
Config values using defaults instead of environment variables
```

**Solution**:
```bash
# Check .env file exists and has correct format
cat .env

# Verify environment variables are loaded
python -c "import os; print(os.getenv('NEO4J_PASSWORD'))"

# Source environment manually if needed
export $(cat .env | xargs)
```

**Problem**: Docker Compose environment issues
```bash
Environment variable NEO4J_PASSWORD is not set
```

**Solution**:
```bash
# Create .env file in project root
cat > .env << EOF
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=activerag123
ENABLE_GRAPH_FEATURES=true
EOF

# Verify variables are available to Docker Compose
docker-compose -f docker-compose.neo4j.yml config
```

### Path Configuration

**Problem**: ChromaDB path issues
```python
PermissionError: [Errno 13] Permission denied: './chroma_db'
```

**Solution**:
```bash
# Create directory with correct permissions
mkdir -p ./chroma_db
chmod 755 ./chroma_db

# Use absolute path in configuration
export CHROMA_PERSIST_DIR="$(pwd)/chroma_db"
```

## Common Error Messages

### "Neo4j service is unavailable"
**Cause**: Neo4j container not running or not ready
**Fix**: `docker-compose -f docker-compose.neo4j.yml up -d && sleep 60`

### "spaCy model 'en_core_web_sm' not found"
**Cause**: spaCy English model not installed
**Fix**: `python -m spacy download en_core_web_sm`

### "Collection 'active_rag' already exists"
**Cause**: ChromaDB collection conflict
**Fix**: Use `get_or_create_collection()` instead of `create_collection()`

### "Entity validation failed"
**Cause**: Missing required properties in entity
**Fix**: Ensure all required properties (id, name) are present

### "Connection timeout" 
**Cause**: Neo4j taking too long to start or high load
**Fix**: Wait longer, check resources, or increase timeout

### "Permission denied" 
**Cause**: File/directory permission issues
**Fix**: Check file permissions, use virtual environment

### "Port already in use"
**Cause**: Another service using Neo4j ports
**Fix**: Stop conflicting service or change ports in docker-compose

### "Memory error"
**Cause**: Large documents or insufficient RAM
**Fix**: Process in chunks, increase system memory

## Getting Additional Help

### Enabling Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# For specific modules
logger = logging.getLogger('active_rag.storage.dual_storage_manager')
logger.setLevel(logging.DEBUG)
```

### Health Monitoring

```python
def system_health_check():
    """Comprehensive system health check"""
    health = {
        "neo4j": check_neo4j_health(),
        "chromadb": check_chromadb_health(), 
        "nlp": check_nlp_health(),
        "memory": check_memory_usage(),
        "disk": check_disk_space()
    }
    
    return health
```

### Community Resources

- **GitHub Issues**: Report bugs and get help from the community
- **Integration Tests**: Use as examples for proper usage patterns
- **Docker Logs**: `docker logs active_rag_neo4j` for Neo4j debugging
- **Neo4j Browser**: http://localhost:7474 for direct graph exploration

Remember: Most issues are environment-related. When in doubt, start with the health check script and work through the setup process step by step.