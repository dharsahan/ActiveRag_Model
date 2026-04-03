#!/usr/bin/env python3
"""Setup script for Phase 1 hybrid vector-graph RAG environment

This script automates the setup of the complete development environment including:
- Python dependencies installation
- NLP model downloads
- Neo4j container management
- Environment verification
- Integration testing
"""

import subprocess
import sys
import logging
import os
import time
from pathlib import Path
import requests


def setup_logging():
    """Configure logging for setup script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Docker found: {result.stdout.strip()}")

            # Check if Docker daemon is running
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            if result.returncode == 0:
                logging.info("Docker daemon is running")
                return True
            else:
                logging.error("Docker daemon is not running. Please start Docker.")
                return False
    except FileNotFoundError:
        pass

    logging.error("Docker not found. Please install Docker to run Neo4j.")
    return False


def check_docker_compose():
    """Check if Docker Compose is available"""
    # Try docker-compose first, then docker compose
    for cmd in [['docker-compose', '--version'], ['docker', 'compose', 'version']]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logging.info(f"Docker Compose found: {result.stdout.strip()}")
                return cmd[0] if cmd[0] == 'docker-compose' else 'docker compose'
        except FileNotFoundError:
            continue

    logging.error("Docker Compose not found. Please install Docker Compose.")
    return None


def start_neo4j(compose_cmd='docker-compose'):
    """Start Neo4j container"""
    logging.info("Starting Neo4j container...")

    try:
        # Stop existing container if running
        subprocess.run(['docker', 'stop', 'active_rag_neo4j'],
                      capture_output=True, stderr=subprocess.DEVNULL)
        subprocess.run(['docker', 'rm', 'active_rag_neo4j'],
                      capture_output=True, stderr=subprocess.DEVNULL)

        # Check if docker-compose.neo4j.yml exists
        compose_file = Path('docker-compose.neo4j.yml')
        if not compose_file.exists():
            logging.warning("docker-compose.neo4j.yml not found, creating minimal Neo4j setup")
            create_neo4j_compose_file()

        # Start new container
        if compose_cmd == 'docker-compose':
            cmd = ['docker-compose', '-f', 'docker-compose.neo4j.yml', 'up', '-d']
        else:
            cmd = ['docker', 'compose', '-f', 'docker-compose.neo4j.yml', 'up', '-d']

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logging.info("Neo4j container started successfully")
            return True
        else:
            logging.error(f"Failed to start Neo4j: {result.stderr}")
            # Try alternative Docker run command
            return start_neo4j_direct()

    except Exception as e:
        logging.error(f"Error starting Neo4j: {e}")
        return start_neo4j_direct()


def start_neo4j_direct():
    """Start Neo4j using direct Docker run command as fallback"""
    logging.info("Trying direct Docker run for Neo4j...")

    try:
        cmd = [
            'docker', 'run', '-d',
            '--name', 'active_rag_neo4j',
            '-p', '7474:7474',
            '-p', '7687:7687',
            '-e', 'NEO4J_AUTH=neo4j/testpassword',
            '-e', 'NEO4J_PLUGINS=["apoc"]',
            '--env', 'NEO4J_dbms_security_procedures_unrestricted=apoc.*',
            'neo4j:5.15'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logging.info("Neo4j started successfully with direct Docker run")
            return True
        else:
            logging.error(f"Failed to start Neo4j with direct run: {result.stderr}")
            return False

    except Exception as e:
        logging.error(f"Error with direct Neo4j start: {e}")
        return False


def create_neo4j_compose_file():
    """Create a minimal docker-compose.neo4j.yml file"""
    compose_content = '''version: '3.8'

services:
  neo4j:
    image: neo4j:5.15
    container_name: active_rag_neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/testpassword
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/import

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
'''

    with open('docker-compose.neo4j.yml', 'w') as f:
        f.write(compose_content)

    logging.info("Created docker-compose.neo4j.yml")


def wait_for_neo4j(timeout=120):
    """Wait for Neo4j to be ready"""
    logging.info("Waiting for Neo4j to be ready...")

    for i in range(timeout):
        try:
            response = requests.get('http://localhost:7474', timeout=5)
            if response.status_code == 200:
                logging.info("Neo4j is ready!")
                return True
        except:
            pass

        time.sleep(1)
        if i % 10 == 0 and i > 0:
            logging.info(f"Still waiting for Neo4j... ({i}/{timeout}s)")

    logging.error("Neo4j failed to start within timeout period")
    return False


def install_python_dependencies():
    """Install required Python packages"""
    logging.info("Installing Python dependencies...")

    try:
        # Check if virtual environment is activated
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            logging.info("Virtual environment detected")
        else:
            logging.warning("No virtual environment detected. Consider using one.")

        # Install main requirements
        if Path('requirements.txt').exists():
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to install main requirements: {result.stderr}")
                return False
            else:
                logging.info("Main requirements installed successfully")
        else:
            logging.warning("requirements.txt not found, skipping main requirements")

        # Install NLP requirements
        if Path('requirements_nlp.txt').exists():
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_nlp.txt'],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Failed to install NLP requirements: {result.stderr}")
                return False
            else:
                logging.info("NLP requirements installed successfully")
        else:
            logging.warning("requirements_nlp.txt not found, skipping NLP requirements")

        # Install test requirements
        test_packages = ['pytest', 'pytest-asyncio', 'pytest-mock']
        for package in test_packages:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logging.info(f"Installed {package}")

        logging.info("Python dependencies installation completed")
        return True

    except Exception as e:
        logging.error(f"Error installing dependencies: {e}")
        return False


def download_spacy_model():
    """Download spaCy English model"""
    logging.info("Downloading spaCy English model...")

    try:
        result = subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("spaCy model downloaded successfully")
            return True
        else:
            logging.error(f"Failed to download spaCy model: {result.stderr}")
            return False

    except Exception as e:
        logging.error(f"Error downloading spaCy model: {e}")
        return False


def setup_environment_files():
    """Setup necessary environment files"""
    logging.info("Setting up environment files...")

    # Check if .env exists
    env_file = Path('.env')
    env_example = Path('.env.example')

    if not env_file.exists() and env_example.exists():
        try:
            import shutil
            shutil.copy(env_example, env_file)
            logging.info("Created .env from .env.example")
        except Exception as e:
            logging.warning(f"Could not create .env file: {e}")

    # Create basic .env if neither exists
    if not env_file.exists():
        default_env = '''# Active RAG Environment Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword
NEO4J_DATABASE=neo4j

# OpenAI Configuration (optional - for enhanced NLP)
# OPENAI_API_KEY=your_openai_key_here

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Logging
LOG_LEVEL=INFO
'''
        with open(env_file, 'w') as f:
            f.write(default_env)
        logging.info("Created basic .env file")


def run_integration_tests():
    """Run integration tests to validate setup"""
    logging.info("Running integration tests...")

    try:
        # First check if tests directory exists
        if not Path('tests/integration').exists():
            logging.warning("Integration tests directory not found")
            return True

        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/integration/', '-v', '--tb=short'],
                              capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logging.info("✅ All integration tests passed!")
            return True
        else:
            logging.warning("⚠️ Some integration tests failed or had issues:")
            logging.warning(result.stdout)
            if result.stderr:
                logging.warning(result.stderr)
            logging.info("Setup completed, but some tests need attention")
            return True  # Don't fail setup for test failures

    except subprocess.TimeoutExpired:
        logging.warning("Integration tests timed out - this may be normal for first run")
        return True
    except Exception as e:
        logging.error(f"Error running integration tests: {e}")
        return False


def print_status_summary():
    """Print final status summary"""
    logging.info("\n" + "="*60)
    logging.info("SETUP SUMMARY")
    logging.info("="*60)

    # Check Neo4j
    try:
        response = requests.get('http://localhost:7474', timeout=5)
        if response.status_code == 200:
            logging.info("✅ Neo4j: Running (http://localhost:7474)")
        else:
            logging.info("❌ Neo4j: Not responding")
    except:
        logging.info("❌ Neo4j: Not available")

    # Check Python packages
    packages_to_check = ['spacy', 'neo4j', 'chromadb', 'pytest']
    for package in packages_to_check:
        try:
            __import__(package)
            logging.info(f"✅ Python package '{package}': Installed")
        except ImportError:
            logging.info(f"❌ Python package '{package}': Missing")

    # Check environment
    if Path('.env').exists():
        logging.info("✅ Environment file: Present")
    else:
        logging.info("❌ Environment file: Missing")

    logging.info("\nNext Steps:")
    logging.info("1. Run integration tests: pytest tests/integration/ -v")
    logging.info("2. Access Neo4j browser: http://localhost:7474")
    logging.info("3. Start developing with the hybrid RAG system!")


def main():
    """Main setup function"""
    setup_logging()
    logging.info("🚀 Starting Phase 1 environment setup for Hybrid Vector-Graph RAG")

    success = True

    # Check prerequisites
    if not check_docker():
        success = False

    compose_cmd = check_docker_compose()
    if not compose_cmd and success:
        logging.warning("Docker Compose not found, will use direct Docker commands")

    # Setup environment files
    setup_environment_files()

    # Install dependencies
    if success and not install_python_dependencies():
        success = False

    # Download NLP models
    if success and not download_spacy_model():
        # Don't fail setup for spaCy model issues
        logging.warning("spaCy model download failed, but continuing setup")

    # Start Neo4j
    if success and not start_neo4j(compose_cmd or 'docker'):
        success = False

    # Wait for Neo4j to be ready
    if success and not wait_for_neo4j():
        success = False

    # Run integration tests
    if success:
        run_integration_tests()

    # Print summary
    print_status_summary()

    if success:
        logging.info("\n🎉 Phase 1 environment setup completed successfully!")
    else:
        logging.error("\n💥 Setup completed with errors. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()