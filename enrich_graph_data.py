#!/usr/bin/env python3
"""
Enhanced Neo4j Data Population Script

Creates a rich knowledge graph with meaningful relationships to showcase
the full power of the ActiveRAG hybrid pipeline.
"""

from active_rag.config import Config
from active_rag.knowledge_graph.neo4j_client import Neo4jClient

def populate_rich_graph_data():
    """Create a comprehensive knowledge graph with connected entities."""

    config = Config()
    client = Neo4jClient(config.neo4j_uri, config.neo4j_username, config.neo4j_password)

    print("🚀 Enriching Neo4j graph database...")

    with client._driver.session() as session:

        # Clear existing test data to avoid conflicts
        print("📋 Clearing existing test entities...")
        session.run("""
            MATCH (n)
            WHERE n.id STARTS WITH 'test_' OR n.id STARTS WITH 'enhanced_'
            DETACH DELETE n
        """)

        # ===== CORE ENTITIES =====
        print("👥 Creating people...")

        # Enhanced Alice Smith with rich context
        session.run("""
            CREATE (alice:Person {
                id: 'enhanced_alice_smith',
                name: 'Alice Smith',
                title: 'Senior AI Researcher',
                email: 'alice.smith@techcorp.com',
                linkedin: 'linkedin.com/in/alicesmith-ai',
                expertise: ['Natural Language Processing', 'Knowledge Graphs', 'RAG Systems'],
                years_experience: 8,
                current_focus: 'Hybrid Vector-Graph Retrieval Systems'
            })
        """)

        # Bob Jones with connections
        session.run("""
            CREATE (bob:Person {
                id: 'enhanced_bob_jones',
                name: 'Bob Jones',
                title: 'Machine Learning Engineer',
                email: 'bob.jones@techcorp.com',
                expertise: ['Deep Learning', 'Computer Vision', 'MLOps'],
                years_experience: 6,
                current_focus: 'Production ML Systems'
            })
        """)

        # Dr. Sarah Chen - Academic connection
        session.run("""
            CREATE (sarah:Person {
                id: 'enhanced_sarah_chen',
                name: 'Dr. Sarah Chen',
                title: 'Professor of Computer Science',
                affiliation: 'Stanford University',
                email: 's.chen@stanford.edu',
                expertise: ['Knowledge Representation', 'Semantic Web', 'AI Ethics'],
                h_index: 42,
                current_focus: 'Ethical AI and Knowledge Systems'
            })
        """)

        print("🏢 Creating organizations...")

        # TechCorp - Primary company
        session.run("""
            CREATE (techcorp:Organization:Company {
                id: 'enhanced_techcorp',
                name: 'TechCorp AI Solutions',
                industry: 'Artificial Intelligence',
                founded: 2018,
                headquarters: 'San Francisco, CA',
                size: '200-500 employees',
                specialization: 'Enterprise AI platforms and RAG systems',
                website: 'https://techcorp-ai.com'
            })
        """)

        # Stanford University
        session.run("""
            CREATE (stanford:Organization:University {
                id: 'enhanced_stanford',
                name: 'Stanford University',
                type: 'Research University',
                location: 'Stanford, CA',
                established: 1885,
                notable_for: 'Computer Science and AI Research'
            })
        """)

        # OpenAI Partnership
        session.run("""
            CREATE (openai:Organization:Company {
                id: 'enhanced_openai',
                name: 'OpenAI',
                industry: 'AI Research',
                founded: 2015,
                headquarters: 'San Francisco, CA',
                known_for: 'Large Language Models and GPT systems'
            })
        """)

        print("💡 Creating projects and technologies...")

        # ActiveRAG Project
        session.run("""
            CREATE (activerag:Project {
                id: 'enhanced_activerag',
                name: 'ActiveRAG System',
                description: 'Hybrid Vector-Graph Retrieval-Augmented Generation platform',
                status: 'Production',
                start_date: '2023-06-15',
                tech_stack: ['Python', 'Neo4j', 'ChromaDB', 'OpenAI API', 'FastAPI'],
                github_url: 'https://github.com/techcorp/activerag',
                primary_use_case: 'Intelligent document retrieval and question answering'
            })
        """)

        # RAG Technology Concept
        session.run("""
            CREATE (rag:Technology:Concept {
                id: 'enhanced_rag_tech',
                name: 'Retrieval-Augmented Generation',
                category: 'AI/ML Architecture Pattern',
                description: 'Combines pre-trained language models with external knowledge retrieval',
                invented_year: 2020,
                key_papers: ['RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks'],
                applications: ['Question Answering', 'Fact Checking', 'Content Generation']
            })
        """)

        # Neo4j Technology
        session.run("""
            CREATE (neo4j:Technology:Database {
                id: 'enhanced_neo4j_tech',
                name: 'Neo4j Graph Database',
                category: 'Graph Database',
                vendor: 'Neo4j, Inc.',
                first_release: 2007,
                query_language: 'Cypher',
                use_cases: ['Knowledge Graphs', 'Recommendation Engines', 'Fraud Detection']
            })
        """)

        print("📚 Creating research papers...")

        # Research paper by Alice and Sarah
        session.run("""
            CREATE (paper1:Document:ResearchPaper {
                id: 'enhanced_paper_hybrid_rag',
                title: 'Hybrid Vector-Graph Approaches to Retrieval-Augmented Generation',
                abstract: 'This paper explores novel architectures combining vector similarity search with graph-based reasoning for enhanced RAG systems.',
                publication_date: '2024-01-15',
                venue: 'International Conference on AI Systems',
                doi: '10.1000/xyz123',
                citation_count: 23,
                pdf_url: 'https://research.techcorp.com/papers/hybrid-rag-2024.pdf'
            })
        """)

        # Technical blog post
        session.run("""
            CREATE (blog1:Document:BlogPost {
                id: 'enhanced_blog_neo4j_rag',
                title: 'Building Production RAG Systems with Neo4j and ChromaDB',
                author: 'Alice Smith',
                publication_date: '2024-02-20',
                platform: 'TechCorp AI Blog',
                url: 'https://blog.techcorp-ai.com/neo4j-chromadb-rag',
                tags: ['RAG', 'Neo4j', 'Vector Databases', 'Production AI'],
                read_time: '12 minutes'
            })
        """)

        print("🔗 Creating rich relationships...")

        # ===== EMPLOYMENT & AFFILIATION =====
        session.run("""
            MATCH (alice:Person {id: 'enhanced_alice_smith'})
            MATCH (bob:Person {id: 'enhanced_bob_jones'})
            MATCH (sarah:Person {id: 'enhanced_sarah_chen'})
            MATCH (techcorp:Organization {id: 'enhanced_techcorp'})
            MATCH (stanford:Organization {id: 'enhanced_stanford'})

            CREATE (alice)-[:WORKS_FOR {
                position: 'Senior AI Researcher',
                start_date: '2022-03-01',
                department: 'AI Research Lab',
                salary_range: '$150k-200k'
            }]->(techcorp)

            CREATE (bob)-[:WORKS_FOR {
                position: 'ML Engineer',
                start_date: '2021-09-15',
                department: 'Engineering',
                reports_to: 'alice'
            }]->(techcorp)

            CREATE (sarah)-[:AFFILIATED_WITH {
                role: 'Professor',
                department: 'Computer Science',
                tenure_track: true,
                start_date: '2019-09-01'
            }]->(stanford)
        """)

        # ===== COLLABORATION & MENTORSHIP =====
        session.run("""
            MATCH (alice:Person {id: 'enhanced_alice_smith'})
            MATCH (bob:Person {id: 'enhanced_bob_jones'})
            MATCH (sarah:Person {id: 'enhanced_sarah_chen'})

            CREATE (alice)-[:COLLABORATES_WITH {
                project: 'ActiveRAG System',
                relationship_type: 'technical_lead_and_engineer',
                start_date: '2023-06-01',
                frequency: 'daily'
            }]->(bob)

            CREATE (alice)-[:MENTORED_BY {
                relationship_type: 'PhD_advisor',
                duration: '2018-2021',
                thesis_topic: 'Neural Information Retrieval Systems'
            }]->(sarah)

            CREATE (sarah)-[:COLLABORATES_WITH {
                project: 'Ethical RAG Research',
                relationship_type: 'academic_industry_partnership',
                start_date: '2023-01-15'
            }]->(alice)
        """)

        # ===== PROJECT & TECHNOLOGY RELATIONSHIPS =====
        session.run("""
            MATCH (alice:Person {id: 'enhanced_alice_smith'})
            MATCH (bob:Person {id: 'enhanced_bob_jones'})
            MATCH (activerag:Project {id: 'enhanced_activerag'})
            MATCH (rag_tech:Technology {id: 'enhanced_rag_tech'})
            MATCH (neo4j_tech:Technology {id: 'enhanced_neo4j_tech'})

            CREATE (alice)-[:LEADS {
                role: 'Technical Lead',
                responsibility: 'Architecture and Research Direction',
                start_date: '2023-06-01'
            }]->(activerag)

            CREATE (bob)-[:CONTRIBUTES_TO {
                role: 'Implementation Engineer',
                focus_area: 'Pipeline Optimization and Testing',
                start_date: '2023-07-15'
            }]->(activerag)

            CREATE (activerag)-[:IMPLEMENTS]->(rag_tech)
            CREATE (activerag)-[:USES_TECHNOLOGY]->(neo4j_tech)
        """)

        # ===== RESEARCH & PUBLICATIONS =====
        session.run("""
            MATCH (alice:Person {id: 'enhanced_alice_smith'})
            MATCH (sarah:Person {id: 'enhanced_sarah_chen'})
            MATCH (paper1:Document {id: 'enhanced_paper_hybrid_rag'})
            MATCH (blog1:Document {id: 'enhanced_blog_neo4j_rag'})
            MATCH (rag_tech:Technology {id: 'enhanced_rag_tech'})

            CREATE (alice)-[:AUTHORED {
                contribution: 'Lead Author',
                role: 'System Design and Implementation'
            }]->(paper1)

            CREATE (sarah)-[:AUTHORED {
                contribution: 'Co-Author',
                role: 'Theoretical Framework and Ethics Review'
            }]->(paper1)

            CREATE (alice)-[:AUTHORED {
                publication_date: '2024-02-20',
                platform: 'company_blog'
            }]->(blog1)

            CREATE (paper1)-[:DISCUSSES]->(rag_tech)
            CREATE (blog1)-[:DISCUSSES]->(rag_tech)
        """)

        # ===== EXPERTISE & RESEARCH INTERESTS =====
        session.run("""
            MATCH (alice:Person {id: 'enhanced_alice_smith'})
            MATCH (bob:Person {id: 'enhanced_bob_jones'})
            MATCH (sarah:Person {id: 'enhanced_sarah_chen'})
            MATCH (rag_tech:Technology {id: 'enhanced_rag_tech'})
            MATCH (neo4j_tech:Technology {id: 'enhanced_neo4j_tech'})

            CREATE (alice)-[:EXPERT_IN {
                proficiency_level: 'advanced',
                years_experience: 5,
                notable_contributions: 'Hybrid retrieval architectures'
            }]->(rag_tech)

            CREATE (alice)-[:EXPERT_IN {
                proficiency_level: 'advanced',
                years_experience: 3,
                focus_area: 'Knowledge graph integration'
            }]->(neo4j_tech)

            CREATE (sarah)-[:RESEARCHES {
                focus_area: 'Ethical implications and bias in RAG systems',
                funding_source: 'NSF Grant #AI-2024-0123',
                start_date: '2023-01-01'
            }]->(rag_tech)
        """)

        # ===== ORGANIZATIONAL RELATIONSHIPS =====
        session.run("""
            MATCH (techcorp:Organization {id: 'enhanced_techcorp'})
            MATCH (stanford:Organization {id: 'enhanced_stanford'})
            MATCH (openai:Organization {id: 'enhanced_openai'})
            MATCH (activerag:Project {id: 'enhanced_activerag'})

            CREATE (techcorp)-[:PARTNERS_WITH {
                partnership_type: 'Technology Integration',
                start_date: '2023-04-01',
                focus: 'LLM API integration'
            }]->(openai)

            CREATE (techcorp)-[:COLLABORATES_WITH {
                collaboration_type: 'Research Partnership',
                agreement_date: '2023-01-15',
                focus: 'Ethical AI Development'
            }]->(stanford)

            CREATE (techcorp)-[:OWNS]->(activerag)
        """)

        print("✅ Rich graph data populated successfully!")
        print("\n📊 New Enhanced Dataset Summary:")

        # Show statistics
        result = session.run("""
            MATCH (n)
            WHERE n.id STARTS WITH 'enhanced_'
            RETURN labels(n) as labels, count(n) as count
            ORDER BY count DESC
        """)

        print("\nNew Entities Created:")
        for record in result:
            labels = ':'.join(record['labels'])
            count = record['count']
            print(f"  • {labels}: {count}")

        # Show relationship statistics
        result = session.run("""
            MATCH (a)-[r]->(b)
            WHERE a.id STARTS WITH 'enhanced_' AND b.id STARTS WITH 'enhanced_'
            RETURN type(r) as rel_type, count(r) as count
            ORDER BY count DESC
        """)

        print("\nNew Relationships Created:")
        for record in result:
            rel_type = record['rel_type']
            count = record['count']
            print(f"  • {rel_type}: {count}")

    client.close()

if __name__ == "__main__":
    populate_rich_graph_data()