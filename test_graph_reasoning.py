#!/usr/bin/env python3
"""
Test the enhanced graph reasoning with direct queries
"""

from active_rag.config import Config
from active_rag.knowledge_graph.neo4j_client import Neo4jClient

def test_enhanced_graph():
    """Test graph queries directly"""

    config = Config()
    client = Neo4jClient(config.neo4j_uri, config.neo4j_username, config.neo4j_password)

    print("🧪 Testing Enhanced Graph Reasoning...")

    with client._driver.session() as session:

        # Test 1: Direct Alice Smith query
        print("\n🎯 Query 1: Alice Smith projects and collaborators")
        result = session.run("""
            MATCH (alice:Person {name: "Alice Smith"})
            WHERE alice.id STARTS WITH 'enhanced_'

            // Find projects Alice leads or contributes to
            OPTIONAL MATCH (alice)-[proj_rel:LEADS|CONTRIBUTES_TO]->(project:Project)

            // Find Alice's direct collaborators
            OPTIONAL MATCH (alice)-[collab_rel:COLLABORATES_WITH]-(collaborator:Person)

            // Find colleagues at same company
            OPTIONAL MATCH (alice)-[:WORKS_FOR]->(company)<-[:WORKS_FOR]-(colleague:Person)
            WHERE colleague.id <> alice.id

            RETURN alice.name as name,
                   alice.title as title,
                   alice.current_focus as focus,
                   collect(DISTINCT {
                       name: project.name,
                       role: type(proj_rel),
                       description: project.description
                   }) as projects,
                   collect(DISTINCT {
                       name: collaborator.name,
                       relationship: type(collab_rel)
                   }) as collaborators,
                   collect(DISTINCT {
                       name: colleague.name,
                       title: colleague.title
                   }) as colleagues
        """)

        for record in result:
            print(f"👤 Name: {record['name']}")
            print(f"💼 Title: {record['title']}")
            print(f"🎯 Focus: {record['focus']}")

            print("\n📋 Projects:")
            for project in record['projects']:
                if project['name']:  # Skip null projects
                    print(f"  • {project['name']} (Role: {project['role']})")
                    print(f"    {project['description']}")

            print("\n🤝 Collaborators:")
            for collab in record['collaborators']:
                if collab['name']:  # Skip null collaborators
                    print(f"  • {collab['name']} ({collab['relationship']})")

            print("\n👥 Colleagues:")
            for colleague in record['colleagues']:
                if colleague['name']:  # Skip null colleagues
                    print(f"  • {colleague['name']} - {colleague['title']}")

        print("\n" + "="*60)

        # Test 2: Multi-hop reasoning - Who works on RAG technology?
        print("\n🔍 Query 2: Who works on RAG technology?")
        result = session.run("""
            MATCH (rag:Technology {name: "Retrieval-Augmented Generation"})

            // Find people who are experts in RAG
            OPTIONAL MATCH (person:Person)-[expert:EXPERT_IN]->(rag)

            // Find people who work on RAG projects
            OPTIONAL MATCH (person2:Person)-[:LEADS|CONTRIBUTES_TO]->(project:Project)-[:IMPLEMENTS]->(rag)

            // Find people who authored papers about RAG
            OPTIONAL MATCH (person3:Person)-[:AUTHORED]->(paper:Document)-[:DISCUSSES]->(rag)

            RETURN collect(DISTINCT {
                       name: person.name,
                       title: person.title,
                       expertise_level: expert.proficiency_level,
                       years: expert.years_experience
                   }) as experts,
                   collect(DISTINCT {
                       name: person2.name,
                       title: person2.title,
                       project: project.name
                   }) as project_workers,
                   collect(DISTINCT {
                       name: person3.name,
                       title: person3.title,
                       paper: paper.title
                   }) as authors
        """)

        for record in result:
            print("🎓 RAG Experts:")
            for expert in record['experts']:
                if expert['name']:
                    print(f"  • {expert['name']} - {expert['title']}")
                    print(f"    Level: {expert['expertise_level']}, Experience: {expert['years']} years")

            print("\n🚧 RAG Project Workers:")
            for worker in record['project_workers']:
                if worker['name']:
                    print(f"  • {worker['name']} - {worker['title']}")
                    print(f"    Project: {worker['project']}")

            print("\n📝 RAG Authors:")
            for author in record['authors']:
                if author['name']:
                    print(f"  • {author['name']} - {author['title']}")
                    print(f"    Paper: {author['paper']}")

        print("\n" + "="*60)

        # Test 3: Company network
        print("\n🏢 Query 3: TechCorp network and partnerships")
        result = session.run("""
            MATCH (techcorp:Company {name: "TechCorp AI Solutions"})

            // Find employees
            OPTIONAL MATCH (employee:Person)-[emp_rel:WORKS_FOR]->(techcorp)

            // Find partnerships
            OPTIONAL MATCH (techcorp)-[partner_rel:PARTNERS_WITH|COLLABORATES_WITH]-(partner:Organization)

            // Find projects owned by TechCorp
            OPTIONAL MATCH (techcorp)-[:OWNS]->(project:Project)

            RETURN collect(DISTINCT {
                       name: employee.name,
                       title: employee.title,
                       department: emp_rel.department
                   }) as employees,
                   collect(DISTINCT {
                       name: partner.name,
                       relationship: type(partner_rel)
                   }) as partners,
                   collect(DISTINCT {
                       name: project.name,
                       status: project.status
                   }) as projects
        """)

        for record in result:
            print("👥 TechCorp Employees:")
            for emp in record['employees']:
                if emp['name']:
                    print(f"  • {emp['name']} - {emp['title']} ({emp['department']})")

            print("\n🤝 Partners:")
            for partner in record['partners']:
                if partner['name']:
                    print(f"  • {partner['name']} ({partner['relationship']})")

            print("\n📋 Projects:")
            for project in record['projects']:
                if project['name']:
                    print(f"  • {project['name']} - Status: {project['status']}")

    client.close()
    print("\n✅ Graph testing complete!")

if __name__ == "__main__":
    test_enhanced_graph()