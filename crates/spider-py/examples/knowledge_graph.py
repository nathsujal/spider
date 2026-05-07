"""
Knowledge Graph Example — Building a knowledge graph from text.

This demonstrates building a knowledge graph with multiple documents
and entities, showing how Spider deduplicates entities across documents
and enables graph traversal.

Run with: python examples/knowledge_graph.py
"""

import os
import tempfile

import spider


def main():
    with tempfile.TemporaryDirectory(prefix="spider_knowledge_graph_") as tmpdir:
        db_path = os.path.join(tmpdir, "knowledge_graph")

        print("Building a Knowledge Graph with Spider\n")
        print("=" * 50)

        with spider.Spider.open(db_path) as db:
            # ---------------------------------------------------------------
            # 1. Ingest multiple documents about related topics
            # ---------------------------------------------------------------
            print("\n1. Ingesting documents about tech companies...\n")

            documents = [
                spider.IngestRequest(
                    title="Apple Overview",
                    propositions=[
                        spider.Proposition(
                            "Apple is headquartered in Cupertino",
                            entities=[
                                spider.Entity("Apple", "ORGANIZATION"),
                                spider.Entity("Cupertino", "LOCATION"),
                            ],
                        ),
                        spider.Proposition(
                            "Tim Cook is CEO of Apple",
                            entities=[
                                spider.Entity("Tim Cook", "PERSON"),
                                spider.Entity("Apple", "ORGANIZATION"),
                            ],
                        ),
                    ],
                ),
                spider.IngestRequest(
                    title="Google Overview",
                    propositions=[
                        spider.Proposition(
                            "Google is headquartered in Mountain View",
                            entities=[
                                spider.Entity("Google", "ORGANIZATION"),
                                spider.Entity("Mountain View", "LOCATION"),
                            ],
                        ),
                        spider.Proposition(
                            "Sundar Pichai is CEO of Google",
                            entities=[
                                spider.Entity("Sundar Pichai", "PERSON"),
                                spider.Entity("Google", "ORGANIZATION"),
                            ],
                        ),
                    ],
                ),
                spider.IngestRequest(
                    title="Tech Industry in California",
                    propositions=[
                        spider.Proposition(
                            "Cupertino is in California",
                            entities=[
                                spider.Entity("Cupertino", "LOCATION"),
                                spider.Entity("California", "LOCATION"),
                            ],
                        ),
                        spider.Proposition(
                            "Mountain View is in California",
                            entities=[
                                spider.Entity("Mountain View", "LOCATION"),
                                spider.Entity("California", "LOCATION"),
                            ],
                        ),
                    ],
                ),
            ]

            for doc in documents:
                result = db.index(doc)
                print(f"  Ingested '{doc.title}'")
                print(f"    Document ID: {result.document_id}")
                print(f"    Entities: {result.entity_count}, Edges: {result.edge_count}")

            # ---------------------------------------------------------------
            # 2. Query the graph
            # ---------------------------------------------------------------
            print("\n2. Querying the knowledge graph...\n")

            # Find all entities
            entity_nodes = db.find_by_label("ENTITY")
            print(f"  Total entity nodes: {len(entity_nodes)}")

            # Find specific entities by name
            for name in ["Apple", "Google", "California", "Cupertino"]:
                nodes = db.find_by_property("name", name)
                print(f"  Nodes named '{name}': {len(nodes)}")

            # ---------------------------------------------------------------
            # 3. Explore relationships
            # ---------------------------------------------------------------
            print("\n3. Exploring relationships...\n")

            # Find the "California" entity
            california_nodes = db.find_by_property("name", "California")
            if california_nodes:
                california_id = california_nodes[0]
                incoming = db.get_neighbors(california_id, spider.Direction.INCOMING)
                print(f"  'California' incoming neighbors: {len(incoming)}")
                for n in incoming:
                    print(f"    <- Node {n.node_id} via Edge {n.edge_id}")

                outgoing = db.get_neighbors(california_id, spider.Direction.OUTGOING)
                print(f"  'California' outgoing neighbors: {len(outgoing)}")

            # Find Apple organization
            apple_nodes = db.find_by_property("name", "Apple")
            if apple_nodes:
                apple_id = apple_nodes[0]
                rels = db.get_relationships(apple_id, spider.Direction.BOTH)
                print(f"\n  'Apple' relationships: {len(rels)}")
                for rel in rels:
                    print(f"    {rel['source_id']} -> {rel['target_id']}")

            # ---------------------------------------------------------------
            # 4. Bio scoring — which entities are most "alive"?
            # ---------------------------------------------------------------
            print("\n4. Bio scoring — entity vitality...\n")

            for entity_name in ["Apple", "Google", "California", "Cupertino", "Tim Cook"]:
                nodes = db.find_by_property("name", entity_name)
                if nodes:
                    node_id = nodes[0]
                    score = db.get_bio_score(node_id)
                    tier = db.get_bio_tier(node_id)
                    print(f"  {entity_name}: score={score:.2f}, tier={tier}")

            # ---------------------------------------------------------------
            # 5. Node count
            # ---------------------------------------------------------------
            print(f"\n5. Total nodes in graph: {db.node_count()}")

            print("\n" + "=" * 50)
            print("Knowledge graph built successfully!")


if __name__ == "__main__":
    main()
