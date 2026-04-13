"""
Quick Start Example — Basic open, ingest, query.

This demonstrates the core Spider API:
1. Opening a database
2. Ingesting a document with entities
3. Querying by label and property
4. Traversing the graph
5. Checking bio scores

Run with: python examples/quick_start.py
"""

import os
import tempfile

import spider


def main():
    # Use a temporary directory for this example
    with tempfile.TemporaryDirectory(prefix="spider_quick_START_") as tmpdir:
        db_path = os.path.join(tmpdir, "my_graph")

        # 1. Open a database (creates it if it doesn't exist)
        print("1. Opening database...")
        with spider.Spider.open(db_path) as db:
            print(f"   Database path: {db.path}")

            # 2. Ingest a document with propositions and entities
            print("\n2. Ingesting document...")
            request = spider.IngestRequest(
                title="Spider — Bio-Inspired Memory",
                propositions=[
                    spider.Proposition(
                        "Spider is a graph database for AI agent memory",
                        entities=[
                            spider.Entity("Spider", "PRODUCT"),
                            spider.Entity("AI Agent", "CONCEPT"),
                        ],
                    ),
                    spider.Proposition(
                        "Spider uses bio-inspired scoring for memory decay",
                        entities=[
                            spider.Entity("Spider", "PRODUCT"),
                            spider.Entity("Bio-Inspired Scoring", "CONCEPT"),
                        ],
                    ),
                ],
            )
            result = db.index(request)
            print(f"   Document ID: {result.document_id}")
            print(f"   Propositions: {result.proposition_count}")
            print(f"   Entities: {result.entity_count}")
            print(f"   Edges: {result.edge_count}")

            # 3. Query by label
            print("\n3. Querying by label...")
            docs = db.find_by_label("DOCUMENT")
            print(f"   Documents: {len(docs)}")

            props = db.find_by_label("PROPOSITION")
            print(f"   Propositions: {len(props)}")

            entities = db.find_by_label("ENTITY")
            print(f"   Entities: {len(entities)}")

            # 4. Query by property
            print("\n4. Querying by property...")
            spider_nodes = db.find_by_property("name", "Spider")
            print(f"   Nodes named 'Spider': {len(spider_nodes)}")

            first_spider = db.find_one_by_property("name", "Spider")
            if first_spider:
                print(f"   First 'Spider' node: {first_spider}")

            # 5. Traverse the graph
            print("\n5. Traversing the graph...")
            if docs:
                doc_id = docs[0]
                neighbors = db.get_neighbors(doc_id, spider.Direction.OUTGOING)
                print(f"   Document neighbors (outgoing): {len(neighbors)}")
                for n in neighbors:
                    print(f"     -> Node {n.node_id} via Edge {n.edge_id}")

                # Get relationships as dicts
                rels = db.get_relationships(doc_id, spider.Direction.OUTGOING)
                print(f"   Document relationships: {len(rels)}")
                for rel in rels:
                    print(f"     Source {rel['source_id']} -> Target {rel['target_id']}")

            # 6. Check bio scores
            print("\n6. Bio scores...")
            for doc_id in docs:
                score = db.get_bio_score(doc_id)
                tier = db.get_bio_tier(doc_id)
                print(f"   Document {doc_id}: score={score:.2f}, tier={tier}")

            for prop_id in props:
                score = db.get_bio_score(prop_id)
                tier = db.get_bio_tier(prop_id)
                print(f"   Proposition {prop_id}: score={score:.2f}, tier={tier}")

            # 7. Node operations
            print("\n7. Node operations...")
            print(f"   Node count: {db.node_count()}")

            if docs:
                doc_id = docs[0]
                count_before = db.node_touch(doc_id)
                print(f"   Touched document {doc_id}, access_count={count_before}")

                db.set_significance(doc_id, 200)
                new_score = db.get_bio_score(doc_id)
                print(f"   New bio score after significance=200: {new_score:.2f}")

            print("\nDone! Database closed automatically by context manager.")


if __name__ == "__main__":
    main()
