"""
Bio Scoring Example — Exploring bio-inspired memory decay.

This demonstrates the bio-inspired vitality scoring system:
1. How significance affects bio score
2. How access frequency (touch) affects bio score
3. How tier classification works based on score
4. Score decay over time (simulated)

Run with: python examples/bio_scoring.py
"""

import os
import tempfile
import time

import spider


def main():
    with tempfile.TemporaryDirectory(prefix="spider_bio_scoring_") as tmpdir:
        db_path = os.path.join(tmpdir, "bio_test")

        print("Bio-Inspired Memory Scoring in Spider\n")
        print("=" * 50)

        with spider.Spider.open(db_path) as db:
            # ---------------------------------------------------------------
            # 1. Create nodes with different significance values
            # ---------------------------------------------------------------
            print("\n1. Creating nodes with different significance...\n")

            request = spider.IngestRequest(
                title="Bio Scoring Test",
                propositions=[
                    spider.Proposition(
                        "High significance node",
                        entities=[spider.Entity("Important", "CONCEPT")],
                    ),
                    spider.Proposition(
                        "Medium significance node",
                        entities=[spider.Entity("Moderate", "CONCEPT")],
                    ),
                    spider.Proposition(
                        "Low significance node",
                        entities=[spider.Entity("Minor", "CONCEPT")],
                    ),
                ],
            )
            result = db.index(request)
            doc_id = result.document_id

            # Find proposition nodes
            prop_nodes = db.find_by_label("PROPOSITION")
            print(f"  Created {len(prop_nodes)} proposition nodes")

            # Set different significance values
            if len(prop_nodes) >= 3:
                high_sig_id = prop_nodes[0]
                med_sig_id = prop_nodes[1]
                low_sig_id = prop_nodes[2]

                db.set_significance(high_sig_id, 255)  # Max significance
                db.set_significance(med_sig_id, 128)   # Medium
                db.set_significance(low_sig_id, 10)    # Low

                print("\n2. Bio scores by significance...\n")
                print(f"  High sig (255):  {db.get_bio_score(high_sig_id):.2f}  -> {db.get_bio_tier(high_sig_id)}")
                print(f"  Med sig (128):   {db.get_bio_score(med_sig_id):.2f}  -> {db.get_bio_tier(med_sig_id)}")
                print(f"  Low sig (10):    {db.get_bio_score(low_sig_id):.2f}  -> {db.get_bio_tier(low_sig_id)}")

            # ---------------------------------------------------------------
            # 3. Effect of touching nodes (access frequency)
            # ---------------------------------------------------------------
            print("\n3. Effect of touching nodes...\n")

            if len(prop_nodes) >= 1:
                touch_id = prop_nodes[0]
                score_before = db.get_bio_score(touch_id)
                print(f"  Score before touch: {score_before:.2f}")

                # Touch multiple times
                for i in range(5):
                    count = db.node_touch(touch_id)
                    score = db.get_bio_score(touch_id)
                    print(f"  After touch #{i+1} (access_count={count}): {score:.2f}")

            # ---------------------------------------------------------------
            # 4. Tier classification thresholds
            # ---------------------------------------------------------------
            print("\n4. BioTier classification thresholds...\n")

            print("  Tier classification based on score:")
            print(f"    Score 100.0 -> {spider.BioTier.from_score(100.0)} (is_active={spider.BioTier.from_score(100.0).is_active()}, is_prunable={spider.BioTier.from_score(100.0).is_prunable()})")
            print(f"    Score 25.0  -> {spider.BioTier.from_score(25.0)} (is_active={spider.BioTier.from_score(25.0).is_active()}, is_prunable={spider.BioTier.from_score(25.0).is_prunable()})")
            print(f"    Score 10.0  -> {spider.BioTier.from_score(10.0)} (is_active={spider.BioTier.from_score(10.0).is_active()}, is_prunable={spider.BioTier.from_score(10.0).is_prunable()})")
            print(f"    Score 5.0   -> {spider.BioTier.from_score(5.0)} (is_active={spider.BioTier.from_score(5.0).is_active()}, is_prunable={spider.BioTier.from_score(5.0).is_prunable()})")
            print(f"    Score 1.0   -> {spider.BioTier.from_score(1.0)} (is_active={spider.BioTier.from_score(1.0).is_active()}, is_prunable={spider.BioTier.from_score(1.0).is_prunable()})")
            print(f"    Score 0.0   -> {spider.BioTier.from_score(0.0)} (is_active={spider.BioTier.from_score(0.0).is_active()}, is_prunable={spider.BioTier.from_score(0.0).is_prunable()})")
            print(f"    Score -5.0  -> {spider.BioTier.from_score(-5.0)} (is_active={spider.BioTier.from_score(-5.0).is_active()}, is_prunable={spider.BioTier.from_score(-5.0).is_prunable()})")

            # ---------------------------------------------------------------
            # 5. Node count and summary
            # ---------------------------------------------------------------
            print(f"\n5. Total nodes: {db.node_count()}")

            print("\n" + "=" * 50)
            print("Bio scoring exploration complete!")
            print("\nKey takeaways:")
            print("  - Higher significance = higher bio score")
            print("  - More access (touch) = higher bio score")
            print("  - Bio tiers classify nodes by vitality:")
            print("    Hot > 20, Warm > 5, Cold > 0, Pruned <= 0")


if __name__ == "__main__":
    main()
