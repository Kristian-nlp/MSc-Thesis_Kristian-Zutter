"""
migrate_instagram_post_ids.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    One-off migration to convert Instagram numeric post_ids to
    shortcode-based IDs. Resolves the inconsistency where the same
    post could appear under both instagram_<shortcode> (from DOM)
    and instagram_<numeric_pk> (from JSON) by extracting the
    shortcode from each permalink and rewriting post_id across all
    referencing tables.

Inputs:
    04_database/scraper.db            posts.permalink + numeric IDs
                                      across all features and counters
                                      foreign-key tables

Outputs:
    04_database/scraper.db            posts.post_id and matching
                                      FK rows rewritten in place

Usage:
    python 04_database/migrate_instagram_post_ids.py
"""

import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "scraper.db"

# Tables with post_id foreign keys
FK_TABLES = ["captures", "counters", "features_text", "features_style",
             "features_visual", "features_temporal", "features_audio"]

SHORTCODE_RE = re.compile(r"/(?:p|reel)/([A-Za-z0-9_-]+)/?")


def extract_shortcode(permalink: str) -> str | None:
    m = SHORTCODE_RE.search(permalink)
    return m.group(1) if m else None


def migrate():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = OFF")  # Disable FK checks during migration

    cursor = conn.execute(
        "SELECT post_id, permalink FROM posts "
        "WHERE platform = 'instagram' AND post_id GLOB 'instagram_[0-9]*'"
    )
    rows = cursor.fetchall()

    if not rows:
        print("No numeric Instagram post_ids found. Nothing to migrate.")
        return

    print(f"Found {len(rows)} numeric Instagram post_ids to migrate.")

    migrated = 0
    skipped = 0

    for old_id, permalink in rows:
        shortcode = extract_shortcode(permalink)
        if not shortcode:
            print(f"  SKIP: no shortcode in permalink for {old_id}: {permalink}")
            skipped += 1
            continue

        new_id = f"instagram_{shortcode}"

        # Check if new_id already exists (duplicate post)
        existing = conn.execute(
            "SELECT post_id FROM posts WHERE post_id = ?", (new_id,)
        ).fetchone()

        if existing:
            # The shortcode-based version already exists. Merge: re-point
            # captures/counters from old_id to new_id, then delete old post row.
            print(f"  MERGE: {old_id} -> {new_id} (shortcode version exists)")
            for table in FK_TABLES:
                # For tables with UNIQUE constraints, skip duplicates
                conn.execute(
                    f"UPDATE OR IGNORE {table} SET post_id = ? WHERE post_id = ?",
                    (new_id, old_id),
                )
                # Delete any remaining rows that couldn't be updated due to UNIQUE
                conn.execute(
                    f"DELETE FROM {table} WHERE post_id = ?", (old_id,)
                )
            conn.execute("DELETE FROM posts WHERE post_id = ?", (old_id,))
        else:
            # Simple rename
            print(f"  RENAME: {old_id} -> {new_id}")
            conn.execute(
                "UPDATE posts SET post_id = ? WHERE post_id = ?",
                (new_id, old_id),
            )
            for table in FK_TABLES:
                conn.execute(
                    f"UPDATE {table} SET post_id = ? WHERE post_id = ?",
                    (new_id, old_id),
                )
        migrated += 1

    conn.commit()

    # Verify
    remaining = conn.execute(
        "SELECT COUNT(*) FROM posts "
        "WHERE platform = 'instagram' AND post_id GLOB 'instagram_[0-9]*'"
    ).fetchone()[0]

    conn.execute("PRAGMA foreign_keys = ON")
    # Run FK integrity check
    fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    conn.close()

    print(f"\nMigration complete: {migrated} migrated, {skipped} skipped.")
    print(f"Remaining numeric post_ids: {remaining}")
    if fk_violations:
        print(f"WARNING: {len(fk_violations)} FK violations found!")
        for v in fk_violations[:10]:
            print(f"  {v}")
    else:
        print("FK integrity check: PASSED")


if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    # Backup first
    backup = DB_PATH.with_suffix(".db.bak_pre_postid_migration")
    if not backup.exists():
        import shutil
        shutil.copy2(DB_PATH, backup)
        print(f"Backup created: {backup}")

    migrate()
