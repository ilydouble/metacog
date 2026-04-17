#!/usr/bin/env python3
"""Dump all memories directly from ChromaDB sqlite3 file"""

import sys
import json
import sqlite3
from pathlib import Path


def dump_collection_from_sqlite(db_path: Path, collection_name: str, output_file: Path):
    """Directly read ChromaDB sqlite3 and dump a collection"""
    print(f"  Reading collection '{collection_name}'...")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Find the collection id
    cur.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
    row = cur.fetchone()
    if not row:
        print(f"  ℹ️  Collection '{collection_name}' not found")
        output_file.write_text(f"Collection '{collection_name}' not found.\n")
        conn.close()
        return

    collection_id = row[0]

    # Get all embeddings (documents + metadata)
    cur.execute("""
        SELECT e.id, e.document, m.key, m.string_value
        FROM embeddings e
        LEFT JOIN embedding_metadata m ON e.id = m.id
        WHERE e.collection_id = ?
        ORDER BY e.id
    """, (collection_id,))
    rows = cur.fetchall()
    conn.close()

    # Group by embedding id
    entries = {}
    for emb_id, doc, key, val in rows:
        if emb_id not in entries:
            entries[emb_id] = {"id": emb_id, "content": doc, "metadata": {}}
        if key:
            entries[emb_id]["metadata"][key] = val

    count = len(entries)
    print(f"  Found {count} entries")

    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"# {collection_name} ({count} entries)\n\n")
        f.write("=" * 80 + "\n\n")

        for idx, entry in enumerate(entries.values(), 1):
            f.write(f"## Entry {idx}/{count}\n")
            f.write(f"ID: {entry['id']}\n")
            for k, v in entry["metadata"].items():
                f.write(f"{k}: {v}\n")
            f.write(f"\nContent:\n{entry['content']}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"  ✓ Written to {output_file}")


def main():
    if len(sys.argv) > 1:
        memory_path = Path(sys.argv[1])
    else:
        memory_path = Path("outputs/math_test_metacog_aime25/memu_db")

    db_path = memory_path / "chroma.sqlite3"
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print(f"Usage: python {sys.argv[0]} [memu_db_path]")
        return

    output_dir = Path("outputs/memory_dump")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Dumping memories from {db_path}...\n")

    print("📚 Semantic memory (math_lessons):")
    dump_collection_from_sqlite(db_path, "math_lessons", output_dir / "semantic_memory.txt")

    print()

    print("💡 Episodic memory (episodic_memory):")
    dump_collection_from_sqlite(db_path, "episodic_memory", output_dir / "episodic_memory.txt")

    print(f"\n✅ Done! Results in {output_dir}/")
    print(f"   - semantic_memory.txt")
    print(f"   - episodic_memory.txt")


if __name__ == "__main__":
    main()
