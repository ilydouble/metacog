#!/usr/bin/env python3
"""
Build Math Knowledge Graph Vector DB from a Pre-defined Ontology JSON.

Top-Down design: the ontology schema is defined in data/math_ontology.json.
This script vectorizes it into a ChromaDB collection for runtime Graph RAG retrieval.

Usage:
    python scripts/build_ontology.py                          # use default data/math_ontology.json
    python scripts/build_ontology.py --ontology path/to.json  # use custom ontology
    python scripts/build_ontology.py --output-dir outputs/my_ontology
"""

import sys
import json
import argparse
from pathlib import Path

from pyvis.network import Network

# Add src to path so we can import metacog
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))

DEFAULT_ONTOLOGY = _root / "data" / "math_ontology.json"


def load_ontology(ontology_path: Path) -> dict:
    """Load the pre-defined ontology JSON."""
    if not ontology_path.exists():
        print(f"❌ Ontology file not found: {ontology_path}")
        sys.exit(1)
    with ontology_path.open() as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    print(f"✅ Loaded ontology: {len(nodes)} nodes, {len(edges)} edges from {ontology_path}")
    return data


def visualize_ontology(ontology: dict, output_file: Path):
    if not ontology or "nodes" not in ontology:
        print("Invalid ontology data.")
        return

    net = Network(height="800px", width="100%", directed=True, bgcolor="#222222", font_color="white")
    net.force_atlas_2based()

    color_map = {
        "Domain": "#FF5733",
        "ProblemType": "#33FF57",
        "Technique": "#3357FF"
    }

    for node in ontology.get("nodes", []):
        node_id = node.get("id")
        label = node.get("label", node_id)
        n_type = node.get("type", "Unknown")
        color = color_map.get(n_type, "#FFFFFF")

        # Build tooltip title
        hover_title = f"Type: {n_type}"
        if n_type == "Technique":
            if "applicable_when" in node:
                hover_title += f"\n\nCondition:\n{node['applicable_when']}"
            if "actionable_steps" in node:
                hover_title += f"\n\nSteps:\n{node['actionable_steps']}"

        net.add_node(node_id, label=label, title=hover_title, color=color, shape="dot", size=25 if n_type=="Domain" else (20 if n_type=="ProblemType" else 15))

    for edge in ontology.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        relation = edge.get("relation", "")
        net.add_edge(source, target, title=relation, label=relation, color="#888888")

    net.save_graph(str(output_file))
    print(f"Visualization saved to {output_file}")

def build_ontology_vector_db(ontology: dict, db_dir: Path, clear: bool = False) -> int:
    """Vectorize ontology ProblemType nodes into ChromaDB.

    Each ProblemType node is stored as one vector document containing its
    domain, problem type label, and all associated techniques with their
    applicable_when / actionable_steps fields.  evidence_count starts at 0
    and is updated at runtime by SuccessAnalyzer.

    Returns the number of nodes stored.
    """
    from metacog.memory.memu_client import MemUClient

    if not ontology or "nodes" not in ontology:
        print("⚠️  Empty ontology, nothing to vectorize.")
        return 0

    print(f"\n🧠 Building Ontology Vector Database at {db_dir} ...")
    client = MemUClient(collection_name="ontology_memory", persist_dir=db_dir)

    if clear:
        client.clear()
        print("   Cleared existing ontology_memory collection.")

    nodes = ontology.get("nodes", [])
    edges = ontology.get("edges", [])

    domains    = {n["id"]: n.get("label", n["id"]) for n in nodes if n.get("type") == "Domain"}
    prob_types = {n["id"]: n.get("label", n["id"]) for n in nodes if n.get("type") == "ProblemType"}
    techs      = {n["id"]: n for n in nodes if n.get("type") == "Technique"}

    domain_to_probs = {d: [] for d in domains}
    prob_to_techs   = {p: [] for p in prob_types}
    unmapped_probs  = set(prob_types.keys())

    for e in edges:
        src, tgt, rel = e.get("source"), e.get("target"), e.get("relation", "")
        if rel == "BELONGS_TO" and src in prob_types and tgt in domains:
            domain_to_probs[tgt].append(src)
            unmapped_probs.discard(src)
        elif rel == "SOLVED_BY" and src in prob_types and tgt in techs:
            prob_to_techs[src].append(tgt)

    def _store_node(p_id: str, p_label: str, domain_label: str | None):
        t_ids = prob_to_techs.get(p_id, [])
        content = (f"Domain: {domain_label}\n" if domain_label else "") + f"Problem Type: {p_label}\n"
        for t_id in t_ids:
            if t_id in techs:
                t = techs[t_id]
                content += f"- Technique: {t.get('label', t_id)}\n"
                if "applicable_when" in t:
                    content += f"  Condition: {t['applicable_when']}\n"
                if "actionable_steps" in t:
                    content += f"  Steps: {t['actionable_steps']}\n"
        meta = {
            "type": "ontology_node",
            "problem_type": p_label,
            "evidence_count": 0,   # updated at runtime by SuccessAnalyzer
        }
        if domain_label:
            meta["domain"] = domain_label
        client.add_memory(content=content, metadata=meta, memory_id=p_id)

    count = 0
    for d_id, probs in domain_to_probs.items():
        for p_id in probs:
            _store_node(p_id, prob_types[p_id], domains[d_id])
            count += 1
    for p_id in unmapped_probs:
        _store_node(p_id, prob_types[p_id], None)
        count += 1

    print(f"✅ Vectorized and stored {count} ontology ProblemType nodes (evidence_count=0).")
    return count


def main():
    parser = argparse.ArgumentParser(description="Build ontology vector DB from pre-defined JSON.")
    parser.add_argument(
        "--ontology", type=Path, default=DEFAULT_ONTOLOGY,
        help=f"Path to the ontology JSON file (default: {DEFAULT_ONTOLOGY})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ontology"),
        help="Output directory for the vector DB and visualization (default: outputs/ontology)"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear existing ontology_memory collection before writing"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip HTML graph visualization"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pre-defined ontology
    ontology = load_ontology(args.ontology)

    # 2. Visualize (optional)
    if not args.no_viz:
        html_out = output_dir / "ontology_graph.html"
        visualize_ontology(ontology, html_out)

    # 3. Vectorize into ChromaDB
    ontology_db_dir = output_dir / "memu_db"
    build_ontology_vector_db(ontology, ontology_db_dir, clear=args.clear)


if __name__ == "__main__":
    main()
