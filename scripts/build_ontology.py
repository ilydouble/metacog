#!/usr/bin/env python3
"""
Extract Math Knowledge Graph (Ontology) from Episodic Memories and Visualize it.
"""

import sys
import json
import sqlite3
import re
from pathlib import Path
import yaml
import litellm
import networkx as nx
from pyvis.network import Network
from dotenv import load_dotenv
import os

# Add src to path so we can import metacog
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))

# Load global env for API keys (e.g. ZAI_API_KEY)
env_path = Path.home() / "Library/Application Support/mini-swe-agent/.env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Fallback: litellm's zai provider expects ZAI_API_KEY, but user might have ZHIPUAI_API_KEY
if not os.getenv("ZAI_API_KEY") and os.getenv("ZHIPUAI_API_KEY"):
    os.environ["ZAI_API_KEY"] = os.environ["ZHIPUAI_API_KEY"]

litellm.set_verbose = False

def get_episodic_memories(db_path: Path) -> list:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id FROM collections WHERE name = 'episodic_memory'")
    row = cur.fetchone()
    if not row:
        print("ℹ️ episodic_memory collection not found.")
        return []
    
    collection_id = row[0]
    cur.execute("""
        SELECT e.id, f.c0
        FROM embeddings e
        JOIN segments s ON e.segment_id = s.id
        JOIN embedding_fulltext_search_content f ON e.id = f.id
        WHERE s.collection = ?
        ORDER BY e.id
    """, (collection_id,))
    doc_rows = cur.fetchall()

    entries = {emb_id: {"id": emb_id, "content": doc, "metadata": {}} for emb_id, doc in doc_rows}

    if entries:
        placeholders = ",".join("?" * len(entries))
        cur.execute(f"""
            SELECT id, key, string_value, int_value, float_value, bool_value
            FROM embedding_metadata
            WHERE id IN ({placeholders})
        """, list(entries.keys()))
        for emb_id, key, sval, ival, fval, bval in cur.fetchall():
            val = sval or ival or fval or bval
            entries[emb_id]["metadata"][key] = val

    conn.close()
    
    memories = []
    for entry in entries.values():
        if entry["metadata"].get("type") == "success_case":
            memories.append(entry)
    return memories

ONTOLOGY_SYSTEM_PROMPT = """\
# Role
You are a Mathematical Knowledge Graph Architect.
You will be provided with a list of successful math problem-solving cases (Episodic Memories).
Your task is to merge, abstract, and synthesize these discrete cases into a highly condensed and reusable Mathematical Ontology (Knowledge Graph).

# Ontology Schema
You must extract Nodes and Edges into the following JSON format.
Nodes should be highly abstract and reusable (e.g., merge similar techniques into one node).

{
  "nodes": [
    {
      "id": "<Unique Node ID, strictly alphanumeric without spaces, e.g., 'DomainGeometry'>",
      "label": "<Display Name, e.g., 'Geometry'>",
      "type": "<Must be exactly one of: 'Domain', 'ProblemType', 'Technique'>",
      "actionable_steps": "<OPTIONAL, ONLY for 'Technique' type: 1-2 sentence highly condensed, actionable execution steps or formulas to apply this technique>",
      "applicable_when": "<OPTIONAL, ONLY for 'Technique' type: A short description of the trigger condition for this technique to be applied, e.g., 'When the problem involves a line intersecting or touching a circle/conic section.'>"
    }
  ],
  "edges": [
    {
      "source": "<Source Node ID>",
      "target": "<Target Node ID>",
      "relation": "<Must be exactly one of: 'BELONGS_TO' (ProblemType -> Domain), 'SOLVED_BY' (ProblemType -> Technique)>"
    }
  ]
}

# CRITICAL Abstraction Rules (FAILURE TO FOLLOW WILL RESULT IN PENALTY)
1. Domain: MUST be limited to broad standard math fields (e.g., Combinatorics, Number Theory, Algebra, Geometry). Do NOT create many domains.
2. ProblemType: High-level abstract problem category. DO NOT use specific problem entities (e.g., use "Combinatorial Enumeration" instead of "Counting Latin Rectangles"). Limit to MAXIMUM 8-12 ProblemTypes in total. YOU MUST MERGE similar concepts.
3. Technique: The core algorithmic or mathematical method used. DO NOT use highly specific names (e.g., use "Symmetry Analysis" instead of "Tangential Trapezoid Properties"). Limit to MAXIMUM 10-15 Techniques in total.
   - For EACH Technique, you MUST provide 'actionable_steps' (e.g., "1. Move constants to one side. 2. Apply Simon's trick.") and 'applicable_when' (trigger condition, e.g., "When the problem involves a line intersecting or touching a circle/conic section.").
4. MAXIMUM REUSE: You MUST merge multiple cases into the same ProblemType or Technique. If you have 20 cases, you should NOT have 20 ProblemTypes. You should have 5-8 ProblemTypes, each connected to multiple Techniques.
5. The goal is a highly interconnected graph, not a flat list of 1-to-1 mappings. Every Technique should ideally be connected to multiple ProblemTypes.

# Output format
CRITICAL: Output ONLY the raw JSON block inside ```json tags. Do NOT output any thought process, explanation, or preamble. Your entire response must be just the JSON block. All content MUST be in English.
"""

def extract_ontology(memories: list, model_config) -> dict:
    if not memories:
        print("No memories to process.")
        return {}

    cases_text = ""
    for i, mem in enumerate(memories, 1):
        meta = mem["metadata"]
        cases_text += f"Case {i}:\n"
        cases_text += f"- Problem Type: {meta.get('problem_type', '')}\n"
        cases_text += f"- Tags: {meta.get('tags', '')}\n"
        cases_text += f"- Key Insight: {meta.get('key_insight', '')}\n"
        cases_text += f"- Approach: {meta.get('approach', '')}\n\n"

    print(f"Calling LLM ({model_config}) to extract ontology from {len(memories)} cases...")

    if isinstance(model_config, dict):
        model_name = model_config.get("model")
        api_base = model_config.get("api_base")
        api_key = model_config.get("api_key")
    else:
        model_name = model_config
        api_base = None
        api_key = None

    response = litellm.completion(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        messages=[
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Cases:\n{cases_text}"}
        ],
        temperature=0.0,
        max_tokens=8192,
    )
    
    output = response.choices[0].message.content
    
    clean = output.strip()
    m = re.search(r"```json\s*(.*?)\s*```", clean, re.DOTALL)
    if m:
        clean = m.group(1)
    else:
        # Fallback: try to find the first { and last }
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean = clean[start:end+1]

    try:
        parsed = json.loads(clean)
        if not parsed or "nodes" not in parsed:
            print(f"⚠️ Parsed JSON doesn't contain 'nodes'. Raw LLM Output:\n{output}")
        return parsed
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}\nRaw LLM Output:\n{output}")
        return {}

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

def build_ontology_vector_db(ontology: dict, db_dir: Path):
    from metacog.memory.memu_client import MemUClient

    if not ontology or "nodes" not in ontology:
        return

    print(f"\n🧠 Building Ontology Vector Database at {db_dir} ...")
    client = MemUClient(collection_name="ontology_memory", persist_dir=db_dir)

    # Optional: clear existing if we want a fresh start, but MemUClient doesn't expose a clear method directly.
    # We will just add them; Chroma handles deduplication by ID if we provide them, or we can just delete the whole dir before this.

    nodes = ontology.get("nodes", [])
    edges = ontology.get("edges", [])

    domains = {n["id"]: n.get("label", n["id"]) for n in nodes if n.get("type") == "Domain"}
    prob_types = {n["id"]: n.get("label", n["id"]) for n in nodes if n.get("type") == "ProblemType"}
    techs = {n["id"]: n for n in nodes if n.get("type") == "Technique"}

    domain_to_probs = {d: [] for d in domains}
    prob_to_techs = {p: [] for p in prob_types}
    unmapped_probs = set(prob_types.keys())

    for e in edges:
        src, tgt, rel = e.get("source"), e.get("target"), e.get("relation", "")
        if rel == "BELONGS_TO" and src in prob_types and tgt in domains:
            domain_to_probs[tgt].append(src)
            unmapped_probs.discard(src)
        elif rel == "SOLVED_BY" and src in prob_types and tgt in techs:
            prob_to_techs[src].append(tgt)

    count = 0
    # Process mapped problem types
    for d_id, probs in domain_to_probs.items():
        domain_label = domains[d_id]
        for p_id in probs:
            p_label = prob_types[p_id]
            t_ids = prob_to_techs[p_id]

            content = f"Domain: {domain_label}\nProblem Type: {p_label}\n"
            for t_id in t_ids:
                if t_id in techs:
                    t = techs[t_id]
                    content += f"- Technique: {t.get('label', t_id)}\n"
                    if "applicable_when" in t:
                        content += f"  Condition: {t['applicable_when']}\n"
                    if "actionable_steps" in t:
                        content += f"  Steps: {t['actionable_steps']}\n"

            client.add_memory(
                content=content,
                metadata={"type": "ontology_node", "problem_type": p_label, "domain": domain_label},
                memory_id=p_id
            )
            count += 1

    # Process unmapped problem types
    for p_id in unmapped_probs:
        p_label = prob_types[p_id]
        t_ids = prob_to_techs[p_id]

        content = f"Problem Type: {p_label}\n"
        for t_id in t_ids:
            if t_id in techs:
                t = techs[t_id]
                content += f"- Technique: {t.get('label', t_id)}\n"
                if "applicable_when" in t:
                    content += f"  Condition: {t['applicable_when']}\n"
                if "actionable_steps" in t:
                    content += f"  Steps: {t['actionable_steps']}\n"

        client.add_memory(
            content=content,
            metadata={"type": "ontology_node", "problem_type": p_label},
            memory_id=p_id
        )
        count += 1

    print(f"✅ Successfully vectorized and stored {count} ontology ProblemType nodes!")

def main():
    import os
    # Force GLM-4 configuration for ontology extraction using the OpenAI-compatible endpoint
    api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZAI_API_KEY")
    if not api_key:
        print("⚠️ ZHIPUAI_API_KEY not found in environment. Ontology extraction might fail.")

    model_config = {
        "model": "openai/glm-4.7",  # GLM-4 series via standard OpenAI format
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": api_key or "sk-dummy-key"
    }

    db_path = Path("outputs/math_test_metacog_aime25_2/memu_db/chroma.sqlite3")
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1]) / "chroma.sqlite3"

    if not db_path.exists():
        db_path = Path("outputs/math_test_metacog_aime25/memu_db/chroma.sqlite3")

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return

    output_dir = Path("outputs/ontology")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Extracting episodic memories from {db_path}...")
    memories = get_episodic_memories(db_path)
    
    ontology = extract_ontology(memories, model_config)
    
    json_out = output_dir / "ontology.json"
    with json_out.open("w") as f:
        json.dump(ontology, f, indent=2)
    print(f"Ontology JSON saved to {json_out}")

    html_out = output_dir / "ontology_graph.html"
    visualize_ontology(ontology, html_out)

    ontology_db_dir = output_dir / "memu_db"
    build_ontology_vector_db(ontology, ontology_db_dir)

if __name__ == "__main__":
    main()
