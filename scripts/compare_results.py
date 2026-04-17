#!/usr/bin/env python3
"""Compare two test runs and show differences"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Add scripts/utils to path for answer comparison
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from evaluation import compare_answers


def load_results_from_trajectories(traj_dir: Path) -> dict:
    """Load results from trajectory files"""
    results = {}

    for traj_file in traj_dir.glob("*.traj.json"):
        try:
            with traj_file.open() as f:
                data = json.load(f)

            problem_id = traj_file.stem.replace(".traj", "")
            info = data.get("info", {})

            # Extract answer from submission
            submission = info.get("submission", "")

            # Clean submission: take the last non-empty line that looks like a number
            extracted_answer = None
            if submission:
                import re
                lines = [l.strip() for l in submission.splitlines() if l.strip()]
                for line in reversed(lines):
                    if re.match(r'^-?\d+\.?\d*$', line) or re.match(r'^-?\d+/\d+$', line):
                        extracted_answer = line
                        break

            results[problem_id] = {
                "id": problem_id,
                "extracted_answer": extracted_answer,
                "exit_status": info.get("exit_status", "Unknown"),
            }
        except Exception as e:
            print(f"Warning: Failed to read {traj_file.name}: {e}")

    return results


def compare_runs(baseline_dir: Path, metacog_dir: Path):
    """Compare baseline and metacog runs"""
    baseline_results = load_results_from_trajectories(baseline_dir / "trajectories")
    metacog_results = load_results_from_trajectories(metacog_dir / "trajectories")

    # Load expected answers from results.jsonl (if exists)
    expected_answers = {}
    results_file = baseline_dir / "results.jsonl"
    if results_file.exists():
        with results_file.open() as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    pid = entry.get("id", "")
                    expected_answers[pid] = entry.get("expected_answer", "")

    # Ensure same problem set
    all_ids = sorted(set(baseline_results.keys()) | set(metacog_results.keys()))
    
    print(f"📊 Comparison: {baseline_dir.name} vs {metacog_dir.name}\n")
    print(f"Total problems: {len(all_ids)}\n")
    
    baseline_correct = 0
    metacog_correct = 0
    improvements = []
    regressions = []
    
    for pid in all_ids:
        b = baseline_results.get(pid, {})
        m = metacog_results.get(pid, {})

        expected = expected_answers.get(pid, "")

        # Determine if passed by comparing with expected answer
        b_ans = b.get("extracted_answer", "")
        m_ans = m.get("extracted_answer", "")

        # Use compare_answers for proper numeric comparison (handles 468 vs 468.0)
        b_passed = compare_answers(b_ans, expected) if expected and b_ans else False
        m_passed = compare_answers(m_ans, expected) if expected and m_ans else False
        
        if b_passed:
            baseline_correct += 1
        if m_passed:
            metacog_correct += 1
        
        if not b_passed and m_passed:
            improvements.append({
                "id": pid,
                "baseline_answer": b_ans or "N/A",
                "metacog_answer": m_ans or "N/A",
                "expected": expected or "N/A",
            })
        elif b_passed and not m_passed:
            regressions.append({
                "id": pid,
                "baseline_answer": b_ans or "N/A",
                "metacog_answer": m_ans or "N/A",
                "expected": expected or "N/A",
                "metacog_exit": m.get("exit_status", "Unknown"),
            })
    
    # Summary
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline correct: {baseline_correct}/{len(all_ids)} ({baseline_correct/len(all_ids)*100:.1f}%)")
    print(f"Metacog correct:  {metacog_correct}/{len(all_ids)} ({metacog_correct/len(all_ids)*100:.1f}%)")
    print(f"\nNet change: {metacog_correct - baseline_correct:+d}")
    print(f"  Improvements: {len(improvements)}")
    print(f"  Regressions:  {len(regressions)}")
    print()
    
    # Improvements
    if improvements:
        print(f"{'='*70}")
        print(f"IMPROVEMENTS ({len(improvements)})")
        print(f"{'='*70}")
        for item in improvements:
            print(f"{item['id']:8} | Expected: {item['expected']:10} | "
                  f"Baseline: {str(item['baseline_answer']):10} → Metacog: {str(item['metacog_answer']):10} ✅")
        print()
    
    # Regressions
    if regressions:
        print(f"{'='*70}")
        print(f"REGRESSIONS ({len(regressions)})")
        print(f"{'='*70}")
        for item in regressions:
            print(f"{item['id']:8} | Expected: {item['expected']:10} | "
                  f"Baseline: {str(item['baseline_answer']):10} → Metacog: {str(item['metacog_answer']):10} ❌")
            print(f"           Exit: {item['metacog_exit']}")
        print()
    
    # Detailed comparison table
    print(f"{'='*70}")
    print(f"DETAILED COMPARISON")
    print(f"{'='*70}")
    print(f"{'ID':8} | {'Baseline':10} | {'Metacog':10} | {'Expected':10} | {'Status':8}")
    print(f"{'-'*70}")
    
    for pid in all_ids:
        b = baseline_results.get(pid, {})
        m = metacog_results.get(pid, {})

        expected = expected_answers.get(pid, "")
        b_ans_full = b.get("extracted_answer", "")
        m_ans_full = m.get("extracted_answer", "")

        b_ans = str(b_ans_full or "—")[:10]
        m_ans = str(m_ans_full or "—")[:10]
        expected_short = str(expected or "—")[:10]

        # Use compare_answers for proper numeric comparison
        b_passed = compare_answers(b_ans_full, expected) if expected and b_ans_full else False
        m_passed = compare_answers(m_ans_full, expected) if expected and m_ans_full else False
        
        if b_passed == m_passed:
            if b_passed:
                status = "Both ✓"
            else:
                status = "Both ✗"
        elif m_passed:
            status = "Improved"
        else:
            status = "Regressed"
        
        print(f"{pid:8} | {b_ans:10} | {m_ans:10} | {expected_short:10} | {status:8}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <baseline_dir> <metacog_dir>")
        print("Example: python compare_results.py outputs/math_test_aime25 outputs/math_test_metacog_aime25")
        sys.exit(1)
    
    baseline_dir = Path(sys.argv[1])
    metacog_dir = Path(sys.argv[2])
    
    if not baseline_dir.exists() or not (baseline_dir / "trajectories").exists():
        print(f"❌ Baseline trajectories not found: {baseline_dir}/trajectories")
        sys.exit(1)

    if not metacog_dir.exists() or not (metacog_dir / "trajectories").exists():
        print(f"❌ Metacog trajectories not found: {metacog_dir}/trajectories")
        sys.exit(1)
    
    compare_runs(baseline_dir, metacog_dir)


if __name__ == "__main__":
    main()
