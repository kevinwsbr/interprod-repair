#!/usr/bin/env python3
"""Check whether high-confidence call relationships are supported by diff changes.

This script links entries from call_relationships_high_confidence.json to
interprocedural_vulnerabilities.json and verifies whether called method names
appear in changed diff hunks for file pairs.
"""

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Any


HIGH_CONF_FILE = "call_relationships_high_confidence.json"
INTERPROC_FILE = "interprocedural_vulnerabilities.json"
OUT_FILE = "call_diff_impact_high_confidence.json"

# Ignore obvious language keywords and noise captured as "methods".
NOISE_METHODS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
}


def pair_key(cve_id: str, file1: str, file2: str) -> Tuple[str, str, str]:
    """Build normalized key for a CVE and unordered file pair."""
    a, b = sorted([file1 or "", file2 or ""])
    return (cve_id, a, b)


def changed_lines_from_diff(diff_text: str) -> List[str]:
    """Extract changed hunk lines from a unified diff."""
    if not diff_text:
        return []
    lines: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            content = line[1:].strip()
            if content:
                lines.append(content)
    return lines


def method_in_lines(method: str, lines: List[str]) -> bool:
    """Check if method token appears in changed lines."""
    if not method or method in NOISE_METHODS:
        return False
    token = re.compile(rf"\b{re.escape(method)}\b")
    return any(token.search(line) for line in lines)


def build_interproc_index(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """Index interprocedural records by CVE and unordered pair."""
    index: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        files = rec.get("files") or []
        if len(files) != 2:
            continue
        f1 = files[0].get("filename", "")
        f2 = files[1].get("filename", "")
        key = pair_key(rec.get("cve_id", ""), f1, f2)
        index[key].append(rec)
    return index


def choose_best_match(candidates: List[Dict[str, Any]], file1: str, file2: str) -> Dict[str, Any]:
    """Pick candidate whose file order matches; otherwise first candidate."""
    if not candidates:
        return {}
    for cand in candidates:
        files = cand.get("files") or []
        if len(files) != 2:
            continue
        if files[0].get("filename") == file1 and files[1].get("filename") == file2:
            return cand
    return candidates[0]


def analyze() -> Dict[str, Any]:
    """Run diff-impact analysis for high-confidence call relationships."""
    with open(HIGH_CONF_FILE, "r") as f:
        high_conf = json.load(f)

    with open(INTERPROC_FILE, "r") as f:
        interproc = json.load(f)

    interproc_index = build_interproc_index(interproc)

    results: List[Dict[str, Any]] = []
    summary = Counter()

    for rel in high_conf:
        cve_id = rel.get("cve_id", "")
        file1 = rel.get("file1_path", "")
        file2 = rel.get("file2_path", "")
        key = pair_key(cve_id, file1, file2)

        candidates = interproc_index.get(key, [])
        matched = choose_best_match(candidates, file1, file2)

        if not matched:
            summary["missing_interproc_match"] += 1
            results.append(
                {
                    "cve_id": cve_id,
                    "file1_path": file1,
                    "file2_path": file2,
                    "status": "missing_interprocedural_record",
                    "impact_class": "unknown",
                }
            )
            continue

        files = matched.get("files") or []
        if len(files) != 2:
            summary["invalid_matched_record"] += 1
            continue

        # Map matched files by filename for robust lookup.
        by_name = {f.get("filename", ""): f for f in files}
        f1_rec = by_name.get(file1, files[0])
        f2_rec = by_name.get(file2, files[1])

        f1_changed = changed_lines_from_diff(f1_rec.get("diff") or "")
        f2_changed = changed_lines_from_diff(f2_rec.get("diff") or "")

        methods = {
            call.get("method")
            for call in (rel.get("method_calls") or [])
            if call.get("method") and call.get("method") not in NOISE_METHODS
        }

        methods_in_f1 = sorted([m for m in methods if method_in_lines(m, f1_changed)])
        methods_in_f2 = sorted([m for m in methods if method_in_lines(m, f2_changed)])
        methods_in_both = sorted(set(methods_in_f1).intersection(methods_in_f2))

        if methods_in_both:
            impact_class = "strong_cross_file_impact"
        elif methods_in_f1 or methods_in_f2:
            impact_class = "partial_impact"
        else:
            impact_class = "no_called_method_in_diff"

        summary[impact_class] += 1
        summary["total_high_conf"] += 1

        results.append(
            {
                "cve_id": cve_id,
                "file1_path": file1,
                "file2_path": file2,
                "impact_class": impact_class,
                "called_methods_considered": sorted(methods),
                "called_methods_in_file1_diff": methods_in_f1,
                "called_methods_in_file2_diff": methods_in_f2,
                "called_methods_in_both_diffs": methods_in_both,
                "file1_changed_lines": len(f1_changed),
                "file2_changed_lines": len(f2_changed),
            }
        )

    payload = {
        "summary": dict(summary),
        "results": results,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


def main() -> None:
    """Execute analysis and print concise summary."""
    payload = analyze()
    summary = payload["summary"]

    total = summary.get("total_high_conf", 0)
    strong = summary.get("strong_cross_file_impact", 0)
    partial = summary.get("partial_impact", 0)
    none = summary.get("no_called_method_in_diff", 0)
    missing = summary.get("missing_interproc_match", 0)

    print("\n=== Call-to-Diff Impact Summary ===")
    print(f"Total high-confidence pairs analyzed: {total}")
    print(f"Strong cross-file impact: {strong}")
    print(f"Partial impact: {partial}")
    print(f"No called method in diff hunks: {none}")
    if missing:
        print(f"Missing interprocedural match: {missing}")
    print(f"Saved detailed report to {OUT_FILE}")


if __name__ == "__main__":
    main()
