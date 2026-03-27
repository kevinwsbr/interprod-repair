#!/usr/bin/env python3
"""Detect directional cross-file impact for high-confidence call relationships.

This script answers the directional question: whether code in file A is
impacted by changes in file B. The heuristic is:

1. File A calls one or more methods defined in file B.
2. File B's diff changes those called methods.
3. Optionally, file A's diff also changes callsites or references to those
   methods, which is treated as stronger evidence.
"""

import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple


HIGH_CONF_FILE = "call_relationships_high_confidence.json"
INTERPROC_FILE = "interprocedural_vulnerabilities.json"
OUT_FILE = "directional_call_impact_high_confidence.json"

NOISE_METHODS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
}

METHOD_PATTERNS = {
    "python": r"def\s+(\w+)\s*\(",
    "java": r"(?:public|private|protected|static|\s)+\w+\s+(\w+)\s*\(",
    "javascript": r"(?:function\s+(\w+)|(\w+)\s*:\s*function|\bconst\s+(\w+)\s*=\s*(?:async\s*)?\()",
    "typescript": r"(?:function\s+(\w+)|(\w+)\s*:\s*function|\bconst\s+(\w+)\s*=\s*(?:async\s*)?\()",
    "c": r"\w+\s+(\w+)\s*\([^)]*\)\s*\{",
    "c++": r"\w+\s+(\w+)\s*\([^)]*\)\s*\{",
    "ruby": r"def\s+(\w+)",
    "go": r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
    "php": r"function\s+(\w+)\s*\(",
}


def pair_key(cve_id: str, file1: str, file2: str) -> Tuple[str, str, str]:
    """Build normalized key for a CVE and unordered file pair."""
    left, right = sorted([file1 or "", file2 or ""])
    return (cve_id, left, right)


def changed_lines_from_diff(diff_text: str) -> List[str]:
    """Extract changed lines from a unified diff."""
    if not diff_text:
        return []

    lines: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            content = line[1:].strip()
            if content:
                lines.append(content)
    return lines


def token_in_lines(token: str, lines: List[str]) -> bool:
    """Check whether a token appears in any changed line."""
    if not token or token in NOISE_METHODS:
        return False
    pattern = re.compile(rf"\b{re.escape(token)}\b")
    return any(pattern.search(line) for line in lines)


def extract_method_names(code: str, language: str) -> Set[str]:
    """Extract method names from source code using simple language heuristics."""
    if not code or not language:
        return set()

    pattern = METHOD_PATTERNS.get(language.lower())
    if not pattern:
        return set()

    methods: Set[str] = set()
    for match in re.findall(pattern, code, re.MULTILINE):
        if isinstance(match, tuple):
            methods.update(item for item in match if item and item not in NOISE_METHODS)
        elif match not in NOISE_METHODS:
            methods.add(match)
    return methods


def find_called_methods(caller_code: str, callee_methods: Set[str]) -> Set[str]:
    """Find which callee methods appear to be called in the caller code."""
    if not caller_code or not callee_methods:
        return set()

    called: Set[str] = set()
    for method in callee_methods:
        patterns = [
            rf"\b{re.escape(method)}\s*\(",
            rf"\.\s*{re.escape(method)}\s*\(",
            rf"->\s*{re.escape(method)}\s*\(",
            rf"::\s*{re.escape(method)}\s*\(",
        ]
        if any(re.search(pattern, caller_code) for pattern in patterns):
            called.add(method)
    return called


def build_interproc_index(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    """Index interprocedural records by CVE and unordered file pair."""
    index: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        files = record.get("files") or []
        if len(files) != 2:
            continue
        key = pair_key(
            record.get("cve_id", ""),
            files[0].get("filename", ""),
            files[1].get("filename", ""),
        )
        index[key].append(record)
    return index


def choose_best_match(candidates: List[Dict[str, Any]], file1: str, file2: str) -> Dict[str, Any]:
    """Choose a candidate preserving file order when possible."""
    if not candidates:
        return {}
    for candidate in candidates:
        files = candidate.get("files") or []
        if len(files) != 2:
            continue
        if files[0].get("filename") == file1 and files[1].get("filename") == file2:
            return candidate
    return candidates[0]


def classify_direction(
    caller_code: str,
    caller_diff: List[str],
    caller_language: str,
    callee_code: str,
    callee_diff: List[str],
    callee_language: str,
) -> Dict[str, Any]:
    """Classify whether caller is impacted by changes in callee."""
    callee_methods = extract_method_names(callee_code, callee_language)
    called_methods = sorted(find_called_methods(caller_code, callee_methods))
    changed_callee_methods = sorted([m for m in called_methods if token_in_lines(m, callee_diff)])
    changed_caller_callsites = sorted([m for m in called_methods if token_in_lines(m, caller_diff)])

    if changed_callee_methods and changed_caller_callsites:
        impact = "strong"
    elif changed_callee_methods:
        impact = "callee_changed"
    elif changed_caller_callsites:
        impact = "caller_changed_only"
    elif called_methods:
        impact = "call_relation_no_diff_evidence"
    else:
        impact = "no_directional_call_relation"

    return {
        "impact": impact,
        "called_methods": called_methods,
        "called_methods_changed_in_callee": changed_callee_methods,
        "called_methods_changed_in_caller": changed_caller_callsites,
    }


def analyze() -> Dict[str, Any]:
    """Run directional analysis for all high-confidence pairs."""
    with open(HIGH_CONF_FILE, "r") as file_handle:
        high_conf = json.load(file_handle)

    with open(INTERPROC_FILE, "r") as file_handle:
        interproc = json.load(file_handle)

    interproc_index = build_interproc_index(interproc)

    results: List[Dict[str, Any]] = []
    summary = Counter()
    directional_summary = Counter()

    for relation in high_conf:
        cve_id = relation.get("cve_id", "")
        file1 = relation.get("file1_path", "")
        file2 = relation.get("file2_path", "")
        key = pair_key(cve_id, file1, file2)
        matched = choose_best_match(interproc_index.get(key, []), file1, file2)

        if not matched:
            summary["missing_interprocedural_record"] += 1
            continue

        files = matched.get("files") or []
        if len(files) != 2:
            summary["invalid_interprocedural_record"] += 1
            continue

        file_map = {item.get("filename", ""): item for item in files}
        file1_record = file_map.get(file1, files[0])
        file2_record = file_map.get(file2, files[1])

        file1_code = file1_record.get("code_before") or ""
        file2_code = file2_record.get("code_before") or ""
        file1_diff = changed_lines_from_diff(file1_record.get("diff") or "")
        file2_diff = changed_lines_from_diff(file2_record.get("diff") or "")

        file1_by_file2 = classify_direction(
            caller_code=file1_code,
            caller_diff=file1_diff,
            caller_language=relation.get("file1_language") or "",
            callee_code=file2_code,
            callee_diff=file2_diff,
            callee_language=relation.get("file2_language") or "",
        )
        file2_by_file1 = classify_direction(
            caller_code=file2_code,
            caller_diff=file2_diff,
            caller_language=relation.get("file2_language") or "",
            callee_code=file1_code,
            callee_diff=file1_diff,
            callee_language=relation.get("file1_language") or "",
        )

        directional_summary[f"file1_by_file2:{file1_by_file2['impact']}"] += 1
        directional_summary[f"file2_by_file1:{file2_by_file1['impact']}"] += 1
        summary["total_pairs"] += 1

        results.append(
            {
                "cve_id": cve_id,
                "file1_path": file1,
                "file2_path": file2,
                "file1_language": relation.get("file1_language"),
                "file2_language": relation.get("file2_language"),
                "file1_impacted_by_file2": file1_by_file2,
                "file2_impacted_by_file1": file2_by_file1,
            }
        )

    payload = {
        "summary": dict(summary),
        "directional_summary": dict(directional_summary),
        "results": results,
    }

    with open(OUT_FILE, "w") as file_handle:
        json.dump(payload, file_handle, indent=2)

    return payload


def main() -> None:
    """Execute directional impact analysis and print a compact summary."""
    payload = analyze()
    directional = payload["directional_summary"]

    print("\n=== Directional Call Impact Summary ===")
    print(f"High-confidence pairs analyzed: {payload['summary'].get('total_pairs', 0)}")
    print(
        "File1 impacted by File2:",
        "strong=", directional.get("file1_by_file2:strong", 0),
        "callee_changed=", directional.get("file1_by_file2:callee_changed", 0),
        "caller_changed_only=", directional.get("file1_by_file2:caller_changed_only", 0),
        "call_relation_no_diff_evidence=", directional.get("file1_by_file2:call_relation_no_diff_evidence", 0),
        "no_relation=", directional.get("file1_by_file2:no_directional_call_relation", 0),
    )
    print(
        "File2 impacted by File1:",
        "strong=", directional.get("file2_by_file1:strong", 0),
        "callee_changed=", directional.get("file2_by_file1:callee_changed", 0),
        "caller_changed_only=", directional.get("file2_by_file1:caller_changed_only", 0),
        "call_relation_no_diff_evidence=", directional.get("file2_by_file1:call_relation_no_diff_evidence", 0),
        "no_relation=", directional.get("file2_by_file1:no_directional_call_relation", 0),
    )
    print(f"Saved detailed report to {OUT_FILE}")


if __name__ == "__main__":
    main()