#!/usr/bin/env python3
"""
Filter two-file vulnerabilities using Tree-sitter call analysis across languages.

This script reads interprocedural_vulnerabilities.json and, for language pairs
with installed Tree-sitter grammars, extracts function definitions and call
sites from each file's code_before content.

A record is considered Tree-sitter interprocedural when at least one file calls
functions defined in the other file.
"""

import importlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tree_sitter import Language, Node, Parser

INPUT_FILE = "interprocedural_vulnerabilities.json"
OUTPUT_FILE = "treesitter_interprocedural_filter.json"

IMPACT_WITH_DIFF_EVIDENCE = {"strong", "callee_changed", "caller_changed_only"}

STRICT_IMPORT_LANGS = {"c", "c++", "cpp", "php"}

LANGUAGE_NOISE_CALLS: Dict[str, Set[str]] = {
    "c": {
        "printf",
        "fprintf",
        "sprintf",
        "snprintf",
        "memcpy",
        "memset",
        "memcmp",
        "strlen",
        "strcpy",
        "strncpy",
        "strcmp",
        "malloc",
        "calloc",
        "realloc",
        "free",
        "exit",
    },
    "c++": {
        "printf",
        "fprintf",
        "snprintf",
        "memcpy",
        "memset",
        "strlen",
        "malloc",
        "free",
        "new",
        "delete",
        "size",
        "begin",
        "end",
        "push_back",
    },
    "php": {
        "count",
        "isset",
        "empty",
        "array_merge",
        "in_array",
        "explode",
        "implode",
        "json_encode",
        "json_decode",
        "trim",
        "strtolower",
        "strtoupper",
        "__construct",
        "__destruct",
        "__call",
        "__invoke",
        "__get",
        "__set",
        "__isset",
        "__unset",
        "__sleep",
        "__wakeup",
        "__toString",
        "__debugInfo",
    },
}

DEFINITION_NODE_TYPES = {
    "function_definition",
    "method_definition",
    "function_declaration",
    "method_declaration",
    "function_item",
}

CALL_NODE_TYPES = {
    "call",
    "call_expression",
    "invocation_expression",
}

IMPORT_NODE_TYPES = {
    "import_statement",
    "import_declaration",
    "import_from_statement",
    "include_expression",
    "require_expression",
    "require_once_expression",
    "include_once_expression",
    "preproc_include",
    "using_declaration",
}

LANG_MODULE_CANDIDATES: Dict[str, List[Tuple[str, str]]] = {
    "python": [("tree_sitter_python", "language")],
    "javascript": [("tree_sitter_javascript", "language")],
    "jsx": [("tree_sitter_javascript", "language")],
    "typescript": [("tree_sitter_typescript", "language_typescript")],
    "typescriptreact": [("tree_sitter_typescript", "language_tsx")],
    "tsx": [("tree_sitter_typescript", "language_tsx")],
    "java": [("tree_sitter_java", "language")],
    "c": [("tree_sitter_c", "language")],
    "c++": [("tree_sitter_cpp", "language")],
    "cpp": [("tree_sitter_cpp", "language")],
    "go": [("tree_sitter_go", "language")],
    "php": [("tree_sitter_php", "language_php"), ("tree_sitter_php", "language")],
    "ruby": [("tree_sitter_ruby", "language")],
    "rust": [("tree_sitter_rust", "language")],
    "c#": [("tree_sitter_c_sharp", "language")],
    "csharp": [("tree_sitter_c_sharp", "language")],
    "objective-c": [("tree_sitter_c", "language")],
    "bash": [("tree_sitter_bash", "language")],
    "shell": [("tree_sitter_bash", "language")],
    "css": [("tree_sitter_css", "language")],
    "html": [("tree_sitter_html", "language")],
    "html+erb": [("tree_sitter_html", "language")],
    "json": [("tree_sitter_json", "language")],
    "lua": [("tree_sitter_lua", "language")],
    "markdown": [("tree_sitter_markdown", "language")],
    "sql": [("tree_sitter_sql", "language")],
    "toml": [("tree_sitter_toml", "language")],
    "yaml": [("tree_sitter_yaml", "language")],
    "kotlin": [("tree_sitter_kotlin", "language")],
    "scala": [("tree_sitter_scala", "language")],
    "swift": [("tree_sitter_swift", "language")],
    "elixir": [("tree_sitter_elixir", "language")],
    "haskell": [("tree_sitter_haskell", "language")],
}


class ParserRegistry:
    """Lazy loader for Tree-sitter parsers by language label."""

    def __init__(self) -> None:
        self.parsers: Dict[str, Parser] = {}
        self.unavailable: Set[str] = set()

    def get_parser(self, raw_language: str) -> Optional[Parser]:
        """Return parser for normalized language, or None if unavailable."""
        lang = (raw_language or "").strip().lower()
        if not lang:
            self.unavailable.add("unknown")
            return None

        if lang in self.parsers:
            return self.parsers[lang]

        candidates = LANG_MODULE_CANDIDATES.get(lang, [])
        for module_name, func_name in candidates:
            try:
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                language = Language(func())
                parser = Parser(language)
                self.parsers[lang] = parser
                return parser
            except (ImportError, AttributeError, TypeError, ValueError):
                continue

        self.unavailable.add(lang)
        return None


def node_text(node: Node, source: bytes) -> str:
    """Extract UTF-8 text for a node from source bytes."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def best_name_from_node(node: Node, source: bytes) -> Optional[str]:
    """Best-effort extraction of a symbol name from a definition/call node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        text = node_text(name_node, source).strip()
        if text:
            return text.split(".")[-1]

    # fallback: use first identifier-like child
    queue = [node]
    while queue:
        current = queue.pop(0)
        if current.type in {"identifier", "property_identifier", "type_identifier"}:
            text = node_text(current, source).strip()
            if text:
                return text.split(".")[-1]
        queue.extend(list(current.children))

    return None


def extract_symbols(parser: Parser, source_code: str) -> Dict[str, Any]:
    """Extract function definitions, calls, and imports from source via Tree-sitter."""
    if not source_code:
        return {
            "function_names": set(),
            "called_names": set(),
            "imports": set(),
            "parse_error": "empty_source",
        }

    source = source_code.encode("utf-8", errors="ignore")
    tree = parser.parse(source)

    function_names: Set[str] = set()
    called_names: Set[str] = set()
    imports: Set[str] = set()

    queue = [tree.root_node]
    while queue:
        node = queue.pop(0)

        if node.type in DEFINITION_NODE_TYPES:
            name = best_name_from_node(node, source)
            if name:
                function_names.add(name)

        if node.type in CALL_NODE_TYPES:
            func_child = node.child_by_field_name("function")
            if func_child is None:
                func_child = node
            name = best_name_from_node(func_child, source)
            if name:
                called_names.add(name)

        if node.type in IMPORT_NODE_TYPES:
            text = node_text(node, source)
            if text:
                imports.add(text.strip())

        queue.extend(list(node.children))

    parse_error = None
    if tree.root_node.has_error:
        parse_error = "tree_has_error_nodes"

    return {
        "function_names": function_names,
        "called_names": called_names,
        "imports": imports,
        "parse_error": parse_error,
    }


def changed_lines_from_diff(diff_text: str) -> List[str]:
    """Extract changed lines from unified diff text."""
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
    """Check if token appears as a whole word in any changed line."""
    if not token:
        return False
    pattern = re.compile(rf"\b{re.escape(token)}\b")
    return any(pattern.search(line) for line in lines)


def module_stem(filepath: str) -> str:
    """Extract file stem for simple import name matching."""
    return Path(filepath or "").stem


def normalize_language(language: str) -> str:
    """Normalize language labels to canonical keys used in heuristics."""
    value = (language or "").strip().lower()
    if value == "cpp":
        return "c++"
    if value == "csharp":
        return "c#"
    return value


def refine_called_methods(
    called_methods: List[str],
    caller_language: str,
    callee_language: str,
    import_relation: bool,
) -> Tuple[List[str], List[str]]:
    """
    Reduce likely false positives using language-aware heuristics.

    For C/C++/PHP, we aggressively remove common library/magic calls and apply
    stricter filtering when there is no import/include relationship.
    """
    caller_lang = normalize_language(caller_language)
    callee_lang = normalize_language(callee_language)

    base_noise = LANGUAGE_NOISE_CALLS.get(caller_lang, set()) | LANGUAGE_NOISE_CALLS.get(
        callee_lang, set()
    )

    filtered: List[str] = []
    dropped: List[str] = []

    for method in called_methods:
        # Keep identifier-like names only.
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", method):
            dropped.append(method)
            continue

        if method in base_noise:
            dropped.append(method)
            continue

        # Without import/include evidence in strict languages, keep only names
        # that are less likely to be generic incidental matches.
        if (caller_lang in STRICT_IMPORT_LANGS or callee_lang in STRICT_IMPORT_LANGS) and not import_relation:
            if len(method) < 4:
                dropped.append(method)
                continue

        filtered.append(method)

    return sorted(set(filtered)), sorted(set(dropped))


def classify_direction(
    caller_symbols: Dict[str, Any],
    caller_language: str,
    caller_diff: List[str],
    callee_symbols: Dict[str, Any],
    callee_language: str,
    callee_diff: List[str],
    callee_path: str,
) -> Dict[str, Any]:
    """Classify directional impact caller <- callee."""
    raw_called_methods = sorted(caller_symbols["called_names"] & callee_symbols["function_names"])

    callee_hint = module_stem(callee_path)
    import_relation = any(callee_hint in imp for imp in caller_symbols["imports"]) if callee_hint else False

    called_methods, dropped_methods = refine_called_methods(
        called_methods=raw_called_methods,
        caller_language=caller_language,
        callee_language=callee_language,
        import_relation=import_relation,
    )

    changed_callee_methods = sorted(m for m in called_methods if token_in_lines(m, callee_diff))
    changed_caller_callsites = sorted(m for m in called_methods if token_in_lines(m, caller_diff))

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
        "import_relation": import_relation,
        "raw_called_methods": raw_called_methods,
        "dropped_called_methods": dropped_methods,
        "called_methods": called_methods,
        "called_methods_changed_in_callee": changed_callee_methods,
        "called_methods_changed_in_caller": changed_caller_callsites,
    }


def analyze_record(record: Dict[str, Any], registry: ParserRegistry) -> Dict[str, Any]:
    """Analyze one two-file record with Tree-sitter where parsers are available."""
    files = record.get("files") or []
    file1 = files[0]
    file2 = files[1]

    file1_lang = file1.get("language") or "Unknown"
    file2_lang = file2.get("language") or "Unknown"

    parser1 = registry.get_parser(file1_lang)
    parser2 = registry.get_parser(file2_lang)

    file1_symbols = {
        "function_names": set(),
        "called_names": set(),
        "imports": set(),
        "parse_error": "parser_unavailable",
    }
    file2_symbols = {
        "function_names": set(),
        "called_names": set(),
        "imports": set(),
        "parse_error": "parser_unavailable",
    }

    parser_available = parser1 is not None and parser2 is not None
    if parser1 is not None:
        file1_symbols = extract_symbols(parser1, file1.get("code_before") or "")
    if parser2 is not None:
        file2_symbols = extract_symbols(parser2, file2.get("code_before") or "")

    file1_diff = changed_lines_from_diff(file1.get("diff") or "")
    file2_diff = changed_lines_from_diff(file2.get("diff") or "")

    file1_by_file2 = classify_direction(
        caller_symbols=file1_symbols,
        caller_language=file1_lang,
        caller_diff=file1_diff,
        callee_symbols=file2_symbols,
        callee_language=file2_lang,
        callee_diff=file2_diff,
        callee_path=file2.get("filename") or "",
    )
    file2_by_file1 = classify_direction(
        caller_symbols=file2_symbols,
        caller_language=file2_lang,
        caller_diff=file2_diff,
        callee_symbols=file1_symbols,
        callee_language=file1_lang,
        callee_diff=file1_diff,
        callee_path=file1.get("filename") or "",
    )

    has_call_relation = bool(file1_by_file2["called_methods"] or file2_by_file1["called_methods"])
    has_diff_evidence = any(
        impact in IMPACT_WITH_DIFF_EVIDENCE
        for impact in (file1_by_file2["impact"], file2_by_file1["impact"])
    )
    is_bidirectional = bool(file1_by_file2["called_methods"] and file2_by_file1["called_methods"])

    return {
        "cve_id": record.get("cve_id"),
        "cwe_id": record.get("cwe_id"),
        "cwe_name": record.get("cwe_name"),
        "severity": record.get("severity"),
        "repository": record.get("repository"),
        "file1_path": file1.get("filename"),
        "file2_path": file2.get("filename"),
        "file1_language": file1_lang,
        "file2_language": file2_lang,
        "parsers_available": parser_available,
        "file1_parse_error": file1_symbols["parse_error"],
        "file2_parse_error": file2_symbols["parse_error"],
        "file1_function_count": len(file1_symbols["function_names"]),
        "file2_function_count": len(file2_symbols["function_names"]),
        "file1_call_count": len(file1_symbols["called_names"]),
        "file2_call_count": len(file2_symbols["called_names"]),
        "file1_by_file2": file1_by_file2,
        "file2_by_file1": file2_by_file1,
        "has_treesitter_call_relation": has_call_relation,
        "has_diff_evidence": has_diff_evidence,
        "is_bidirectional_call_relation": is_bidirectional,
    }


def run_filter(input_file: str = INPUT_FILE, output_file: str = OUTPUT_FILE) -> Dict[str, Any]:
    """Run full Tree-sitter filtering and store JSON results."""
    with open(input_file, "r") as file_handle:
        records = json.load(file_handle)

    two_file_records = [record for record in records if len(record.get("files") or []) == 2]

    registry = ParserRegistry()
    results = [analyze_record(record, registry) for record in two_file_records]

    by_language_pair = Counter()
    parser_coverage = Counter()
    directional_summary = Counter()

    for row in results:
        left = (row.get("file1_language") or "Unknown").lower()
        right = (row.get("file2_language") or "Unknown").lower()
        lang_pair = " + ".join(sorted([left, right]))
        by_language_pair[lang_pair] += 1

        if row["parsers_available"]:
            parser_coverage["both_parsers_available"] += 1
        else:
            parser_coverage["missing_parser_for_one_or_both_files"] += 1

        directional_summary[f"file1_by_file2:{row['file1_by_file2']['impact']}"] += 1
        directional_summary[f"file2_by_file1:{row['file2_by_file1']['impact']}"] += 1

    filtered_call_relation = [r for r in results if r["has_treesitter_call_relation"]]
    filtered_diff_evidence = [r for r in results if r["has_diff_evidence"]]
    filtered_bidirectional = [r for r in results if r["is_bidirectional_call_relation"]]

    payload = {
        "summary": {
            "input_records": len(records),
            "two_file_records": len(two_file_records),
            "treesitter_call_relation_records": len(filtered_call_relation),
            "treesitter_diff_evidence_records": len(filtered_diff_evidence),
            "treesitter_bidirectional_records": len(filtered_bidirectional),
        },
        "parser_coverage": dict(parser_coverage),
        "unavailable_languages": sorted(registry.unavailable),
        "by_language_pair": dict(by_language_pair),
        "directional_summary": dict(directional_summary),
        "results": results,
        "filtered": {
            "call_relation": filtered_call_relation,
            "diff_evidence": filtered_diff_evidence,
            "bidirectional": filtered_bidirectional,
        },
    }

    with open(output_file, "w") as file_handle:
        json.dump(payload, file_handle, indent=2)

    return payload


def main() -> None:
    """Execute Tree-sitter filter and print compact summary."""
    payload = run_filter()
    summary = payload["summary"]

    print("\nTree-sitter interprocedural filter (multi-language)")
    print("=" * 58)
    print(f"Input records:                     {summary['input_records']}")
    print(f"Two-file records:                  {summary['two_file_records']}")
    print(f"Tree-sitter call relation records: {summary['treesitter_call_relation_records']}")
    print(f"Tree-sitter diff evidence records: {summary['treesitter_diff_evidence_records']}")
    print(f"Tree-sitter bidirectional records: {summary['treesitter_bidirectional_records']}")
    print(f"Parser coverage:                   {payload['parser_coverage']}")
    print(f"Unavailable languages:             {payload['unavailable_languages']}")
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
