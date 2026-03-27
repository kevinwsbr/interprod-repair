#!/usr/bin/env python3
"""
Run prepared experiment prompts with Claude Haiku 4.5.

This script consumes a prepared manifest plus prompt directory, executes each
experiment in order, and writes resumable outputs:

- raw model text per round
- parsed JSON per round when parse succeeds
- append-only JSONL result rows with token usage and cost

It is safe to interrupt and rerun with the same --run-name. Completed rounds
are skipped automatically.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from run_rci_experiment import CRITIQUE_PROMPT, extract_json_from_response, load_dotenv_file, safe_slug


MODEL_ALIAS = "claude-haiku-4-5"
INPUT_COST_PER_MTOK = 1.0
OUTPUT_COST_PER_MTOK = 5.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0


def get_anthropic_client():
    """Return an Anthropic client, raising clearly if unavailable."""
    try:
        import anthropic
    except ImportError:
        print("✗ anthropic package not installed.", file=sys.stderr)
        sys.exit(1)

    load_dotenv_file()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ ANTHROPIC_API_KEY environment variable not set (checked shell env and .env).", file=sys.stderr)
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def parse_prompt_file(path: Path) -> Dict[str, str]:
    """Parse a generated prompt file into sections."""
    text = path.read_text(encoding="utf-8")
    markers = [
        "=== System Prompt ===\n",
        "=== Initial Prompt ===\n",
        "=== RCI Improve Prompt ===\n",
    ]
    if not all(marker in text for marker in markers):
        raise ValueError(f"Prompt file missing required sections: {path}")

    _, rest = text.split(markers[0], 1)
    system_prompt, rest = rest.split(markers[1], 1)
    initial_prompt, improve_prompt = rest.split(markers[2], 1)
    return {
        "system_prompt": system_prompt.strip(),
        "initial_prompt": initial_prompt.strip(),
        "improve_prompt": improve_prompt.strip(),
    }


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL manifest."""
    rows: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def result_row_has_artifacts(row: Dict[str, Any]) -> bool:
    """Return True only if the saved artifacts for this round are present."""
    if row.get("api_ok") is False:
        error_path = row.get("error_output_path")
        return bool(error_path) and Path(error_path).exists()

    raw_path = row.get("raw_output_path")
    api_path = row.get("api_response_path")
    parsed_path = row.get("parsed_output_path")
    parse_ok = bool(row.get("parse_ok"))

    if not raw_path or not Path(raw_path).exists():
        return False
    if not api_path or not Path(api_path).exists():
        return False
    if parse_ok and (not parsed_path or not Path(parsed_path).exists()):
        return False
    return True


def derive_prompt_dir_from_manifest(manifest_path: Path) -> Path:
    """Infer the prepared prompt directory from a manifest filename."""
    stem = manifest_path.name
    suffix = "_manifest.jsonl"
    if not stem.endswith(suffix):
        raise ValueError(
            f"Cannot infer prompt directory from manifest name: {manifest_path.name}"
        )
    prompt_dir_name = stem[: -len(suffix)]
    return manifest_path.parent.parent / "prompts" / prompt_dir_name


def load_run_state(results_path: Path) -> Tuple[set[Tuple[str, int]], set[str]]:
    """Return completed rounds and experiments with terminal API failures."""
    completed: set[Tuple[str, int]] = set()
    failed_experiments: set[str] = set()
    if not results_path.exists():
        return completed, failed_experiments

    with open(results_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if result_row_has_artifacts(row):
                    completed.add((row["experiment_id"], row["round"]))
                    if row.get("api_ok") is False:
                        failed_experiments.add(row["experiment_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return completed, failed_experiments


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSON row and flush it durably."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically enough for experiment artifacts."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_json_safe(value: Any) -> Any:
    """Convert nested values into JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return repr(value)


def build_error_artifact(
    exception: Exception,
    *,
    experiment_id: str,
    round_number: int,
    condition: str,
    model: str,
    prompt_path: Path,
    system_prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Build a structured, JSON-safe error payload for failed API calls."""
    exception_attrs: Dict[str, Any] = {}
    for attr in ("status_code", "request_id", "body", "response"):
        if hasattr(exception, attr):
            exception_attrs[attr] = make_json_safe(getattr(exception, attr))

    return {
        "experiment_id": experiment_id,
        "round": round_number,
        "condition": condition,
        "model": model,
        "prompt_file": str(prompt_path),
        "system_prompt": system_prompt,
        "messages": make_json_safe(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "exception_module": type(exception).__module__,
        "exception_attrs": exception_attrs,
        "traceback": traceback.format_exc(),
        "timestamp": int(time.time()),
    }


def summarize_results(results_path: Path) -> Dict[str, Any]:
    """Aggregate token, cost, and parse stats across the run results file."""
    summary = {
        "rows_completed": 0,
        "successful_api_rows": 0,
        "failed_api_rows": 0,
        "parsed_rows": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd_estimate": 0.0,
        "last_updated": None,
    }
    if not results_path.exists():
        return summary

    last_timestamp: Optional[int] = None
    with open(results_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            summary["rows_completed"] += 1
            if row.get("api_ok") is False:
                summary["failed_api_rows"] += 1
            else:
                summary["successful_api_rows"] += 1
            if row.get("parse_ok"):
                summary["parsed_rows"] += 1
            summary["input_tokens"] += int(row.get("input_tokens", 0) or 0)
            summary["output_tokens"] += int(row.get("output_tokens", 0) or 0)
            summary["total_tokens"] += int(row.get("total_tokens", 0) or 0)
            summary["cost_usd_estimate"] += float(row.get("cost_usd_estimate", 0.0) or 0.0)

            timestamp = row.get("timestamp")
            if isinstance(timestamp, int):
                last_timestamp = timestamp

    summary["cost_usd_estimate"] = round(summary["cost_usd_estimate"], 6)
    summary["last_updated"] = last_timestamp
    return summary


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimate Anthropic Haiku 4.5 cost in USD."""
    return round(
        (input_tokens / 1_000_000.0) * INPUT_COST_PER_MTOK
        + (output_tokens / 1_000_000.0) * OUTPUT_COST_PER_MTOK,
        6,
    )


def call_claude(
    client,
    model: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Call Claude and return text plus usage metadata."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=messages,
    )
    text = "".join(block.text for block in response.content if getattr(block, "text", None))
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    stop_reason = getattr(response, "stop_reason", None)

    raw_response: Dict[str, Any]
    if hasattr(response, "model_dump"):
        raw_response = response.model_dump()
    else:
        raw_response = {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stop_reason": stop_reason,
        }

    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason,
        "raw_response": raw_response,
    }


def build_round_messages(
    initial_prompt: str,
    improve_prompt: str,
    rci_rounds: int,
) -> List[Tuple[int, List[Dict[str, str]], bool]]:
    """
    Build the message list for each round.

    Returns tuples of:
    - round number
    - full message history for that round
    - whether this is a critique round
    """
    rounds: List[Tuple[int, List[Dict[str, str]], bool]] = []
    rounds.append((0, [{"role": "user", "content": initial_prompt}], False))

    if rci_rounds >= 1:
        rounds.append((1, [], True))
    if rci_rounds >= 2:
        rounds.append((2, [], False))
    return rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prepared prompts with Claude Haiku 4.5.")
    parser.add_argument("--manifest", required=True, help="Path to prepared manifest JSONL")
    parser.add_argument("--prompt-dir", default=None, help="Path to prepared prompt directory (optional; inferred from manifest by default)")
    parser.add_argument("--run-name", required=True, help="Stable run name for resumable outputs")
    parser.add_argument("--model", default=MODEL_ALIAS, help=f"Anthropic model alias (default: {MODEL_ALIAS})")
    parser.add_argument("--rci-rounds", type=int, default=0, choices=[0, 1, 2], help="0=single-shot, 1=+critique, 2=+improve")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max completion tokens per call")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N manifest entries")
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="Delay between API calls")
    parser.add_argument("--dry-run", action="store_true", help="Validate prompt loading without API calls")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    prompt_root = Path(args.prompt_dir) if args.prompt_dir else derive_prompt_dir_from_manifest(manifest_path)
    if not manifest_path.exists():
        print(f"✗ Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    if not prompt_root.exists():
        print(f"✗ Prompt directory not found: {prompt_root}", file=sys.stderr)
        sys.exit(1)

    run_root = Path("results/prepared_runs") / safe_slug(args.run_name)
    raw_root = run_root / "raw"
    parsed_root = run_root / "parsed"
    api_root = run_root / "api"
    error_root = run_root / "errors"
    run_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    parsed_root.mkdir(parents=True, exist_ok=True)
    api_root.mkdir(parents=True, exist_ok=True)
    error_root.mkdir(parents=True, exist_ok=True)

    results_path = run_root / "results.jsonl"
    metadata_path = run_root / "run_metadata.json"
    metadata = {
        "run_name": args.run_name,
        "manifest": str(manifest_path),
        "prompt_dir": str(prompt_root),
        "model": args.model,
        "rci_rounds": args.rci_rounds,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "failure_policy": "skip_on_rerun",
        "usage_summary": summarize_results(results_path),
    }
    write_json(metadata_path, metadata)

    rows = load_manifest(manifest_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    completed, failed_experiments = load_run_state(results_path)
    print(f"Loaded {len(rows)} manifest rows")
    print(f"Results file: {results_path}")
    print(f"Completed rounds already present: {len(completed)}")
    print(f"Experiments with saved API failures: {len(failed_experiments)}")

    client = None if args.dry_run else get_anthropic_client()

    rounds_per_experiment = 1 + (1 if args.rci_rounds >= 1 else 0) + (1 if args.rci_rounds >= 2 else 0)
    total_round_targets = len(rows) * rounds_per_experiment
    done = 0

    try:
        for row in rows:
            experiment_id = row["experiment_id"]
            condition = row["condition"]
            task = "repair"

            if experiment_id in failed_experiments:
                done += rounds_per_experiment
                print(
                    f"[{done:>4}/{total_round_targets}] {experiment_id[:50]:<50} "
                    "skipped (previous API failure)"
                )
                continue

            prompt_path: Optional[Path] = None
            if row.get("prompt_file"):
                candidate = Path(row["prompt_file"])
                if candidate.exists():
                    prompt_path = candidate

            if prompt_path is None:
                prompt_subdir = prompt_root / task / condition
                matches = list(prompt_subdir.glob(f"{safe_slug(experiment_id)}*.txt"))
                if len(matches) != 1:
                    print(f"⚠ {experiment_id}: expected exactly one prompt file, found {len(matches)}; skipping")
                    continue
                prompt_path = matches[0]

            prompts = parse_prompt_file(prompt_path)

            messages: List[Dict[str, str]] = []
            round_plan = build_round_messages(
                prompts["initial_prompt"],
                prompts["improve_prompt"],
                args.rci_rounds,
            )

            for round_index, (round_number, _unused_messages, is_critique) in enumerate(round_plan):
                done += 1
                key = (experiment_id, round_number)
                if key in completed:
                    print(f"[{done:>4}/{total_round_targets}] {experiment_id[:50]:<50} round {round_number} skipped")
                    # Reconstruct conversation state approximately for future rounds.
                    if round_number == 0:
                        raw_path = raw_root / f"{safe_slug(experiment_id)}_round0.txt"
                        if raw_path.exists():
                            messages.append({"role": "user", "content": prompts["initial_prompt"]})
                            messages.append({"role": "assistant", "content": raw_path.read_text(encoding='utf-8')})
                    elif round_number == 1:
                        raw_path = raw_root / f"{safe_slug(experiment_id)}_round1.txt"
                        if raw_path.exists():
                            messages.append({"role": "user", "content": CRITIQUE_PROMPT})
                            messages.append({"role": "assistant", "content": raw_path.read_text(encoding='utf-8')})
                    elif round_number == 2:
                        pass
                    continue

                if round_number == 0:
                    messages.append({"role": "user", "content": prompts["initial_prompt"]})
                elif round_number == 1:
                    messages.append({"role": "user", "content": CRITIQUE_PROMPT})
                elif round_number == 2:
                    messages.append({"role": "user", "content": prompts["improve_prompt"]})

                if args.dry_run:
                    print(f"[{done:>4}/{total_round_targets}] {experiment_id[:50]:<50} round {round_number} dry-run")
                    dummy_text = messages[-1]["content"][:200]
                    messages.append({"role": "assistant", "content": dummy_text})
                    continue

                safe_id = safe_slug(experiment_id)
                raw_path = raw_root / f"{safe_id}_round{round_number}.txt"
                parsed_path = parsed_root / f"{safe_id}_round{round_number}.json"
                api_path = api_root / f"{safe_id}_round{round_number}.json"
                error_path = error_root / f"{safe_id}_round{round_number}.json"

                try:
                    result = call_claude(
                        client=client,
                        model=args.model,
                        system_prompt=prompts["system_prompt"],
                        messages=messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                except Exception as exc:
                    error_payload = build_error_artifact(
                        exc,
                        experiment_id=experiment_id,
                        round_number=round_number,
                        condition=condition,
                        model=args.model,
                        prompt_path=prompt_path,
                        system_prompt=prompts["system_prompt"],
                        messages=messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    write_json(error_path, error_payload)

                    result_row = {
                        **row,
                        "task": task,
                        "round": round_number,
                        "is_critique_round": is_critique,
                        "model": args.model,
                        "api_ok": False,
                        "raw_output_path": None,
                        "parsed_output_path": None,
                        "api_response_path": None,
                        "error_output_path": str(error_path),
                        "raw_text": None,
                        "parsed_output": None,
                        "parse_ok": False,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd_estimate": 0.0,
                        "stop_reason": None,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "timestamp": error_payload["timestamp"],
                    }
                    append_jsonl(results_path, result_row)
                    completed.add(key)
                    failed_experiments.add(experiment_id)
                    metadata["usage_summary"] = summarize_results(results_path)
                    write_json(metadata_path, metadata)

                    print(
                        f"[{done:>4}/{total_round_targets}] {experiment_id[:50]:<50} "
                        f"round {round_number} API ERROR {type(exc).__name__}: {exc}"
                    )

                    remaining_rounds = len(round_plan) - round_index - 1
                    if remaining_rounds > 0:
                        done += remaining_rounds
                        print(
                            f"[{done:>4}/{total_round_targets}] {experiment_id[:50]:<50} "
                            "remaining rounds skipped after API failure"
                        )
                    break

                text = result["text"]
                parsed = extract_json_from_response(text)

                raw_path.write_text(text, encoding="utf-8")
                if parsed is not None:
                    write_json(parsed_path, parsed)
                write_json(api_path, result["raw_response"])

                result_row = {
                    **row,
                    "task": task,
                    "round": round_number,
                    "is_critique_round": is_critique,
                    "model": args.model,
                    "api_ok": True,
                    "raw_output_path": str(raw_path),
                    "parsed_output_path": str(parsed_path) if parsed is not None else None,
                    "api_response_path": str(api_path),
                    "error_output_path": None,
                    "raw_text": text,
                    "parsed_output": parsed,
                    "parse_ok": parsed is not None,
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["input_tokens"] + result["output_tokens"],
                    "cost_usd_estimate": estimate_cost_usd(result["input_tokens"], result["output_tokens"]),
                    "stop_reason": result["stop_reason"],
                    "timestamp": int(time.time()),
                }
                append_jsonl(results_path, result_row)
                completed.add(key)
                metadata["usage_summary"] = summarize_results(results_path)
                write_json(metadata_path, metadata)

                print(
                    f"[{done:>4}/{total_round_targets}] {experiment_id[:50]:<50} "
                    f"round {round_number} tokens={result_row['total_tokens']} cost=${result_row['cost_usd_estimate']:.6f}"
                )

                messages.append({"role": "assistant", "content": text})
                time.sleep(args.sleep_seconds)

    except KeyboardInterrupt:
        print("\nInterrupted. Safe to rerun with the same --run-name to continue.")
        sys.exit(130)

    metadata["usage_summary"] = summarize_results(results_path)
    write_json(metadata_path, metadata)
    print("\nDone.")
    print(f"Run directory: {run_root}")
    print(f"Results JSONL: {results_path}")


if __name__ == "__main__":
    main()
