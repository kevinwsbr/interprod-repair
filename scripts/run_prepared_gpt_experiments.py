#!/usr/bin/env python3
"""
Run prepared experiment prompts with OpenAI GPT models.

This script mirrors the prepared Anthropic runner, but uses the OpenAI API
so the same manifest/prompt workflow can be reused for GPT-family models.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from run_rci_experiment import (
    CRITIQUE_PROMPT,
    extract_json_from_response,
    load_dotenv_file,
    safe_slug,
)
from run_prepared_haiku45_experiments import (
    append_jsonl,
    build_error_artifact,
    build_round_messages,
    derive_prompt_dir_from_manifest,
    load_manifest,
    load_run_state,
    parse_prompt_file,
    summarize_results,
    write_json,
)


MODEL_ALIAS = "gpt-5-mini"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0


def get_openai_client():
    """Return an OpenAI client, raising clearly if unavailable."""
    try:
        import openai
    except ImportError:
        print("✗ openai package not installed.", file=sys.stderr)
        sys.exit(1)

    load_dotenv_file()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "✗ OPENAI_API_KEY environment variable not set (checked shell env and .env).",
            file=sys.stderr,
        )
        sys.exit(1)
    return openai.OpenAI(api_key=api_key)


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_mtok: float,
    output_cost_per_mtok: float,
) -> float:
    """Estimate OpenAI usage cost in USD."""
    return round(
        (input_tokens / 1_000_000.0) * input_cost_per_mtok
        + (output_tokens / 1_000_000.0) * output_cost_per_mtok,
        6,
    )


def extract_openai_text(response) -> str:
    """Best-effort extraction of assistant text from a chat completion."""
    if not getattr(response, "choices", None):
        return ""

    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content or "")


def call_openai(
    client,
    model: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Call OpenAI and return text plus usage metadata."""
    request_messages = [{"role": "system", "content": system_prompt}, *messages]
    base_payload = {
        "model": model,
        "messages": request_messages,
    }

    last_exc: Optional[Exception] = None
    response = None
    request_variants = [
        {"max_completion_tokens": max_tokens, "temperature": temperature},
        {"max_completion_tokens": max_tokens},
        {"max_tokens": max_tokens, "temperature": temperature},
        {"max_tokens": max_tokens},
    ]
    for extra_payload in request_variants:
        try:
            response = client.chat.completions.create(
                **base_payload,
                **extra_payload,
            )
            break
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            last_exc = exc
            message = str(exc).lower()
            unsupported_temperature = (
                "temperature" in extra_payload and "temperature" in message
            )
            unsupported_token_arg = (
                ("max_completion_tokens" in extra_payload and "max_completion_tokens" in message)
                or ("max_tokens" in extra_payload and "max_tokens" in message)
            )
            generic_parameter_error = (
                "unsupported parameter" in message
                or "unsupported value" in message
                or "unknown parameter" in message
                or "unexpected argument" in message
            )
            if unsupported_temperature or unsupported_token_arg or generic_parameter_error:
                continue
            raise

    if response is None:
        assert last_exc is not None
        raise last_exc

    text = extract_openai_text(response)
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    stop_reason = None
    if getattr(response, "choices", None):
        stop_reason = getattr(response.choices[0], "finish_reason", None)

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


def resolve_prompt_path(row: Dict[str, Any], prompt_root: Path, task: str) -> Optional[Path]:
    """Resolve the exact prepared prompt file for a manifest row."""
    prompt_file = row.get("prompt_file")
    if prompt_file:
        candidate = Path(prompt_file)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.exists():
            return candidate

    condition = row["condition"]
    experiment_id = row["experiment_id"]
    prompt_subdir = prompt_root / task / condition
    safe_id = safe_slug(experiment_id)
    exact_name = re.compile(rf"^{re.escape(safe_id)}_[0-9a-f]{{10}}\.txt$")

    matches = [
        path
        for path in prompt_subdir.glob("*.txt")
        if exact_name.match(path.name)
    ]
    if len(matches) == 1:
        return matches[0]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prepared prompts with OpenAI GPT models.")
    parser.add_argument("--manifest", required=True, help="Path to prepared manifest JSONL")
    parser.add_argument(
        "--prompt-dir",
        default=None,
        help="Path to prepared prompt directory (optional; inferred from manifest by default)",
    )
    parser.add_argument("--run-name", required=True, help="Stable run name for resumable outputs")
    parser.add_argument(
        "--model",
        default=MODEL_ALIAS,
        help=f"OpenAI model alias (default: {MODEL_ALIAS})",
    )
    parser.add_argument(
        "--rci-rounds",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0=single-shot, 1=+critique, 2=+improve",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max completion tokens per call",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N manifest entries")
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="Delay between API calls")
    parser.add_argument(
        "--input-cost-per-mtok",
        type=float,
        default=0.0,
        help="Estimated USD cost per 1M input tokens (defaults to 0.0 if unknown)",
    )
    parser.add_argument(
        "--output-cost-per-mtok",
        type=float,
        default=0.0,
        help="Estimated USD cost per 1M output tokens (defaults to 0.0 if unknown)",
    )
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

    if args.input_cost_per_mtok == 0.0 and args.output_cost_per_mtok == 0.0:
        print("⚠ Cost estimation disabled (token counts still recorded).")

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
        "provider": "openai",
        "model": args.model,
        "rci_rounds": args.rci_rounds,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "input_cost_per_mtok": args.input_cost_per_mtok,
        "output_cost_per_mtok": args.output_cost_per_mtok,
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

    client = None if args.dry_run else get_openai_client()

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

            prompt_path = resolve_prompt_path(row, prompt_root, task)
            if prompt_path is None:
                prompt_subdir = prompt_root / task / condition
                safe_id = safe_slug(experiment_id)
                broad_matches = [
                    path.name for path in prompt_subdir.glob(f"{safe_id}*.txt")
                ]
                print(
                    f"⚠ {experiment_id}: could not resolve unique prompt file; "
                    f"candidate matches={len(broad_matches)}; skipping"
                )
                continue

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
                    print(
                        f"[{done:>4}/{total_round_targets}] {experiment_id[:50]:<50} round {round_number} skipped"
                    )
                    if round_number == 0:
                        raw_path = raw_root / f"{safe_slug(experiment_id)}_round0.txt"
                        if raw_path.exists():
                            messages.append({"role": "user", "content": prompts["initial_prompt"]})
                            messages.append({"role": "assistant", "content": raw_path.read_text(encoding="utf-8")})
                    elif round_number == 1:
                        raw_path = raw_root / f"{safe_slug(experiment_id)}_round1.txt"
                        if raw_path.exists():
                            messages.append({"role": "user", "content": CRITIQUE_PROMPT})
                            messages.append({"role": "assistant", "content": raw_path.read_text(encoding="utf-8")})
                    continue

                if round_number == 0:
                    messages.append({"role": "user", "content": prompts["initial_prompt"]})
                elif round_number == 1:
                    messages.append({"role": "user", "content": CRITIQUE_PROMPT})
                elif round_number == 2:
                    messages.append({"role": "user", "content": prompts["improve_prompt"]})

                if args.dry_run:
                    print(
                        f"[{done:>4}/{total_round_targets}] {experiment_id[:50]:<50} round {round_number} dry-run"
                    )
                    dummy_text = messages[-1]["content"][:200]
                    messages.append({"role": "assistant", "content": dummy_text})
                    continue

                safe_id = safe_slug(experiment_id)
                raw_path = raw_root / f"{safe_id}_round{round_number}.txt"
                parsed_path = parsed_root / f"{safe_id}_round{round_number}.json"
                api_path = api_root / f"{safe_id}_round{round_number}.json"
                error_path = error_root / f"{safe_id}_round{round_number}.json"

                try:
                    result = call_openai(
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
                        "provider": "openai",
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
                    "provider": "openai",
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
                    "cost_usd_estimate": estimate_cost_usd(
                        result["input_tokens"],
                        result["output_tokens"],
                        args.input_cost_per_mtok,
                        args.output_cost_per_mtok,
                    ),
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
