# Interprocedural Vulnerability Research Pipeline

This repository contains a research workflow for analyzing interprocedural (cross-file) vulnerability relationships in CVE repair data:

- Extracts vulnerable and fixed code records from VulnFixes dataset.
- Builds two-file interprocedural CVE examples.
- Detects call relationships with Tree-sitter and Python AST analysis.
- Scores call/diff impact and directional dependencies.
- Produces stratified JSON/CSV artifacts for evaluation.
- Runs evaluation experiments (Semgrep, RCI, and repair experiment runners).

## Main scripts

- `scripts/extract_interprocedural_vulnerabilities.py` - build interprocedural CVE records
- `scripts/filter_interprocedural_with_treesitter.py` - detect cross-language call evidence
- `scripts/analyze_call_relationships.py` - call linkage and confidence stratification
- `scripts/analyze_call_diff_impact.py` - diff-aware impact checks
- `scripts/analyze_directional_call_impact.py` - caller/callee directional impact
- `scripts/run_prepared_gemini_experiments.py` - run prepared repair experiments with Gemini
- `scripts/run_prepared_gpt_experiments.py` - run prepared repair experiments with GPT models
- `scripts/run_prepared_haiku45_experiments.py` - run prepared repair experiments with Haiku 4.5

## Key outputs

Generated artifacts are organized under `results/` by run folder, for example:

- `results/gemini3flashpreview_bidiff_repair_r0/`
- `results/gemini3flashpreview_bidiff_repair_r2/`
- `results/gpt5mini_bidiff_repair_r0/`
- `results/gpt5mini_bidiff_repair_r2/`
- `results/haiku45_bidiff_repair_r0/`
- `results/haiku45_bidiff_repair_r2/`

Each run folder contains:

- `results.jsonl` - one record per processed example/round
- `run_metadata.json` - run config and usage summary
- `api/` - saved API responses per example
- `raw/` - raw model text outputs
- `parsed/` - parsed JSON outputs
- `errors/` - error artifacts when failures occur

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline stages you need (for example):

```bash
python scripts/extract_interprocedural_vulnerabilities.py
python scripts/filter_interprocedural_with_treesitter.py
python scripts/analyze_call_relationships.py
python scripts/analyze_call_diff_impact.py
```
