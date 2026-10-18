#!/usr/bin/env python3
"""Run the Step 47A GPT-OSS smoke pilot through Cloudflare Workers AI.

This is a provider-separated replay of the existing Groq pilot. It uses the
same prompts, pool items, conditions, repetitions, progress display, retry
behavior, and raw JSONL schema, but writes a distinct output file and record
IDs so the two providers cannot be mixed accidentally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from run_gpt_oss_pilot import (
        DEFAULT_CONFIG,
        DEFAULT_CONDITIONS,
        DEFAULT_POOL,
        DEFAULT_PREDICTION,
        DEFAULT_TEST_EVIDENCE,
        DEFAULT_TRAIN_EVIDENCE,
        GroqChatClient,
        load_repo_env,
        parse_mapping,
        run_pilot,
    )
except ImportError:  # pragma: no cover - package-style imports
    from research.scripts.run_gpt_oss_pilot import (
        DEFAULT_CONFIG,
        DEFAULT_CONDITIONS,
        DEFAULT_POOL,
        DEFAULT_PREDICTION,
        DEFAULT_TEST_EVIDENCE,
        DEFAULT_TRAIN_EVIDENCE,
        GroqChatClient,
        load_repo_env,
        parse_mapping,
        run_pilot,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "research/experiments/pilot/step47a_cloudflare_gpt_oss_generation.jsonl"
DEFAULT_ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
DEFAULT_MODEL = "@cf/openai/gpt-oss-20b"


def main() -> None:
    load_repo_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_TEST_EVIDENCE)
    parser.add_argument("--training-evidence", type=Path, default=DEFAULT_TRAIN_EVIDENCE)
    parser.add_argument("--prediction-file", type=parse_mapping, default=("loginsight_lstm", DEFAULT_PREDICTION))
    parser.add_argument("--attribution-file", type=parse_mapping)
    parser.add_argument("--architecture", default="loginsight_lstm")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-items", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    # Cloudflare exposes GPT-OSS reasoning tokens inside the completion
    # budget. 512 tokens can be consumed before visible JSON is returned on
    # long prompts, so the provider-specific default is 2048.
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    token = os.getenv("CLOUDFLARE_API_TOKEN")
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if not token:
        raise SystemExit("CLOUDFLARE_API_TOKEN is required; no API call was made")
    if not account_id:
        raise SystemExit("CLOUDFLARE_ACCOUNT_ID is required; no API call was made")

    prediction_architecture, prediction_path = args.prediction_file
    if prediction_architecture != args.architecture:
        raise SystemExit("--prediction-file architecture must match --architecture")
    attribution_path = None
    if args.attribution_file:
        attribution_architecture, attribution_path = args.attribution_file
        if attribution_architecture != args.architecture:
            raise SystemExit("--attribution-file architecture must match --architecture")
    conditions = tuple(value.strip() for value in args.conditions.split(",") if value.strip())
    unknown = set(conditions) - set(DEFAULT_CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")

    client = GroqChatClient(
        api_key=token,
        endpoint=DEFAULT_ENDPOINT.format(account_id=account_id),
        model=DEFAULT_MODEL,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        retries=args.retries,
        extra_body={"reasoning_effort": "low"},
    )
    report = run_pilot(
        pool_path=args.pool.resolve(),
        evidence_path=args.evidence.resolve(),
        training_evidence_path=args.training_evidence.resolve(),
        prediction_path=prediction_path,
        attribution_path=attribution_path,
        output_path=args.output.resolve(),
        config_path=args.config.resolve(),
        architecture=args.architecture,
        max_items=args.max_items,
        repetitions=args.repetitions,
        conditions=conditions,
        top_k=args.top_k,
        shuffle_seed=args.shuffle_seed,
        client=client,
        show_progress=not args.no_progress,
        provider_name="cloudflare_workers_ai",
        run_label="cloudflare_gpt_oss",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
