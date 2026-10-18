#!/usr/bin/env python3
"""Run the Step 47A exploratory Mistral local/Colab generation pilot.

This runner reuses the frozen Step 47 prompt construction and ledger logic for
the active Mistral model. It replaces the historical hosted GPT-OSS client
with a local Transformers client. CUDA uses 4-bit bitsandbytes quantization;
Apple Silicon uses the MPS backend with float16 weights. Requests are
generated sequentially and written to resumable JSONL output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from run_gpt_oss_pilot import (
        DEFAULT_CONFIG,
        DEFAULT_CONDITIONS,
        DEFAULT_POOL,
        DEFAULT_PREDICTION,
        DEFAULT_TEST_EVIDENCE,
        DEFAULT_TRAIN_EVIDENCE,
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
        load_repo_env,
        parse_mapping,
        run_pilot,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "research/experiments/pilot/step47a_mistral_generation.jsonl"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"


class TransparentProgress:
    """Persist and print machine-readable live status for a 47A run."""

    def __init__(self, status_path: Path, *, output_path: Path, model: str, max_items: int | None, conditions: tuple[str, ...], repetitions: int) -> None:
        self.status_path = status_path
        self.base = {
            "step": "47A",
            "status": "initializing",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "output_path": str(output_path),
            "status_path": str(status_path),
            "max_items": max_items,
            "conditions": list(conditions),
            "repetitions": repetitions,
        }

    def __call__(self, event: dict[str, Any]) -> None:
        status = dict(self.base)
        status.update(event)
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        if event["event"] in {"preflight_failed", "failed"}:
            status["status"] = "failed"
        elif event["event"] == "run_finished":
            status["status"] = "finished"
        else:
            status["status"] = "running"
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.status_path)
        counts = event.get("status_counts", {})
        location = event.get("record_id") or ""
        message = (
            f"[47A] {event['event']} "
            f"complete={counts.get('complete', 0)} "
            f"skipped={counts.get('skipped', 0)} "
            f"failed={counts.get('failed', 0)}"
        )
        if location:
            message += f" | {location}"
        print(message, file=sys.stderr, flush=True)


def _preflight(args: argparse.Namespace, conditions: tuple[str, ...], attribution_path: Path | None) -> None:
    required = {
        "annotation_pool": args.pool,
        "test_evidence": args.evidence,
        "training_evidence": args.training_evidence,
        "prediction_file": args.prediction_file[1],
    }
    if attribution_path is not None:
        required["attribution_file"] = attribution_path
    missing = {name: str(path) for name, path in required.items() if not Path(path).exists()}
    if missing:
        raise SystemExit(f"47A preflight failed; missing input artifacts: {json.dumps(missing, sort_keys=True)}")
    attribution_conditions = {"unblinded", "shuffled_integrated_gradients", "unblinded_no_normal", "shuffled_integrated_gradients_no_normal"}
    if any(condition in attribution_conditions for condition in conditions) and attribution_path is None:
        raise SystemExit("47A preflight failed; full conditions require --attribution-file ARCH=PATH")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise SystemExit("47A preflight failed; install PyTorch and Transformers in a CUDA, MPS, or CPU runtime") from error


class MistralTransformersChatClient:
    """Open one local Mistral checkpoint and expose the pilot client contract."""

    endpoint = "local_transformers"
    extra_body: dict[str, Any]

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        max_output_tokens: int,
        four_bit: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "Mistral local inference requires torch and transformers"
            ) from error

        self.model = model_name
        self.model_name = model_name
        self.revision = revision
        self.max_output_tokens = max_output_tokens
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.cuda_available = bool(torch.cuda.is_available())
        self.mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        if self.cuda_available:
            self.device = torch.device("cuda")
            dtype = torch.float16
        elif self.mps_available:
            self.device = torch.device("mps")
            dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32

        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if self.cuda_available:
            load_kwargs["device_map"] = "auto"
        if four_bit and self.cuda_available:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError as error:  # pragma: no cover - depends on runtime
                raise RuntimeError(
                    "4-bit Mistral loading requires bitsandbytes"
                ) from error
        self.loaded_at = time.time()
        self.model_instance = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )
        if not self.cuda_available:
            self.model_instance.to(self.device)
        self.model_instance.eval()
        resolved_revision = getattr(self.model_instance.config, "_commit_hash", None)
        self.resolved_revision = resolved_revision or revision
        self.extra_body = {
            "revision": self.resolved_revision,
            "quantization": "4-bit-nf4-double-quant" if four_bit and self.cuda_available else "none",
            "runtime": {
                "python": __import__("sys").version,
                "torch": torch.__version__,
                "cuda_available": self.cuda_available,
                "mps_available": self.mps_available,
                "device": str(self.device),
                "gpu": torch.cuda.get_device_name(0) if self.cuda_available else None,
            },
        }

    def generate(self, prompt: str, *, request_id: str) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": "Return only one JSON object matching the requested explanation schema.",
            },
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model_instance.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw_text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = int(output.shape[1] - prompt_tokens)
        payload = {
            "id": request_id,
            "object": "chat.completion",
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": raw_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "local_generation_seconds": time.monotonic() - started,
        }
        return {"provider_payload": payload, "raw_text": raw_text, "attempt": 1}


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
    parser.add_argument("--full", action="store_true", help="Run the complete annotation pool; mutually exclusive with --max-items")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--status-file", type=Path, help="Live JSON status path (defaults beside --output)")
    args = parser.parse_args()

    if args.full and args.max_items != 2:
        raise SystemExit("use either --full or --max-items, not both")
    if args.full:
        args.max_items = None

    prediction_architecture, prediction_path = args.prediction_file
    if prediction_architecture != args.architecture:
        raise SystemExit("--prediction-file architecture must match --architecture")
    attribution_path = None
    if args.attribution_file:
        attribution_architecture, attribution_path = args.attribution_file
        if attribution_architecture != args.architecture:
            raise SystemExit("--attribution-file architecture must match --architecture")
    conditions = tuple(value.strip() for value in args.conditions.split(",") if value.strip())
    status_path = args.status_file or args.output.with_suffix(".status.json")
    reporter = TransparentProgress(
        status_path,
        output_path=args.output.resolve(),
        model=args.model,
        max_items=args.max_items,
        conditions=conditions,
        repetitions=args.repetitions,
    )
    reporter({"event": "preflight_started", "status_counts": {"complete": 0, "skipped": 0, "failed": 0}})
    try:
        _preflight(args, conditions, attribution_path)
    except SystemExit as error:
        reporter({
            "event": "preflight_failed",
            "error": str(error),
            "status_counts": {"complete": 0, "skipped": 0, "failed": 0},
        })
        raise
    reporter({"event": "loading_model", "status_counts": {"complete": 0, "skipped": 0, "failed": 0}})

    client = MistralTransformersChatClient(
        model_name=args.model,
        revision=args.revision,
        max_output_tokens=args.max_output_tokens,
        four_bit=not args.no_4bit,
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
        provider_name="huggingface_local_or_colab",
        run_label="mistral_7b",
        progress_callback=reporter,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
