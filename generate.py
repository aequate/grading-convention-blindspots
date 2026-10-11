# /// script
# requires-python = ">=3.11"
# ///
"""Regenerate responses with the pinned model; defaults to a CPU-only schedule check."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from inference import ExperimentConfig, load_model, seed_for_case

OUTPUT_FIELDS = (
    "raw_response",
    "generated_token_ids",
    "generated_tokens",
    "input_tokens",
    "finish_reason",
    "token_limit_reached",
    "case_seed",
    "enable_thinking",
)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", nargs="+", default=["greedy"], help="greedy, sample_42–46, or all")
    parser.add_argument("--generate", action="store_true", help="Load weights and run on an NVIDIA GPU")
    parser.add_argument("--output", type=Path, default=root / "regenerated")
    args = parser.parse_args()
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    with (root / "dataset/cases.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    cases = {row["case_id"]: row for row in rows}
    if len(cases) != len(rows) or not cases:
        parser.error("Cases must have unique, nonempty identifiers.")
    blocks = list(settings["blocks"]) if args.blocks == ["all"] else args.blocks
    if len(set(blocks)) != len(blocks) or any(block not in settings["blocks"] for block in blocks):
        parser.error("Select each block once: greedy, sample_42–46, or all.")
    configs = {}
    for block in blocks:
        values = dict(settings["blocks"][block])
        values["precision"] = settings["model"]["compute_precision"]
        config = ExperimentConfig(**values)
        config.validate()
        schedule = settings["schedules"][block]
        scheduled_ids = [entry["case_id"] for entry in schedule]
        if len(schedule) != len(cases) or set(scheduled_ids) != set(cases):
            parser.error(f"The {block} schedule does not match the dataset.")
        for position, entry in enumerate(schedule, 1):
            if (entry["case_seed"] != seed_for_case(config.seed, entry["case_id"])
                    or entry["seed_block"] != config.seed or entry["position"] != position):
                parser.error(f"The {block} schedule has an inconsistent position or seed.")
        configs[block] = config
    planned = len(cases) * len(blocks)
    print(f"{len(cases)} cases; {len(blocks)} blocks; {planned} generations")
    print(f"Model: {settings['model']['id']} at {settings['model']['revision']}")
    print("NF4, float16 compute, thinking disabled, fresh conversation for each response")
    if not args.generate:
        print("Schedule check passed. Add --generate to load the model and produce new responses.")
        return
    output = args.output.resolve()
    if output == root or any(output.is_relative_to(root / name) for name in ("data", "dataset", "results", "notebooks")):
        parser.error("Use a separate output directory for new generations.")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "responses.jsonl"
    if destination.exists():
        parser.error("The output already contains responses. Choose a fresh output directory.")
    base = dict(settings["blocks"]["greedy"])
    base["precision"] = settings["model"]["compute_precision"]
    runner = load_model(ExperimentConfig(**base))
    completed = 0
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        for block, config in configs.items():
            active = runner.with_config(config)
            for entry in settings["schedules"][block]:
                case_id = entry["case_id"]
                prepared = active.prepare(cases[case_id]["prompt"])
                result = active.generate(prepared, entry["case_seed"])
                record = {
                    "block": block,
                    "case_id": case_id,
                    "input": prepared["snapshot"],
                    "output": {field: result[field] for field in OUTPUT_FIELDS},
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                completed += 1
                print(f"{completed}/{planned}: {block} {case_id} ({result['finish_reason']})", flush=True)
    print(f"Responses saved to {destination}. New responses require separate human review.")


if __name__ == "__main__":
    main()
