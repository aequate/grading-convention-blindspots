# /// script
# requires-python = ">=3.11"
# ///
"""Recalculate descriptive results from the packaged final reviews; no GPU or dependencies required."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

CONDITIONS = ("N0", "N_AR", "N_CAIRO", "N_FULL", "N_GUC", "R_LOW", "R_HIGH")
OUTCOMES = {'expressed_direction': ('higher_is_better', 'lower_is_better', 'conditional', 'not_expressed', 'inconsistent_or_unclear'), 'rubric_consistency': ('consistent', 'opposite_direction', 'other_error', 'not_applicable', 'unresolved'), 'input_tracking': ('preserved', 'changed', 'unresolved'), 'arithmetic': ('correct', 'incorrect', 'omitted', 'not_applicable', 'unresolved'), 'reporting_policy': ('supported_or_conditional', 'unsupported', 'not_applicable', 'unresolved'), 'range_compatibility': ('correct', 'incorrect', 'omitted', 'not_applicable', 'unresolved')}
METADATA_FIELDS = ("study", "core_id", "condition_id", "record_class", "task_family", "cue", "rubric")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(root):
    settings = json.loads((root / "settings.json").read_text(encoding="utf-8"))
    plan = {"study": settings["study"], "planned_blocks": settings["blocks"]}
    cases = read_csv(root / "dataset/cases.csv")
    case_ids = [case["case_id"] for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate case identifiers.")
    metadata = {case["case_id"]: {key:case[key] for key in METADATA_FIELDS} for case in cases}
    cores = sorted({case["core_id"] for case in cases if case["record_class"] == "main"})
    responses = {block:{} for block in settings["blocks"]}
    annotations = {block:{} for block in settings["blocks"]}
    for line in (root / "data/responses.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        block, cid = record["block"], record["case_id"]
        if block not in responses or cid not in metadata or cid in responses[block]:
            raise ValueError(f"Invalid or duplicate response: {block} {cid}")
        if len(record["output"]["generated_token_ids"]) != record["output"]["generated_tokens"]:
            raise ValueError(f"Token count mismatch: {block} {cid}")
        responses[block][cid] = record
    for row in read_csv(root / "data/reviews.csv"):
        block, cid = row["block"], row["case_id"]
        if block not in annotations or cid not in metadata or cid in annotations[block]:
            raise ValueError(f"Invalid or duplicate review: {block} {cid}")
        if cid not in responses[block] or row["review_status"] != "completed":
            raise ValueError(f"Review is missing its response or is incomplete: {block} {cid}")
        for key in ("is_correct", "unsupported_assumption", "abstained"):
            if row[key].lower() not in {"true","false"}:
                raise ValueError(f"Invalid boolean: {block} {cid} {key}")
            row[key] = row[key].lower() == "true"
        for key, values in OUTCOMES.items():
            if row[key] not in values or row[key] == "unresolved":
                raise ValueError(f"Invalid final outcome: {block} {cid} {key}")
        annotations[block][cid] = row
    runs = {block for block, records in responses.items() if records}
    modes = {}
    for mode in ("greedy", "sampled"):
        blocks = [
            name for name, cfg in plan["planned_blocks"].items() if cfg["do_sample"] == (mode == "sampled")
        ]
        prompts = {}
        for case_id in case_ids:
            complete = [
                annotations[name][case_id]
                for name in blocks
                if annotations.get(name, {}).get(case_id, {}).get("review_status") == "completed"
            ]
            n = len(complete)
            completed_outputs = sum(case_id in responses.get(name, {}) for name in blocks)
            prompts[case_id] = {
                **metadata[case_id],
                "case_id": case_id,
                "planned_draws": len(blocks),
                "completed_outputs": completed_outputs,
                "missing_outputs": len(blocks) - completed_outputs,
                "reviewed_draws": n,
                "unreviewed_or_pending_outputs": completed_outputs - n,
                "complete_review": n == len(blocks),
                "correctness": sum(a["is_correct"] for a in complete) / n if n else None,
                "unsupported_assumption": sum(a["unsupported_assumption"] for a in complete) / n
                if n
                else None,
                "abstention": sum(a["abstained"] for a in complete) / n if n else None,
                "outcomes": {
                    key: {
                        value: sum(a[key] == value for a in complete) / n if n else None
                        for value in values
                        if value != "unresolved"
                    }
                    for key, values in OUTCOMES.items()
                },
            }
        conditions = {}
        for condition in CONDITIONS:
            members = [
                r for r in prompts.values() if r["record_class"] == "main" and r["condition_id"] == condition
            ]
            qualified = [r for r in members if r["complete_review"]]
            ncores = len(qualified)
            conditions[condition] = {
                "eligible_cores": len(members),
                "complete_reviewed_cores": ncores,
                "planned_draws": sum(r["planned_draws"] for r in members),
                "completed_outputs": sum(r["completed_outputs"] for r in members),
                "missing_outputs": sum(r["missing_outputs"] for r in members),
                "reviewed_draws": sum(r["reviewed_draws"] for r in members),
                "unreviewed_or_pending_outputs": sum(r["unreviewed_or_pending_outputs"] for r in members),
                "mean_core_correctness": sum(r["correctness"] for r in qualified) / ncores
                if ncores
                else None,
                "mean_core_unsupported_assumption": sum(r["unsupported_assumption"] for r in qualified)
                / ncores
                if ncores
                else None,
                "mean_core_abstention": sum(r["abstention"] for r in qualified) / ncores if ncores else None,
                "mean_core_outcomes": {
                    key: {
                        value: sum(r["outcomes"][key][value] for r in qualified) / ncores if ncores else None
                        for value in values
                        if value != "unresolved"
                    }
                    for key, values in OUTCOMES.items()
                },
            }
        core_conditions = {
            (row["core_id"], row["condition_id"]): row
            for row in prompts.values()
            if row["record_class"] == "main"
        }
        comparisons = {}
        for left, right in [("N0", c) for c in ("N_AR", "N_CAIRO", "N_FULL", "N_GUC")] + [
            ("N_FULL", "N_GUC"),
            ("R_LOW", "R_HIGH"),
        ]:
            pairs = []
            for core in cores:
                a, b = core_conditions[core, left], core_conditions[core, right]
                if a["complete_review"] and b["complete_review"]:
                    pairs.append(
                        {
                            "core_id": core,
                            "correctness_difference": b["correctness"] - a["correctness"],
                            "higher_direction_difference": b["outcomes"]["expressed_direction"][
                                "higher_is_better"
                            ]
                            - a["outcomes"]["expressed_direction"]["higher_is_better"],
                            "opposite_direction_difference": (
                                b["outcomes"]["rubric_consistency"]["opposite_direction"]
                                - a["outcomes"]["rubric_consistency"]["opposite_direction"]
                                if left == "R_LOW" and right == "R_HIGH"
                                else None
                            ),
                        }
                    )
            comparisons[f"{right}_minus_{left}"] = {
                "left": left,
                "right": right,
                "eligible_cores": len(cores),
                "complete_reviewed_pairs": len(pairs),
                "pairs": pairs,
                **{
                    f"mean_{key}": (
                        sum(p[key] for p in pairs) / len(pairs)
                        if pairs and all(p[key] is not None for p in pairs)
                        else None
                    )
                    for key in (
                        "correctness_difference",
                        "higher_direction_difference",
                        "opposite_direction_difference",
                    )
                },
            }
        main = [r for r in prompts.values() if r["record_class"] == "main"]
        controls = [r for r in prompts.values() if r["record_class"] == "control"]
        modes[mode] = {
            "planned_blocks": blocks,
            "available_blocks": [b for b in blocks if b in runs],
            "counts": {
                key: sum(r[key] for r in main)
                for key in (
                    "planned_draws",
                    "completed_outputs",
                    "missing_outputs",
                    "reviewed_draws",
                    "unreviewed_or_pending_outputs",
                )
            },
            "conditions": conditions,
            "matched_comparisons": comparisons,
            "per_prompt": prompts,
            "range_controls": controls,
            "control_counts": {
                key: sum(r[key] for r in controls)
                for key in (
                    "planned_draws",
                    "completed_outputs",
                    "missing_outputs",
                    "reviewed_draws",
                    "unreviewed_or_pending_outputs",
                )
            },
        }
    return {
        "study": plan["study"],
        "modes": modes,
        "independent_core_count": len(cores),
        "uncertainty_method": "Descriptive only. Seed draws nested within prompt; conditions matched within core. "
        "Equal core weighting uses complete planned reviews; controls remain separate. No p-values.",
    }

def condition_csv(summary):
    fields = ("mode", "condition", "eligible_cores", "complete_reviewed_cores", "planned_draws",
              "completed_outputs", "missing_outputs", "reviewed_draws", "unreviewed_or_pending_outputs",
              "mean_core_correctness", "mean_core_unsupported_assumption", "mean_core_abstention",
              *OUTCOMES["expressed_direction"])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
    writer.writeheader()
    for mode, results in summary["modes"].items():
        for condition, row in results["conditions"].items():
            writer.writerow({"mode":mode,"condition":condition,**row,
                             **row["mean_core_outcomes"]["expressed_direction"]})
    return stream.getvalue()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Compare with the saved results without writing")
    action.add_argument("--write", action="store_true", help="Write recalculated summaries")
    parser.add_argument("--output",type=Path,default=root/"results")
    args = parser.parse_args()
    summary = summarize(root)
    table = condition_csv(summary)
    if args.write:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"summary_metrics.json").write_text(
            json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
        (args.output/"condition_metrics.csv").write_text(table,encoding="utf-8",newline="\n")
        print(f"Summaries written to {args.output}")
    else:
        saved = json.loads((args.output/"summary_metrics.json").read_text(encoding="utf-8"))
        if summary != saved:
            raise SystemExit("Saved summary differs from the final reviews.")
        if table != (args.output/"condition_metrics.csv").read_text(encoding="utf-8"):
            raise SystemExit("Saved condition table differs from the final reviews.")
        print("Saved summaries match the final reviews.")
    print(f"Independent cores: {summary['independent_core_count']}")
    for mode, result in summary["modes"].items():
        print(f"{mode}: {result['counts']['reviewed_draws']} main reviews; "
              f"{result['control_counts']['reviewed_draws']} control reviews")
        for condition in ("R_LOW","R_HIGH"):
            row=result["conditions"][condition]
            print(f"  {condition}: full correctness {row['mean_core_correctness']:.1%}; "
                  f"key consistency {row['mean_core_outcomes']['rubric_consistency']['consistent']:.1%}")


if __name__ == "__main__":
    main()
