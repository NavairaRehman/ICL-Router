#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


def load_train(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("train_router.json must be a JSON list of records")
    required = {"query", "model", "is_correct_direct", "task"}
    missing = required - set(data[0].keys())
    if missing:
        raise ValueError(f"Missing fields in train data: {missing}")
    return data


GROUP_ORDER = ["olympiadbench", "bbh_logicbench", "mmlupro", "mbpp"]
GROUP_TASKS = {
    "olympiadbench": ["olympiadbench"],
    "bbh_logicbench": ["bbh", "logicbench"],
    "mmlupro": ["mmlupro"],
    "mbpp": ["mbpp"],
}


def yesno(flag: bool) -> str:
    return "Yes" if flag else "No"


def build_balanced_expert_info(records, total_per_model: int, seed: int):
    # Identify tasks present in data (keep original names)
    tasks_present = set()
    for r in records:
        tasks_present.add(r.get("task"))

    canonical = {"olympiadbench", "bbh", "logicbench", "mmlupro", "mbpp"}
    use_group_mode = tasks_present == canonical

    # Build per-model buckets
    by_task = defaultdict(lambda: defaultdict(list))
    models = set()
    for r in records:
        m = r["model"]
        t = r["task"]
        by_task[m][t].append(r)
        models.add(m)

    rng = random.Random(seed)

    out = {}
    if use_group_mode:
        # Group-mode (exactly the canonical 5 tasks, with bbh+logicbench in one group)
        base = total_per_model // len(GROUP_ORDER)
        rem = total_per_model % len(GROUP_ORDER)

        for model in sorted(models):
            items = []
            # Shuffle buckets for determinism
            for t in ["olympiadbench", "bbh", "logicbench", "mmlupro", "mbpp"]:
                rng.shuffle(by_task[model][t])

            # Determine targets per group for this model
            group_targets = {g: base for g in GROUP_ORDER}
            for g in GROUP_ORDER[:rem]:
                group_targets[g] += 1

            # Sample per group in the specified order, preserving original task names
            for g in GROUP_ORDER:
                n_target = group_targets[g]
                tasks_in_group = GROUP_TASKS[g]

                if len(tasks_in_group) == 1:
                    t = tasks_in_group[0]
                    bucket = by_task[model][t]
                    take = min(n_target, len(bucket))
                    for r in bucket[:take]:
                        items.append({
                            "input": r["query"],
                            "label": yesno(bool(r["is_correct_direct"])),
                            "task": r["task"],
                        })
                else:
                    # Evenly split across tasks within the group (bbh + logicbench)
                    per = n_target // len(tasks_in_group)
                    extra = n_target % len(tasks_in_group)
                    selected = []
                    for t in tasks_in_group:
                        bucket = by_task[model][t]
                        take = min(per, len(bucket))
                        selected.extend((t, r) for r in bucket[:take])
                    # Distribute remainder
                    if extra:
                        for t in tasks_in_group:
                            if extra == 0:
                                break
                            bucket = by_task[model][t]
                            already = sum(1 for tt, _ in selected if tt == t)
                            if already < len(bucket):
                                selected.append((t, bucket[already]))
                                extra -= 1
                    # Borrow if needed
                    if len(selected) < n_target:
                        needed = n_target - len(selected)
                        leftovers = []
                        for t in tasks_in_group:
                            bucket = by_task[model][t]
                            already = sum(1 for tt, _ in selected if tt == t)
                            leftovers.extend((t, r) for r in bucket[already:])
                        rng.shuffle(leftovers)
                        selected.extend(leftovers[:needed])

                    # Randomize order between bbh and logicbench
                    rng.shuffle(selected)
                    for t, r in selected[:n_target]:
                        items.append({
                            "input": r["query"],
                            "label": yesno(bool(r["is_correct_direct"])),
                            "task": r["task"],
                        })

            # Top-up if needed, preserve group order then shuffle leftovers
            if len(items) < total_per_model:
                needed = total_per_model - len(items)
                leftovers = []
                for g in GROUP_ORDER:
                    for t in GROUP_TASKS[g]:
                        used = sum(1 for it in items if it["task"] == t)
                        leftovers.extend({
                            "input": r["query"],
                            "label": yesno(bool(r["is_correct_direct"])),
                            "task": r["task"],
                        } for r in by_task[model][t][used:])
                rng.shuffle(leftovers)
                items.extend(leftovers[:needed])

            out[model] = items[:total_per_model]

        return out, GROUP_ORDER
    else:
        # Generic per-task balancing across whatever tasks appear
        tasks_sorted = sorted(tasks_present)
        base = total_per_model // len(tasks_sorted)
        rem = total_per_model % len(tasks_sorted)

        for model in sorted(models):
            items = []
            # Shuffle buckets for determinism
            for t in tasks_sorted:
                rng.shuffle(by_task[model][t])

            # Task-level targets
            task_targets = {t: base for t in tasks_sorted}
            for t in tasks_sorted[:rem]:
                task_targets[t] += 1

            # Sample per task
            for t in tasks_sorted:
                n_target = task_targets[t]
                bucket = by_task[model][t]
                take = min(n_target, len(bucket))
                for r in bucket[:take]:
                    items.append({
                        "input": r["query"],
                        "label": yesno(bool(r["is_correct_direct"])),
                        "task": r["task"],
                    })

            # Top-up if underfilled: take from any remaining across tasks
            if len(items) < total_per_model:
                needed = total_per_model - len(items)
                leftovers = []
                for t in tasks_sorted:
                    used = sum(1 for it in items if it["task"] == t)
                    leftovers.extend({
                        "input": r["query"],
                        "label": yesno(bool(r["is_correct_direct"])),
                        "task": r["task"],
                    } for r in by_task[model][t][used:])
                rng.shuffle(leftovers)
                items.extend(leftovers[:needed])

            out[model] = items[:total_per_model]

        return out, tasks_sorted


def main():
    p = argparse.ArgumentParser(description="Generate balanced experts_information JSON from train_router.json")
    p.add_argument("--input", default="data/train_router.json", help="Path to train_router.json")
    p.add_argument("--output", default="data/experts_information_500_balanced.json", help="Output JSON path")
    p.add_argument("--total", type=int, default=500, help="Total examples per model (default: 100)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output file")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {out_path}. Use --overwrite to proceed.")

    records = load_train(in_path)
    expert_info, groups = build_balanced_expert_info(records, args.total, args.seed)

    # Quick summary
    summary = {m: Counter(e["task"] for e in v) for m, v in expert_info.items()}
    print("Group order:", groups)
    for m in sorted(summary.keys()):
        print(m, dict(summary[m]), sum(summary[m].values()))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(expert_info, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
