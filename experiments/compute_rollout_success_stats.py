"""
Reports per-task and aggregate (in-training vs. out-of-training) rollout success rates for a
baseline checkpoint, from the log.db written by scripts/evaluate.py.

Why not just read the aggregate `eval/success_rate_in_dist` / `eval/success_rate_out_dist` tags
that scripts.eval_util.test_transformer computes: those two aggregate tags (and
eval/mean_reward_in_dist, eval/mean_reward_out_dist) are only written when a SummaryWriter exists,
which scripts/evaluate.py only builds when `--bn` is omitted (multi-checkpoint/--watch mode) - see
eval_util.py's `if writer is not None: ... dblog.log(...)` guards around lines 381-393. Single
checkpoint runs (`--bn <step>`, e.g. baseline_train/eval_paper_scale.sh) leave those two aggregate
tags missing from log.db entirely, even though the per-task tag
`eval/{task_id}/success_rate_out_dist` (misleadingly named "out_dist" for every task, in-dist or
out) IS always written, unguarded. So this script recomputes the in/out aggregates itself from the
per-task tags plus config.yaml's train_tasks/test_tasks split, rather than relying on the missing
aggregate tags.

Usage:
    python experiments/compute_rollout_success_stats.py <checkpoint_dir> [--step N] [--out-json PATH]

<checkpoint_dir> is a single run's dir (contains config.yaml, log.db, model_save-*.pt), e.g.
.../osvi_awda_baseline/pick_place_simple_test0_5_10_15/bc_inv_ckpt-1786176302
If --step is omitted, uses the highest step present for any per-task success_rate_out_dist tag.
Besides the printed table, always writes a summary JSON (default:
<checkpoint_dir>/rollout_success_stats_<step>.json, override with --out-json).
"""
import argparse
import os
import sqlite3
import json

import yaml


def task_label(task_id):
    items = ["Milk", "Bread", "Cereal", "Can"]
    object_id, bin_id = task_id // 4, task_id % 4
    item = items[object_id] if object_id < len(items) else f"obj{object_id}"
    return f"{item}_bin{bin_id}"


def read_tag(con, tag):
    """Returns {step: value} for a given db tag."""
    rows = con.execute("select ind, val from log where tag = ?", (tag,)).fetchall()
    return {int(ind): json.loads(val) for ind, val in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--step", type=int, default=None,
                         help="checkpoint step to report; defaults to the latest step with logged results")
    parser.add_argument("--out-json", default=None,
                         help="where to write the summary JSON (default: <checkpoint_dir>/rollout_success_stats_<step>.json)")
    args = parser.parse_args()

    config_path = os.path.join(args.checkpoint_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train_tasks = config["dataset"]["train_tasks"]
    test_tasks = config["dataset"]["test_tasks"]

    db_path = os.path.join(args.checkpoint_dir, "log.db")
    con = sqlite3.connect(db_path)

    all_tasks = train_tasks + test_tasks
    per_task_sr = {t: read_tag(con, f"eval/{t}/success_rate_out_dist") for t in all_tasks}
    per_task_rw = {t: read_tag(con, f"eval/{t}/mean_reward_out_dist") for t in all_tasks}

    steps = sorted(set().union(*[set(d.keys()) for d in per_task_sr.values()]))
    if not steps:
        raise SystemExit(f"no per-task success_rate_out_dist entries found in {db_path}")
    step = args.step if args.step is not None else steps[-1]
    if step not in steps:
        raise SystemExit(f"step {step} not found; available steps: {steps}")

    print(f"{args.checkpoint_dir}\nstep {step}\n")
    print(f"{'task':>6}  {'label':<12} {'split':<12} {'success_rate':>13} {'mean_reward':>12}")
    task_records = []
    in_srs, out_srs = [], []
    for t in all_tasks:
        sr = per_task_sr[t].get(step)
        rw = per_task_rw[t].get(step)
        split = "in-training" if t in train_tasks else "out-of-training"
        if t in train_tasks and sr is not None:
            in_srs.append(sr)
        elif t in test_tasks and sr is not None:
            out_srs.append(sr)
        sr_str = f"{sr:.3f}" if sr is not None else "n/a"
        rw_str = f"{rw:.3f}" if rw is not None else "n/a"
        print(f"{t:>6}  {task_label(t):<12} {split:<12} {sr_str:>13} {rw_str:>12}")
        task_records.append({
            "task_id": t,
            "label": task_label(t),
            "split": split,
            "success_rate": sr,
            "mean_reward": rw,
        })

    print()
    if in_srs:
        print(f"in-training  (n={len(in_srs)} tasks) avg success rate: {sum(in_srs)/len(in_srs):.3f}")
    else:
        print("in-training: no data")
    if out_srs:
        print(f"out-of-training (n={len(out_srs)} tasks) avg success rate: {sum(out_srs)/len(out_srs):.3f}")
    else:
        print("out-of-training: no data")

    con.close()

    summary = {
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "step": step,
        "tasks": task_records,
        "aggregate": {
            "in_training": {
                "n_tasks": len(in_srs),
                "avg_success_rate": sum(in_srs) / len(in_srs) if in_srs else None,
            },
            "out_of_training": {
                "n_tasks": len(out_srs),
                "avg_success_rate": sum(out_srs) / len(out_srs) if out_srs else None,
            },
        },
    }
    out_json = args.out_json or os.path.join(args.checkpoint_dir, f"rollout_success_stats_{step}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
