"""
Computes the REAL total optimizer-step target implied by experiments/pick_place_simple.yaml
(unmodified paper config), instead of guessing. The yaml only sets `epochs: 2000` (no `batches`
key), and `hem/models/trainer.py::Trainer.train()`'s own math is
`max_batches = epochs * len(train_loader)` when `batches` is absent - but `len(train_loader)`
isn't knowable analytically here because `dataset.type: "agent teacher"`'s same-task pairing
(hem/datasets/agent_teacher_dataset.py::load_ost) makes dataset size a property of the actual
collected files, not just the yaml. This script builds the real train dataset (CPU-only, no GPU
needed) and reproduces Trainer's own arithmetic exactly.

Run AFTER the full 1600x2 data collection (section 2 of the plan), inside the 'awda' conda env:
    conda activate awda
    EXPERT_DATA=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda/dataset \
    PYTHONPATH=. python baseline_train/compute_target_steps.py
"""
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hem.util import parse_basic_config  # noqa: E402
from hem.datasets import get_dataset  # noqa: E402

EXPERIMENT_YAML = os.path.join(REPO_ROOT, "experiments", "pick_place_simple.yaml")


def main():
    os.environ.setdefault(
        "EXPERT_DATA", "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/osvi-awda/dataset")
    cfg = parse_basic_config(EXPERIMENT_YAML)
    ds_cfg = dict(cfg["dataset"])
    ds_type = ds_cfg.pop("type")
    train_ds = get_dataset(ds_type)(**ds_cfg, mode="train")

    n_pairs = len(train_ds)
    batch_size = cfg["batch_size"]
    loader_len = math.ceil(n_pairs / batch_size)
    epochs = cfg.get("epochs", 1)
    batches = cfg.get("batches", 0)
    if batches > 0:
        epochs = math.ceil(batches / loader_len)
        total_steps = batches
    else:
        total_steps = epochs * loader_len

    print(f"train pairs (len(dataset)): {n_pairs}")
    print(f"batch_size: {batch_size}")
    print(f"batches/epoch (len(train_loader)): {loader_len}")
    print(f"epochs (config): {cfg.get('epochs', 1)}")
    print(f"TOTAL_TARGET_STEPS: {total_steps}")


if __name__ == "__main__":
    main()
