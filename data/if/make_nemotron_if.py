"""
Download nvidia/Llama-Nemotron-Post-Training-Dataset RL/instruction_following split
and convert to the verl parquet schema.

Columns produced:
  data_source   "nemotron-rl-if"
  prompt        list[{role, content}]  (the user turn from the dataset)
  ability       "ifeval"
  reward_model  {"style": "rule", "ground_truth": {"instruction_id_list": [...], "kwargs": [...]}}
  extra_info    {index, split, category, reasoning, used_in_training, version, system_prompt}

The ground_truth format matches what IFEvalRewardModel.compute_score() expects:
  instruction_id_list  list of IFEval constraint IDs
  kwargs               list of cleaned kwargs dicts (None-valued keys stripped)

Run:
  python data/if/make_nemotron_if.py
"""

import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "nvidia/Llama-Nemotron-Post-Training-Dataset"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FRAC = 0.95
SEED = 42


def _clean_kwargs(raw: dict) -> dict:
    """Strip None-valued keys so IFEval build_description() receives only real args."""
    return {k: v for k, v in raw.items() if v is not None}


def convert(row: dict, idx: int, split: str) -> dict:
    messages = list(row["input"])  # already [{role, content}]

    args = row["args"]
    ground_truth = {
        "instruction_id_list": args["instruction_id_list"],
        "kwargs": [_clean_kwargs(kw) for kw in args["instruction_kwargs"]],
    }

    return {
        "data_source": "nemotron-rl-if",
        "prompt": np.array(messages, dtype=object),
        "ability": "ifeval",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": idx,
            "split": split,
            "category": row.get("category", "instruction_following"),
            "reasoning": row.get("reasoning", ""),
            "used_in_training": row.get("used_in_training", ""),
            "version": row.get("version", ""),
            "system_prompt": row.get("system_prompt", ""),
        },
    }


def main() -> None:
    print(f"Loading {DATASET_ID} RL/instruction_following ...")
    ds = load_dataset(DATASET_ID, name="RL", split="instruction_following")
    print(f"  {len(ds):,} rows")

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(ds))
    n_train = int(len(ds) * TRAIN_FRAC)
    train_idx = sorted(indices[:n_train].tolist())
    val_idx = sorted(indices[n_train:].tolist())
    print(f"  train: {len(train_idx):,}   val: {len(val_idx):,}")

    for name, idx_list in [("train", train_idx), ("val", val_idx)]:
        rows = []
        for i, orig_i in enumerate(tqdm(idx_list, desc=name)):
            rows.append(convert(ds[int(orig_i)], i, name))
        df = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f"nemotron_if_{name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df):,} rows → {out_path}")


if __name__ == "__main__":
    main()
