"""
Build mgs_train.pq and mgs_test.pq by combining three datasets:

  1. Math   – DigitalLearningGmbH/MATH-lighteval
              reward_model.style = "rule"
              Handled by verl/utils/reward_score/__init__.py (lighteval/MATH branch)

  2. IF     – nvidia/Llama-Nemotron-Post-Training-Dataset  RL/instruction_following
              reward_model.style = "rule"
              Handled by verl/utils/reward_score/ifeval.py  (nemotron-rl-if)

  3. Chat   – allenai/WildChat-1M  (English, non-toxic, single or multi-turn)
              reward_model.style = "model"
              Scored at training time by Skywork/Skywork-Reward-V2-Qwen3-4B
              via the RewardModelWorker; NaiveRewardManager reads rm_scores.

All rows share the same 5-column schema expected by RLHFDataset / NaiveRewardManager:
  data_source, prompt, ability, reward_model, extra_info

Usage
-----
  python data/mgs/make_mgs_data.py            # defaults: 7500 train / 500 test per source
  python data/mgs/make_mgs_data.py --n_train 5000 --n_test 300
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ── helpers ──────────────────────────────────────────────────────────────────

SEED = 42
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# For math answer extraction (mirrors examples/data_preprocess/math_dataset.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _remove_boxed(s: str):
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return None
    return s[len(left):-1]


def _last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i, right_brace_idx, num_open = idx, None, 0
    while i < len(string):
        if string[i] == "{":
            num_open += 1
        if string[i] == "}":
            num_open -= 1
            if num_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx: right_brace_idx + 1] if right_brace_idx is not None else None


def _extract_math_answer(solution: str):
    return _remove_boxed(_last_boxed_only_string(solution) or "") if solution else None


# ── 1. Math ──────────────────────────────────────────────────────────────────

MATH_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
MATH_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def build_math_rows(split: str, n: int, rng: np.random.Generator) -> list[dict]:
    print(f"\n[Math] Loading {MATH_SOURCE} / {split} ...")
    ds = load_dataset(MATH_SOURCE, split=split)
    print(f"  {len(ds):,} rows available")

    indices = rng.permutation(len(ds))[:n].tolist()
    rows = []
    for local_i, orig_i in enumerate(tqdm(sorted(indices), desc=f"Math {split}")):
        ex = ds[int(orig_i)]
        answer = _extract_math_answer(ex["solution"])
        if answer is None:
            continue
        question = ex["problem"].strip() + " " + MATH_INSTRUCTION
        rows.append({
            "data_source": MATH_SOURCE,
            "prompt": np.array([{"role": "user", "content": question}], dtype=object),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "index": local_i,
                "split": split,
                "type": ex.get("type", ""),
                "level": ex.get("level", ""),
            },
        })
    print(f"  → {len(rows):,} rows after answer extraction")
    return rows


# ── 2. Instruction Following (Nemotron) ──────────────────────────────────────

IF_SOURCE = "nemotron-rl-if"
IF_HF_ID = "nvidia/Llama-Nemotron-Post-Training-Dataset"
IF_TRAIN_FRAC = 0.95


def _clean_if_kwargs(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if v is not None}


def build_if_rows(split: str, n: int, rng: np.random.Generator) -> list[dict]:
    print(f"\n[IF] Loading {IF_HF_ID} RL/instruction_following (streaming) ...")
    # Use streaming to avoid schema-casting issues with the Nemotron dataset.
    ds_stream = load_dataset(IF_HF_ID, name="RL", split="instruction_following", streaming=True)

    # Collect all rows first so we can reproducibly split train/test.
    all_examples: list[dict] = []
    for ex in tqdm(ds_stream, desc="IF scan"):
        all_examples.append(ex)
    print(f"  {len(all_examples):,} total rows")

    all_idx = rng.permutation(len(all_examples))
    n_train = int(len(all_examples) * IF_TRAIN_FRAC)
    pool = all_idx[:n_train] if split == "train" else all_idx[n_train:]
    chosen = sorted(pool[:n].tolist())

    rows = []
    for local_i, orig_i in enumerate(tqdm(chosen, desc=f"IF {split} build")):
        ex = all_examples[int(orig_i)]
        messages = list(ex["input"])
        args = ex["args"]
        ground_truth = {
            "instruction_id_list": args["instruction_id_list"],
            "kwargs": [_clean_if_kwargs(kw) for kw in args["instruction_kwargs"]],
        }
        rows.append({
            "data_source": IF_SOURCE,
            "prompt": np.array(messages, dtype=object),
            "ability": "ifeval",
            # Serialize ground_truth as JSON string so that all sources share a uniform
            # string type in the parquet reward_model.ground_truth column.
            # ifeval.compute_score handles both dict and JSON-string inputs.
            "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth)},
            "extra_info": {
                "index": local_i,
                "split": split,
                "category": ex.get("category", "instruction_following"),
                "reasoning": ex.get("reasoning", ""),
                "used_in_training": ex.get("used_in_training", ""),
                "version": ex.get("version", ""),
                "system_prompt": ex.get("system_prompt", ""),
            },
        })
    print(f"  → {len(rows):,} rows")
    return rows


# ── 3. Chat (WildChat) ───────────────────────────────────────────────────────

CHAT_SOURCE = "wildchat"
CHAT_HF_ID = "allenai/WildChat-1M"
# Skywork-Reward-V2 scores these at training time; ground_truth is unused.
CHAT_TRAIN_FRAC = 0.97  # held-out 3% for test


def _build_chat_prompt(conversation: list[dict]) -> list[dict] | None:
    """Return the conversation with any trailing assistant turns removed."""
    msgs = [{"role": m["role"], "content": m["content"]}
            for m in conversation
            if isinstance(m.get("role"), str) and isinstance(m.get("content"), str) and m["content"].strip()]
    # drop trailing assistant turns to avoid answer leakage
    while msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
    return msgs if msgs else None


def collect_chat_usable(n_total: int) -> list[dict]:
    """Stream WildChat-1M once and return n_total usable (English, non-toxic) items."""
    print(f"\n[Chat] Loading {CHAT_HF_ID} (streaming). Need {n_total:,} usable examples ...")
    ds_stream = load_dataset(CHAT_HF_ID, split="train", streaming=True)
    collected: list[dict] = []
    scanned = 0
    for ex in ds_stream:
        scanned += 1
        if ex.get("language") != "English":
            continue
        if ex.get("toxic"):
            continue
        if ex.get("redacted"):
            continue
        prompt = _build_chat_prompt(ex.get("conversation", []))
        if prompt is None:
            continue
        collected.append({"prompt": prompt, "hash": ex.get("conversation_hash", "")})
        if len(collected) >= n_total:
            break
    print(f"  Scanned {scanned:,} raw rows → collected {len(collected):,} usable")
    return collected


def _make_chat_rows(items: list[dict], split: str, start_idx: int = 0) -> list[dict]:
    rows = []
    for local_i, item in enumerate(items):
        rows.append({
            "data_source": CHAT_SOURCE,
            "prompt": np.array(item["prompt"], dtype=object),
            "ability": "chat",
            # ground_truth is unused for model-style rewards (Skywork RM scores the response).
            # Use empty string so reward_model.ground_truth stays a consistent string type.
            "reward_model": {"style": "model", "ground_truth": ""},
            "extra_info": {
                "index": start_idx + local_i,
                "split": split,
                "conversation_hash": item["hash"],
            },
        })
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build mgs_train.pq and mgs_test.pq")
    parser.add_argument("--n_train", type=int, default=7500,
                        help="Target rows per source in training split (default 7500)")
    parser.add_argument("--n_test", type=int, default=500,
                        help="Target rows per source in test split (default 500)")
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help="Directory to write output parquets (default: same dir as this script)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Collect chat data once for both splits to avoid double streaming.
    chat_pool = collect_chat_usable(args.n_train + args.n_test)
    chat_train_items = chat_pool[: args.n_train]
    chat_test_items = chat_pool[args.n_train: args.n_train + args.n_test]

    for split, n, chat_items in [
        ("train", args.n_train, chat_train_items),
        ("test", args.n_test, chat_test_items),
    ]:
        out_name = "mgs_train.pq" if split == "train" else "mgs_test.pq"
        out_path = os.path.join(args.out_dir, out_name)

        math_rows = build_math_rows(split, n, rng)
        if_rows = build_if_rows(split, n, rng)
        chat_rows = _make_chat_rows(chat_items, split)

        all_rows = math_rows + if_rows + chat_rows
        # shuffle so sources are interleaved, reproducibly
        idx = rng.permutation(len(all_rows)).tolist()
        all_rows = [all_rows[i] for i in idx]

        df = pd.DataFrame(all_rows)
        df.to_parquet(out_path, index=False)
        print(f"\nSaved {len(df):,} rows → {out_path}")
        print(f"  Source breakdown:")
        print(df["data_source"].value_counts().to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
