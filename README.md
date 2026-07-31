# Actor-Curator: Co-adaptive Curriculum Learning via Policy-Improvement Bandits for Scalable RL Post-training

[**🌐 Project page**](https://actor-curator.github.io/) · [**📄 arXiv:2602.20532**](https://arxiv.org/abs/2602.20532)

Reference implementation for **Actor-Curator (AC)** — a scalable, fully automated framework that learns a
neural *curator* to adaptively select training problems for RL post-training of LLMs, by directly optimizing
for expected policy improvement. Problem selection is cast as a non-stationary stochastic bandit and the
curator is trained online with an **Online Stochastic Mirror Descent (OSMD)** objective stabilized by a
PPO-style proximal clip.

> Across Countdown, Zebra, ARC-1D, MATH, AMC, and AIME24, Actor-Curator consistently beats uniform sampling
> and strong learning-based baselines — e.g. **+30.5% on ARC-1D** and **+28.6% on AIME24** peak accuracy over
> the strongest baseline, with **up to ~80% fewer GPU-hours** to reach comparable performance.

This repository is a fork of **[verl](https://github.com/volcengine/verl)** (Apache-2.0, ByteDance Seed team);
the Actor-Curator method is implemented on top of verl's PPO training loop. See
[Attribution & license](#attribution--license).

---

## How it works

At each RL step the curator scores every candidate problem, samples a training subset (instead of uniform
sampling), the actor is updated on that subset, and a **bandit reward based on realized post-update policy
improvement** trains the curator. As the actor improves, the curator adapts to keep selecting the
highest-improvement problems.

- **Curator** — a small network (Qwen3-0.6B) that assigns a positive score `w_φ(x)` to each problem, inducing a
  distribution `p_φ(x) ∝ w_φ(x)`.
- **Utility signal** — per-problem policy improvement `Û_x = importance-weighted average advantage` (the
  `perf_diff` target), estimated from a single forward pass of the *updated* actor on the previous rollouts.
- **Objective** — an OSMD surrogate with a proximal clip (`L_PCO`) that keeps curator updates close to mirror
  descent.

### Where the method lives in the code
| Component | File | What |
|---|---|---|
| Curator training-loop integration | `verl/trainer/ppo/ray_trainer.py` | PHASE 1 (pre-rollout) scores the pool & selects the subset; PHASE 2 (post-rollout) builds the target, trains the curator, maintains the replay buffer |
| Samplers (`uniform`/`softmax`/`stochastic_topk`/`osmd`/`metropolis`) | `verl/trainer/ppo/ray_trainer.py` | the 5-way selection dispatch + warmup/dormant gating + annealing |
| OSMD / PCO loss & curator head | `verl/workers/roles/critic.py` | `osmd_loss_fn` (proximal-clipped OSMD), per-style post-fns; the curator **reuses the Critic role** |
| Curator == critic wiring | `verl/trainer/ppo/utils.py` | `need_critic` returns `adv_predictor.enable` |
| `perf_diff` target pipeline | `verl/trainer/ppo/ray_trainer.py` | extra `compute_log_prob` after the actor update → importance ratio → windowed policy-improvement target |
| Reward verifiers | `verl/utils/reward_score/{countdown,zebra,arc}.py` | rule-based scorers for the reasoning benchmarks |

---

## Installation

This fork tracks **verl `0.5.x`** (see `verl/version/version`) and does not change the dependency stack, so the
simplest and most reproducible setup is to use the **official verl Docker image that matches this verl version**.
Using a verl-0.4 (or other) image will pull a mismatched CUDA/torch/vLLM/transformers stack and is not supported.

```bash
# Recommended: the verl 0.5 app image (FSDP + vLLM rollout), matching this repo's verl version.
docker run --gpus all -it \
  verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 bash
# (SGLang backend: verlai/verl:app-verl0.5-transformers4.55.4-sglang0.4.10.post2-mcore0.13.0-te2.2)

# then install this repo into the image:
git clone <this-repo-url> && cd verl_latest
pip install -e .
```

See [`docker/README.md`](docker/README.md) for the full image matrix; always pick the **`app-verl0.5-…`** tag
(the base image is `verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.7.4`). The paper's experiments were run
with the FSDP backend + vLLM rollout on **2× A100 (80GB)** for the 3B actor.

---

## Data

The runs read pre-built problem banks (parquet) from `${WORKING_DIR}/data/`:

| Benchmark | Train bank | Test bank | Size |
|---|---|---|---|
| Countdown | `data/combined/train_countdown.parquet` | `data/combined/test_countdown.parquet` | 30k |
| Zebra | `data/combined/train_zebra.parquet` | `data/combined/test_zebra.parquet` | 30k |
| ARC-1D | `data/combined/train_arc.parquet` | `data/combined/test_arc.parquet` | 30k |
| MATH | `data/math/math_train.parquet` | `data/math/math_test.parquet` | 12k |

**Obtain the banks and place them under `data/`** (download link / dataset release — see project page).
The MATH/DeepScaleR + AIME data can additionally be regenerated with:

```bash
python recipe/deepscaler/prepare_data.py   # writes data/deepscaler_math.parquet, data/aime2425.parquet
```

> **Note:** the Countdown/Zebra/ARC banks are supplied as artifacts; a standalone generator for them is not
> included in this repo.

---

## Reproducing the main results

Every experiment is a single script in [`recipe/deepscaler/`](recipe/deepscaler/). The **method is defined by
config knobs**, not the filename:

| Method | `adv_predictor.enable` | `adv_predictor.target` | `adv_predictor.sampler` | `critic.model.style` |
|---|---|---|---|---|
| **Actor-Curator (AC, ours)** | `true` | `perf_diff` | `uniform` | `osmd` |
| **Uniform** (baseline) | `false` | — | — | — |
| **SEC** (abs-adv buckets) | `true` | `abs_adv` | `stochastic_topk` | `osmd` |
| **PCL** (success-prob value) | `true` | `abs_adv` | `uniform` | `value_head` |

Common backbone: actor = **Qwen2.5-3B** trained with **GSPO** (`policy_loss.loss_mode=gspo`); curator =
**Qwen3-0.6B**. Alternative actors (Llama3.2-3B, Qwen2.5-7B) are in App. K.

### Quick start — Countdown (the fully self-contained 3B example)

Countdown ships a clean Qwen2.5-3B script for all four methods:

```bash
# 1) point the scripts at your checkout + data (they default to /workspace/mnt/verl_latest)
#    edit WORKING_DIR near the top of each script, or symlink your data there.

cd recipe/deepscaler
bash countdown_perf_diff.sh   # Actor-Curator (AC)
bash countdown_baseline.sh    # Uniform
bash countdown_topk.sh        # SEC
bash countdown_ps.sh          # PCL
```

### Per-benchmark scripts

| Benchmark | AC (ours) | Uniform | SEC | PCL |
|---|---|---|---|---|
| Countdown | `countdown_perf_diff.sh` | `countdown_baseline.sh` | `countdown_topk.sh` | `countdown_ps.sh` |
| Zebra | `zebra_perf_diff.sh` * | `zebra_baseline.sh` | — † | `zebra_ps_value.sh` |
| ARC-1D | `arc_perf_diff.sh` * | `arc_baseline.sh` ‡ | — † | `arc_ps_value.sh` |
| MATH | `math_perf_diff.sh` * | `math_baseline.sh` ‡ | — † | — † |

`*` The `arc/zebra/math` `*_perf_diff.sh` scripts ship with **Qwen2.5-7B** (App. K). To reproduce the **3B**
main-table numbers, set `MODEL_PATH=Qwen/Qwen2.5-3B` at the top of the script — the curator config is otherwise
identical.
`‡` `arc_baseline.sh` / `math_baseline.sh` ship at 7B; set `MODEL_PATH=Qwen/Qwen2.5-3B` for the 3B baseline.
`†` A clean 3B script for this cell is not included; see [Known gaps](#known-gaps--notes).

### Evaluation protocol

Validation runs on the held-out `test_*` banks during training. Following the paper, report the **peak test
accuracy within the first 100 steps** (`trainer.test_freq` controls cadence; training curves in the paper go to
500 steps). Reward is rule-based (exact-match / verifier), so no reward model is needed.

### Alternative models (App. K)
- **Qwen2.5-7B:** `{arc,math,zebra}_perf_diff.sh`, `{arc,math}_baseline.sh` (ship at 7B).
- **Llama3.2-3B:** `math_topk.sh`, `zebra_topk.sh`.
- **Qwen3-1.7B:** `math_baseline_qwen3.sh`.

---

## Config reference

The curator is configured entirely through Hydra overrides. `+adv_predictor.*` controls the curriculum logic;
because the curator **reuses verl's Critic role**, the curator *model* is configured under `critic.*`. Below is
every knob the Actor-Curator scripts set.

**Enable & selection budget**
| Knob | Meaning |
|---|---|
| `adv_predictor.enable` | Master switch. When `true`, the curator replaces the value critic and drives problem selection each step; when `false` you get plain uniform sampling (the Uniform baseline). |
| `adv_predictor.num_samples` | How many problems the curator selects from the candidate pool to actually roll out and train the actor on each step (the effective training batch). |

**What the curator optimizes (bandit target)**
| Knob | Meaning |
|---|---|
| `adv_predictor.target` | The per-problem learning signal: `perf_diff` = realized policy improvement (importance-weighted advantage from the *updated* actor) — the Actor-Curator signal; `abs_adv` = mean \|advantage\| per problem — the heuristic used by SEC/ablations. |
| `adv_predictor.perf_diff_amplifier` | Scalar gain on the `perf_diff` target (the raw signal is tiny; e.g. `1000.0`) so the curator gets a usable gradient. |
| `adv_predictor.use_sampling_prior` | If `true`, divide the estimate by the sampling probability `q(x)` (importance correction) instead of treating the drawn subset as unweighted. |
| `adv_predictor.use_window_avg_target` / `history_length` | Average a problem's target over the last `history_length` times it was sampled (sliding window) to reduce bandit-feedback variance. |
| `adv_predictor.ema_coeff` | EMA coefficient for smoothing difficulty-level / target statistics across steps. |

**How problems are selected (sampler)**
| Knob | Meaning |
|---|---|
| `adv_predictor.sampler` | Selection rule over the scored pool: `uniform` (nucleus top-p over `softmax(scores/τ)` padded with a uniform tail — the paper's default), `softmax` (Gumbel-softmax top-k), `stochastic_topk` (biases toward mid-difficulty; used by SEC), `osmd` (mirror-descent sampler with exploration floor), `metropolis` (Metropolis-Hastings accept/reject vs. the previous batch). |
| `adv_predictor.temperature` / `temperature_annealing` | Softmax temperature τ mapping curator scores → sampling distribution, and whether to anneal τ over training. |
| `adv_predictor.top_p` / `top_p_annealing` | Nucleus cutoff for the `uniform` sampler: fraction of mass drawn from the top-scored problems; the remainder is filled uniformly at random (an exploration floor). Optionally annealed. |

**Warm-up & scheduling**
| Knob | Meaning |
|---|---|
| `adv_predictor.dormant_steps` | Initial steps that select problems *randomly* before the learned curator takes over — lets the actor/curator warm up on unbiased data. |
| `adv_predictor.critic_warmup` | Steps to pre-train the curator (on collected targets) before it starts influencing selection. |
| `adv_predictor.train_critic_only` | Debug/ablation: update only the curator, freezing the actor. |

**Curator optimization**
| Knob | Meaning |
|---|---|
| `adv_predictor.train_batch_size` | Minibatch size for the curator's own SGD updates. |
| `adv_predictor.replay_buffer_size` | Number of past steps' (problem, target) data retained to train the curator (cross-step replay). |

**Curator model & loss (`critic.*`)**
| Knob | Meaning |
|---|---|
| `critic.model.style` | Curator head + loss function: `osmd` = proximal-clipped OSMD classifier (Actor-Curator & SEC); `value_head` = predict success probability (PCL); `ordinal` = ordinal/cumulative-link head (ablation). |
| `critic.model.path` | The curator model — **Qwen3-0.6B** in all paper runs. |
| `critic.clip_range` | The PCO proximal clip ρ: clamps the curator distribution ratio `p_φ/p_t` (PPO-style) for stable OSMD updates. |
| `critic.model.num_labels` / `num_heads` | Output head shape (`1` for the scalar-score `osmd` curator). |
| `critic.optim.lr` | Curator learning rate (e.g. `1e-6`). |
| `critic.model.use_remove_padding`, `critic.model.fsdp_config.{param_offload,optimizer_offload}` | Throughput/memory: sequence packing + FSDP CPU-offload of curator params/optimizer state. |

**Actor update (unchanged verl knobs)**
| Knob | Meaning |
|---|---|
| `actor_rollout_ref.actor.policy_loss.loss_mode` | Actor RL algorithm: `gspo` (paper default) or `grpo`. |
| `algorithm.adv_estimator` | Advantage estimator feeding both the actor and the curator's target (`grpo`). |

**Paper ablations** (also scripted): absolute-adv (`*_perf_diff.sh` with `target=abs_adv`), regression/ordinal
head (`*_ordinal.sh`), sampler variants (`*_topk.sh`, `*_metropolis.sh`), and a GRPO actor
(`countdown_perf_diff_grpo.sh`).

---

## Gaps & notes

- **`WORKING_DIR` is hardcoded** to `/workspace/mnt/verl_latest` in each script — edit it (or symlink) to your
  checkout/data location.
- **Data banks** for Countdown/Zebra/ARC are provided as artifacts, not regenerable from this repo (only
  MATH/DeepScaleR via `prepare_data.py`).
- **Not every dataset×method cell has a clean 3B script.** Countdown is complete; for the others, the AC and
  Uniform scripts default to 7B (change `MODEL_PATH` for 3B), and SEC-on-3B / MATH-PCL scripts are not included.
- `recipe/deepscaler/` also contains **exploratory runs** (contrastive, curriculum, greso, metropolis, ordinal,
  critic-only, tau/model sweeps) that are **not part of the paper's main results** — ignore them for
  reproduction.

---

## Citation

```bibtex
@article{gu2026actorcurator,
  title   = {Actor-Curator: Co-adaptive Curriculum Learning via Policy-Improvement Bandits for Scalable RL Post-Training},
  author  = {Gu, Zhengyao and Light, Jonathan and Astudillo, Raul and Ye, Ziyu and He, Langzhou and
             Zou, Henry Peng and Cheng, Wei and Paternain, Santiago and Yu, Philip S. and Yue, Yisong},
  journal = {arXiv preprint arXiv:2602.20532},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.20532}
}
```

Project page: https://actor-curator.github.io/ · Paper: https://arxiv.org/abs/2602.20532

## Attribution & license

This project is a fork of **[verl](https://github.com/volcengine/verl)** and is distributed under the
**Apache License 2.0** (see [`LICENSE`](LICENSE) and [`Notice.txt`](Notice.txt), © 2023–2024 ByteDance Ltd.).
The Actor-Curator additions (curator integration, OSMD/PCO loss, samplers, `perf_diff` target, and the
`recipe/deepscaler/` experiment scripts) are contributed on top of that base and released under the same license.
