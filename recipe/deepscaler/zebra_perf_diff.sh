set -xeuo pipefail

# can make training faster, depends on your infrastructure
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# Set how many GPUs we actually have on this node.
export GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
echo "Number of GPUs available: $GPUS_PER_NODE"

NNODES=1
export NNODES

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

echo "Using $NNODES nodes for training..."

# ------------------------------------- Setup xp params ---------------------------------------
project_name='DISC'

adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
MODEL_PATH=Qwen/Qwen2.5-7B
CRITIC_MODEL_PATH=Qwen/Qwen3-0.6B
offload=True # it's a small model, offloading will just slow-down training
rollout_engine=vllm
rollout_mode=sync # can be async to speedup large scale xps
gpu_memory_utilization=0.85
reward_manager=sec
adv_estimator=grpo
shuffle_dataset=true
first_time_dataset_prep=true # prepare dataset

test_freq=10
save_freq=50
total_epochs=100
total_training_steps=600
val_before_train=true

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1
candidate_batch_size=2048 # how many to sample from the dataloader
train_batch_size=256 # how many chosen by the critic
ppo_mini_batch_size=64 # maintain 4 mini-batches as recommended by the paper, see Sec. 5.1
ppo_micro_batch_size_per_gpu=8 # setup depending on your GPU memory
n_resp_per_prompt=8

critic_train_batch_size=256 # number of samples from the replay buffer
replay_buffer_size=2

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 4))
# dapo reward manager params
enable_overlong_buffer=false # true
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Paths and namings
SFT_MODEL=$(basename $MODEL_PATH)
exp_name="zebra_7B_temp_1_topp_0.9_window_avg_10"

# Sampling params at rollouts
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 6))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 8))
offload=true
gen_tp=1
entropy_checkpointing=true # This enables entropy recomputation specifically for the entropy calculation, lowering memory usage during training.

#rollout method
sampling_method=greedy

WORKING_DIR=/workspace/mnt/verl_latest
train_files=${WORKING_DIR}/data/combined/train_zebra.parquet
test_files=${WORKING_DIR}/data/combined/test_zebra.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.shuffle=$shuffle_dataset \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.train_batch_size=${candidate_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=true \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.rollout.sampling_method=${sampling_method} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    reward_model.reward_manager=${reward_manager} \
    +adv_predictor.enable=true \
    +adv_predictor.dormant_steps=20 \
    +adv_predictor.critic_warmup=5 \
    +adv_predictor.replay_buffer_size=${replay_buffer_size} \
    +adv_predictor.train_batch_size=${critic_train_batch_size} \
    +adv_predictor.sampler=uniform \
    +adv_predictor.temperature_annealing=false \
    +adv_predictor.temperature=1.0 \
    +adv_predictor.top_p_annealing=false \
    +adv_predictor.top_p=0.9 \
    +adv_predictor.num_samples=${train_batch_size} \
    +adv_predictor.train_critic_only=false \
    +adv_predictor.ema_coeff=0.2 \
    +adv_predictor.target=perf_diff \
    +adv_predictor.use_sampling_prior=false \
    +adv_predictor.perf_diff_amplifier=1000.0 \
    +adv_predictor.target_importance_ratio_cliprange=3.0 \
    +adv_predictor.use_window_avg_target=true \
    +adv_predictor.history_length=10 \
    critic.optim.lr=1e-6 \
    +critic.model.style=osmd \
    +critic.model.num_labels=1 \
    +critic.model.num_heads=1 \
    +critic.clip_range=0.2 \
    critic.model.use_remove_padding=True \
    critic.model.path=${CRITIC_MODEL_PATH} \
    critic.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.use_legacy_worker_impl=disable \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.resume_mode=disable \
    trainer.log_val_generations=10 \
    $@