set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=128
group_size=8
export MODEL_PATH=./xiaoyingzhang/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75
export RUN_NAME=grpo_webshop_qwen25_7b_instruct_intrinsic_rewards_pairwise_mem_potential_grounding_credit_assignment_mem_half_group_retrieval_sim_utility_ucb_large_invalid_penality_gen_length2048
export HDFS_CHECKPOINT_PATH=./xiaoyingzhang/agent_training/qwen25_7b_instruct_retroagent_grpo_webshop/ckpts
mkdir -p $HDFS_CHECKPOINT_PATH
export HYDRA_FULL_ERROR=1

export REFLECTION_FILE="./xiaoyingzhang/agent_training/qwen25_7b_instruct_retroagent_grpo_webshop/memory_cache/grpo_webshop_qwen25_7b_instruct_intrinsic_rewards_pairwise_mem_potential_grounding_credit_assignment_mem_half_group_retrieval_sim_utility_ucb_large_invalid_penality_gen_length2048/webshop_reflections.json"
export REFLECTION_ANALYSIS_FILE="./xiaoyingzhang/agent_training/qwen25_7b_instruct_retroagent_grpo_webshop/memory_cache/grpo_webshop_qwen25_7b_instruct_intrinsic_rewards_pairwise_mem_potential_grounding_credit_assignment_mem_half_group_retrieval_sim_utility_ucb_large_invalid_penality_gen_length2048/reflect_analysis.jsonl"
export TRAJECTORY_ANALYSIS_FILE="./xiaoyingzhang/agent_training/qwen25_7b_instruct_retroagent_grpo_webshop/memory_cache/grpo_webshop_qwen25_7b_instruct_intrinsic_rewards_pairwise_mem_potential_grounding_credit_assignment_mem_half_group_retrieval_sim_utility_ucb_large_invalid_penality_gen_length2048/trajectory_analysis.jsonl"
mkdir -p $(dirname "$REFLECTION_FILE")
# --- FIX END ---# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size
    
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=16384 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.reflect_log_path=$REFLECTION_ANALYSIS_FILE \
    +data.trajectory_log_path=$TRAJECTORY_ANALYSIS_FILE \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    +actor_rollout_ref.actor.tis_imp_ratio_cap=-1 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    +algorithm.intrinsic_reward_coefficient=1.0 \
    +algorithm.intrinsic_hard_cutoff=False \
    +algorithm.reflection_reference_policy=False \
    +algorithm.credit_assignment=True \
    env.env_name=Webshop \
    env.reflection_memory.top_k=3 \
    +env.reflection_memory.filepath=$REFLECTION_FILE \
    env.reflection_memory.beta=0.05 \
    +env.reflection_memory.enable_memory=True \
    +env.reflection_memory.retrieve_mode=both \
    +env.reflection_memory.memory_start_cutoff=0.0 \
    +env.reflection_memory.reflection_decay=False \
    +env.reflection_memory.potential_based_on_binary_success=False \
    +env.reflection_memory.group_relative_intrinsic_rewards=False \
    +env.train_retrieve_type=ucb \
    +env.eval_retrieve_type=greedy \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
