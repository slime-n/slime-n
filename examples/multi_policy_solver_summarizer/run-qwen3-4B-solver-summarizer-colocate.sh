#!/bin/bash

# Colocate variant of run-qwen3-4B-solver-summarizer-nocolocate.sh: `--colocate`
# timeshares each policy's Megatron actor and SGLang engines on the same GPUs
# (offload/onload between train and rollout). Sizing and H200 tuning are derived
# from config-colocate.yaml (8x H200: 4 GPUs/policy).

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Qwen3-4B-Thinking-2507 shares the Qwen3-4B arch but uses rope_theta 5e6.
MODEL_ARGS_ROTARY_BASE=5000000 source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh"

# Per-policy training fields live in config.yaml; run-level orchestration is CLI.

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.multi_policy_solver_summarizer.rollout_with_multi_agents.generate_with_multi_agents
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --disable-rollout-trim-samples
   --rollout-max-context-len 32768
   --rollout-max-response-len 32768
   --rollout-temperature 1
   --balance-data
)

NUM_GPUS=8

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/config-colocate.yaml"
   --save-interval 5
   # Dumps land under <dump-details>/<policy_name>/{rollout,train,packed}_data/.
   --dump-details /tmp/multi_policy_solver_summarizer/dump_details_colocate
)
# Note: --colocate auto-enables offload_train + offload_rollout.

EVAL_ARGS=(
   # AIME-2024 eval (eval_config.yaml) with a custom per-role eval function.
   # --log-passrate intentionally off: it would trip a train-side group_size
   # assertion that the chain's num_parallel samples per call don't satisfy.
   --eval-interval 2
   --eval-config "${SCRIPT_DIR}/eval_config.yaml"
   --eval-function-path examples.multi_policy_solver_summarizer.eval_fn.eval_with_multi_agents
   --eval-max-response-len 32768
   --eval-top-p 1
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-4B-Thinking-2507-solver-summarizer-colocate
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here —
# torch_memory_saver (sglang's colocate pause/resume) won't init with it on.

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_multi_policy.py \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${TRAIN_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]}
