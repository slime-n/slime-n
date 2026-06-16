# Multi-Policy Solver + Summarizer

Two trainable paired policies cooperate on math problems (DAPO-math-17k). The **solver** generates N candidate solutions in parallel; the **summarizer** then sees ALL N candidates and synthesizes one final answer in the standard `Answer: \boxed{...}` format. Both policies receive direct correctness rewards on their own completions.

| schema | slime<sup>n</sup> |
|:---:|:---:|
| ![solver+summarizer schema](./imgs/schema.png) | ![solver+summarizer framework](./imgs/arch.png) |

*Left: the solver produces N candidates, the summarizer sees all N and emits one final `\boxed{...}` answer. Right: two trainable pairs (Megatron + SGLang); the chain is owned by the custom rollout function, and each policy has its own optimizer, buffer, and RLVR reward.*

## Files

* `config-nocolocate.yaml` / `config-colocate.yaml`: solver + summarizer policy schema (both trainable, paired with their own SGLang engines). The no-colocate config spreads rollout across 8 GPUs (3 sglang engines per policy); the colocate config fits 2 GPUs.
* `eval_config.yaml`: AIME-2024 eval-dataset config (rm_type, n_samples). Consumed via `--eval-config`.
* `run-qwen3-4B-solver-summarizer-nocolocate.sh` / `-colocate.sh`: launch scripts (ray start + train_multi_policy.py).
* `agent_system.py`: per-prompt rollout orchestration (solver → summarizer dispatch).
* `rollout_with_multi_agents.py`: top-level multi-agent rollout entrypoint.
* `eval_fn.py`: custom eval function — aggregates chain samples into summarizer / solver-mean / solver-max metrics.
* `prompts.py`: solver / summarizer prompt templates.

## Quick Start

```bash
cd slime-n
bash examples/multi_policy_solver_summarizer/run-qwen3-4B-solver-summarizer-nocolocate.sh
```

## How It Works

* Pipeline (N=4 trajectories per prompt): N solvers run in parallel → N summarizers that each see all N solver candidates and synthesize a final answer.
* Reward: solver gets RLVR correctness on its own response; summarizer gets correctness on its synthesized answer (graded directly, no index lookup). Group shaping multiplies both roles by 1.2 if mean parsed-summarizer reward > 0.5, else by 0.8. If the summarizer phase fails entirely, raw rewards are preserved (anti-train guard).
* Each policy has its own buffer (`buffer_mode: split`), routed by `Sample.policy_name`. `n_samples_per_prompt = num_parallel = 4` for GRPO group-norm.


## Eval

Every `--eval-interval` rollouts, the full chain runs on AIME-2024 (30 prompts; 30 × 8 = 240 generations per eval). The custom eval function (`eval_fn.eval_with_multi_agents`, wired via `--eval-function-path`) computes the unbiased pass@k estimator per prompt from the **raw** RM rewards (unscaled by the 0.8/1.2 training weights) and emits, for k ∈ {1, 2, 4}:

* `eval/aime_summarizer_pass{1,2,4}/score` — summarizer pass@k.
* `eval/aime_solver_pass{1,2,4}/score` — solver pass@k.

Pass@k is computed in the eval function itself (not via `--log-passrate`) because that global flag would also trigger train-side pass-rate logging, whose group-size assertion doesn't match the chain's `num_parallel × n_samples_per_prompt` per-prompt sample count.

Headline: `aime_summarizer_pass4` (best-of-4 final-answer quality) vs `aime_solver_pass4` (skyline ceiling). Their difference diagnoses whether the summarizer is synthesizing nontrivially or just aggregating (or destroying) signal the solver produced.

Two limitations: eval generation flows through the first-listed policy's SGLang engine (the solver's), and metrics are split by role name, not by per-policy namespace.


## Results

Reference curves below are from the original 1873-step run on Qwen3-0.6B
(the example previously shipped at that size). The recipe is unchanged;
only the model was swapped to Qwen3-4B.

**Per-role raw reward** — both policies trend up; summarizer mean
~0.54, solver mean ~0.43. The summarizer benefits from seeing all N
solver candidates so its peak (~0.78) exceeds the solver's (~0.65).

![reward](imgs/reward.png)
