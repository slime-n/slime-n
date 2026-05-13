"""PolicyConfig dataclass + flat-fields parser for multi-policy training.

Parses examples/<run>/config.yaml of the form:

    policies:
      - name: solver
        role: actor
        hf_checkpoint: ...
        load: ...
        save: ...
        ref_load: ...                   # optional
        buffer_mode: split | shared
        num_gpus_per_node: 8
        megatron_num_nodes: 1
        sglang_num_nodes: 1
        megatron:                       # nested: training fields
          tensor_model_parallel_size: ...
          ...
        sglang:                         # nested: engine deployment fields
          model_path: ...               # defaults to top-level hf_checkpoint
          server_groups: [...]
          ...

Parser flattens megatron[*] into top-level PolicyConfig fields, keeps sglang as
a raw dict (later projected to a SglangConfig.ModelConfig). 1:1 pairing is
implicit: PolicyConfig.sglang_server defaults to PolicyConfig.name.

Diverges from upstream parse_megatron_role_args (commit f65c6e8): no overrides:
sub-dict, no silent critic field forcing. Validation only.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PolicyConfig:
    # ── identity ──
    name: str
    role: str = "actor"  # only "actor" supported in v1; critic deferred

    # ── model / checkpoint ──
    hf_checkpoint: str = ""
    load: str | None = None
    save: str | None = None
    ref_load: str | None = None  # KL ref; with_ref=True when kl_coef != 0 or use_kl_loss=True

    # ── Megatron parallel ──
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    sequence_parallel: bool = False

    # ── Megatron numerical / dropout (RL-correctness defaults) ──
    # Megatron defaults attention_dropout / hidden_dropout to 0.1; RL training
    # needs deterministic forward (rollout-time and train-time log probs must
    # match) so we default these to 0 for any policy.
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True

    # ── Memory: chunked log-prob computation ──
    # >0 chunks the [T, V] logits tensor along T to reduce forward peak; -1
    # disables chunking. Critical for vocab-heavy models on tight VRAM (e.g.
    # Qwen3 with 152K vocab on L40S).
    log_probs_chunk_size: int = -1

    # ── Recompute (memory/compute trade) ──
    recompute_granularity: str | None = None  # "full" | "selective" | None
    recompute_method: str | None = None  # "uniform" | "block" | None
    recompute_num_layers: int | None = None

    # ── Weight-load mode ──
    # "raw" (default): expects a Megatron torch_dist checkpoint at `load`.
    # "bridge": when `load` is unset or stale, slime falls back to loading from
    # `hf_checkpoint` via mbridge. Useful for first-time runs from an HF model.
    megatron_to_hf_mode: str = "raw"

    # ── Batching ──
    micro_batch_size: int = 1
    global_batch_size: int = 64
    use_dynamic_batch_size: bool = False
    max_tokens_per_gpu: int | None = None
    log_probs_max_tokens_per_gpu: int | None = None  # falls back to max_tokens_per_gpu

    # ── Optimizer (per-policy — each actor has its own optimizer state) ──
    optimizer: str = "adam"
    lr: float = 1.0e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    optimizer_cpu_offload: bool = False
    overlap_cpu_optimizer_d2h_h2d: bool = False
    use_precision_aware_optimizer: bool = False

    # ── Loss / RL hyperparameters ──
    eps_clip: float = 0.2
    eps_clip_high: float | None = None
    kl_coef: float = 0.0
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    use_kl_loss: bool = False
    custom_advantage_function_path: str | None = None
    advantage_estimator: str = "grpo"
    n_samples_per_prompt: int = 1
    rewards_normalization: bool = True
    grpo_std_normalization: bool = True

    # ── Multi-policy orchestration ──
    sglang_server: str | None = None  # 1:1 pairing; defaults to PolicyConfig.name
    buffer_mode: str = "split"  # "split" | "shared"
    # trainable=False marks a frozen producer (e.g., OPD Megatron teacher):
    # the train loop runs its forward-only train() and merges the returned
    # dict into trainable consumers' external_data. Frozen actors skip
    # weights_backuper / weight_updater / optimizer-state save / weight push.
    trainable: bool = True

    # ── On-policy distillation (per-policy; legacy was CLI-global) ──
    # use_opd / opd_type / opd_kl_coef live on the trainable consumer (the
    # student). For the new external-data path, set opd_type="megatron_actor"
    # — this skips the legacy in-process tag teacher path at
    # placement_group.py:159 (which fires only when opd_type == "megatron").
    # opd_teacher_load is kept for the legacy in-process path only.
    use_opd: bool = False
    opd_type: str | None = None
    opd_kl_coef: float = 1.0
    opd_teacher_load: str | None = None

    # ── GPU placement (cluster-level num_gpus_per_node, per-side node counts) ──
    num_gpus_per_node: int = 8
    megatron_num_nodes: int = 1
    sglang_num_nodes: int = 1

    # ── sglang sub-block (raw dict, projected to SglangConfig.ModelConfig later) ──
    sglang: dict | None = None

    # ── Predicates used by the multi-policy framework ──
    # These exist because "trainable" and "has SGLang engine" used to coincide
    # (every trainable policy had a paired engine), but the standalone-SGLang
    # teacher shape (m✗ s✓ trainable=false) breaks that equivalence. Filters
    # that gate engine-related decisions should use has_engine(); allocation
    # of a Megatron training actor should use has_megatron().
    def has_engine(self) -> bool:
        return self.sglang_num_nodes > 0 and self.sglang is not None

    def has_megatron(self) -> bool:
        return self.megatron_num_nodes > 0


def validate_policy_config(cfg: PolicyConfig) -> None:
    if cfg.role != "actor":
        raise ValueError(
            f"{cfg.name}: only role='actor' supported, got {cfg.role!r} " f"(critic deferred — PPO out of scope)"
        )
    if cfg.sglang_server is None:
        raise ValueError(f"{cfg.name}: actor requires sglang_server (1:1 pairing)")
    if cfg.buffer_mode not in {"shared", "split"}:
        raise ValueError(f"{cfg.name}: buffer_mode must be 'shared' or 'split'")
    if not cfg.hf_checkpoint:
        raise ValueError(f"{cfg.name}: hf_checkpoint required")

    # Sglang placement consistency. Applies to every policy that hosts an
    # engine — trainable paired pipelines AND frozen standalone engines
    # (e.g. OPD SGLang teacher, judge / RM). Megatron-only producers
    # (m✓ s✗) are skipped because they don't run an engine at all.
    if cfg.has_engine():
        sglang_total = cfg.sglang_num_nodes * cfg.num_gpus_per_node
        groups_total = sum(g["num_gpus"] for g in cfg.sglang.get("server_groups", []))
        if sglang_total != groups_total:
            raise ValueError(
                f"{cfg.name}: sglang_num_nodes × num_gpus_per_node ({sglang_total}) "
                f"must equal sum of sglang.server_groups[].num_gpus ({groups_total})"
            )


def parse_policy_configs(config_path: str) -> list[PolicyConfig]:
    """Parse config.yaml into list[PolicyConfig].

    Each policies[i] entry has top-level fields and two nested sub-blocks (megatron, sglang).
    Parser flattens megatron[*] into PolicyConfig fields; sglang stays as a raw dict
    (build_sglang_config_from_policies projects it to a SglangConfig later).
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    if "policies" not in raw or not isinstance(raw["policies"], list):
        raise ValueError("config must have top-level 'policies' list")

    configs: list[PolicyConfig] = []
    for entry in raw["policies"]:
        # sglang.model_path defaults to top-level hf_checkpoint
        sglang_block = dict(entry.get("sglang") or {})
        sglang_block.setdefault("model_path", entry.get("hf_checkpoint"))

        flat: dict[str, Any] = {
            "name": entry["name"],
            "role": entry.get("role", "actor"),
            "trainable": entry.get("trainable", True),
            "hf_checkpoint": entry.get("hf_checkpoint", ""),
            "load": entry.get("load"),
            "save": entry.get("save"),
            "ref_load": entry.get("ref_load"),
            "buffer_mode": entry.get("buffer_mode", "split"),
            "num_gpus_per_node": entry.get("num_gpus_per_node", 8),
            "megatron_num_nodes": entry.get("megatron_num_nodes", 1),
            "sglang_num_nodes": entry.get("sglang_num_nodes", 1),
            **(entry.get("megatron") or {}),
            "sglang": sglang_block,
            "sglang_server": entry["name"],  # 1:1 pairing: server name = policy name
        }
        configs.append(PolicyConfig(**flat))

    for cfg in configs:
        validate_policy_config(cfg)
    _validate_unique_names(configs)
    _validate_unique_sglang_servers(configs)
    _validate_shared_buffer_consistency(configs)
    return configs


def _validate_unique_names(configs: list[PolicyConfig]) -> None:
    names = [c.name for c in configs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate policy names: {names}")


def _validate_unique_sglang_servers(configs: list[PolicyConfig]) -> None:
    """1:1 pairing: each engine has exactly one trainable owner."""
    servers = [c.sglang_server for c in configs]
    if len(servers) != len(set(servers)):
        raise ValueError(f"two policies cannot push to the same sglang_server: {servers}")


def _validate_shared_buffer_consistency(configs: list[PolicyConfig]) -> None:
    """Shared-buffer policies must agree on advantage_estimator and n_samples_per_prompt
    (group-norm depends on these)."""
    shared = [c for c in configs if c.buffer_mode == "shared"]
    if len(shared) <= 1:
        return
    estimators = {c.advantage_estimator for c in shared}
    if len(estimators) > 1:
        raise ValueError(
            f"shared-buffer policies must agree on advantage_estimator: "
            f"{ {c.name: c.advantage_estimator for c in shared} }"
        )
    nsp = {c.n_samples_per_prompt for c in shared}
    if len(nsp) > 1:
        raise ValueError(
            f"shared-buffer policies must agree on n_samples_per_prompt: "
            f"{ {c.name: c.n_samples_per_prompt for c in shared} }"
        )


def build_sglang_config_from_policies(configs: list[PolicyConfig]):
    """Project each policy's sglang sub-block into a SglangConfig.ModelConfig.

    The user-facing sglang sub-block is FLAT (no overrides: indirection).
    Model-level fields (model_path, num_gpus_per_engine, update_weights, server_groups)
    map directly onto ModelConfig. All other fields are sglang ServerArgs and get
    folded into each ServerGroupConfig.overrides for upstream's _compute_server_args.
    """
    from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig

    MODEL_FIELDS = {"model_path", "num_gpus_per_engine", "update_weights", "server_groups"}

    models = []
    for cfg in configs:
        # Detect inconsistent shape: sglang_num_nodes asks for engine GPUs but
        # no sglang block was provided. Caught early here so the user gets a
        # clear error instead of a silent skip.
        if cfg.sglang is None and cfg.sglang_num_nodes > 0:
            raise ValueError(f"{cfg.name}: missing 'sglang' sub-block in config")
        # Skip policies without an engine: Megatron-only producers
        # (m✓ s✗ frozen teacher / critic). Engine-hosting policies — both
        # trainable paired (m✓ s✓) and frozen standalone (m✗ s✓ judge / RM /
        # OPD SGLang teacher) — fall through to build a ModelConfig.
        if not cfg.has_engine():
            continue
        sg = dict(cfg.sglang)

        # Server-args (everything except model-level fields) flow into each group's overrides.
        server_arg_overrides = {k: v for k, v in sg.items() if k not in MODEL_FIELDS}

        groups = []
        for g in sg.get("server_groups", []):
            g = dict(g)
            # Per-group overrides win over model-level server-args.
            merged_overrides = {**server_arg_overrides, **g.pop("overrides", {})}
            groups.append(ServerGroupConfig(**g, overrides=merged_overrides))

        models.append(
            ModelConfig(
                name=cfg.name,  # 1:1 pairing: server name = policy name
                model_path=sg.get("model_path"),
                num_gpus_per_engine=sg.get("num_gpus_per_engine"),
                update_weights=sg.get("update_weights", True),
                server_groups=groups,
            )
        )
    return SglangConfig(models=models)


def derive_cluster_sizing(configs: list[PolicyConfig], colocate: bool) -> tuple[int, int, int]:
    """Compute (actor_gpus, rollout_gpus, total_gpus) from PolicyConfig list.

    Used by:
      - launcher (to set ray start --num-gpus)
      - create_placement_groups_multi (to size the global PG)
    """
    actor_gpus = sum(c.megatron_num_nodes * c.num_gpus_per_node for c in configs)
    # Engine-hosting policies contribute to the rollout-side budget regardless
    # of trainability: both trainable paired pipelines (m✓ s✓) and frozen
    # standalone engines (m✗ s✓ judge / RM / OPD SGLang teacher) need GPUs.
    # Megatron-only frozen producers (m✓ s✗) have no engine and are excluded.
    rollout_gpus = sum(c.sglang_num_nodes * c.num_gpus_per_node for c in configs if c.has_engine())
    total_gpus = max(actor_gpus, rollout_gpus) if colocate else actor_gpus + rollout_gpus
    return actor_gpus, rollout_gpus, total_gpus


@dataclasses.dataclass
class PolicyHandle:
    """One trainable Megatron actor + its 1:1-paired sglang engine handle.

    Built by create_training_models_multi (slime.ray.placement_group). The driver
    iterates a dict[name, PolicyHandle] returned from there.
    """

    config: PolicyConfig
    args: Any  # PolicyConfig projected onto a Namespace for downstream Megatron code
    train_group: Any  # RayTrainGroup


def config_to_namespace(cfg: PolicyConfig, base_args):
    """Project PolicyConfig fields onto a Namespace, copying everything directly.

    Pulls non-policy globals (rollout cadence, data paths, perf args) from base_args.
    Sets policy_name = cfg.name so downstream code (update_weights routing,
    Sample.policy_name tagging) can read it from args.

    Pure function — used by create_training_models_multi to build per-policy Namespaces.
    """
    from argparse import Namespace

    ns = Namespace(**vars(base_args))
    for f in dataclasses.fields(cfg):
        setattr(ns, f.name, getattr(cfg, f.name))
    ns.actor_num_nodes = cfg.megatron_num_nodes
    ns.actor_num_gpus_per_node = cfg.num_gpus_per_node
    ns.num_gpus_per_node = cfg.num_gpus_per_node
    ns.world_size = cfg.megatron_num_nodes * cfg.num_gpus_per_node
    ns.policy_name = cfg.name
    # Megatron's tokenizer fallback runs at parse_args time when global
    # hf_checkpoint may still be None. Re-derive per policy now that
    # cfg.hf_checkpoint is known.
    if getattr(ns, "tokenizer_model", None) is None:
        ns.tokenizer_model = cfg.hf_checkpoint
        if not getattr(ns, "tokenizer_type", None):
            ns.tokenizer_type = "HuggingFaceTokenizer"

    # Mirror slime_validate_args' per-policy-affecting defaults. Upstream
    # runs these once on global args at parse_args time; per-policy actors
    # only inherit values from their own cfg, so we re-apply the same logic
    # here. (Bugs of the form "policy A trains fine because eps_clip_high=0.28
    # was on the actor's args, but policy B crashes because its YAML omitted
    # eps_clip_high and ns.eps_clip_high stayed None.")
    if getattr(ns, "eps_clip_high", None) is None:
        ns.eps_clip_high = ns.eps_clip
    # n_samples_per_prompt == 1 means the GRPO group has size 1; the std is
    # zero and the group-norm divide produces NaN. Upstream forces the flag
    # off in that case.
    if getattr(ns, "n_samples_per_prompt", 1) == 1:
        ns.grpo_std_normalization = False
    # When dynamic batching is on, the log-probs forward pass uses the same
    # token budget as the train forward unless the user overrides it.
    if getattr(ns, "use_dynamic_batch_size", False):
        if getattr(ns, "log_probs_max_tokens_per_gpu", None) is None:
            ns.log_probs_max_tokens_per_gpu = ns.max_tokens_per_gpu

    # Re-run slime_validate_args' load resolution per policy. Upstream slime
    # runs this once at parse_args time against the GLOBAL args namespace, but
    # the globals don't see per-policy values (megatron_to_hf_mode / ref_load /
    # load all live in each policy's YAML entry). So we apply the exact same
    # logic — character-for-character — to the per-policy ns. No multi-policy-
    # specific shortcuts: behavior matches upstream when run with a single
    # policy, and each per-policy actor lands at the same args state it would
    # have reached through legacy CLI flags.
    import os

    if ns.megatron_to_hf_mode == "bridge":
        if (
            ns.load is not None
            and os.path.exists(ns.load)
            and os.path.exists(os.path.join(ns.load, "latest_checkpointed_iteration.txt"))
        ):
            # Megatron torch_dist ckpt at ns.load → resume from it; mbridge
            # only used in model_provider for the architecture spec.
            pass
        else:
            if ns.load is None:
                ns.load = ns.ref_load or cfg.hf_checkpoint
            # HF/bridge load → start at rollout 0
            ns.start_rollout_id = 0
    else:
        if (
            ns.load is None
            or not os.path.exists(ns.load)
            or not os.path.exists(os.path.join(ns.load, "latest_checkpointed_iteration.txt"))
        ):
            ns.no_load_optim = True
            ns.no_load_rng = True
            ns.finetune = True
            ns.load = ns.ref_load
            if getattr(ns, "ref_ckpt_step", None) is not None:
                ns.ckpt_step = ns.ref_ckpt_step
            ns.start_rollout_id = 0
    return ns


def derive_policy_slices(configs: list[PolicyConfig], total_idxs: list[int], colocate: bool) -> dict[str, list[int]]:
    """Carve global placement-group indices into per-policy actor slices + a rollout slice.

    Pure function — no Ray. Mirrors the carving logic of create_placement_groups_multi
    (Step 7 in plan.md) so it can be unit-tested without a real placement group.

    Returns dict[name, list[int]] where:
      result[<policy_name>] = idxs assigned to that policy's Megatron actor
      result["rollout"]     = idxs for the rollout (shared with actors when colocate)
    """
    actor_gpus, rollout_gpus, total = derive_cluster_sizing(configs, colocate=colocate)
    if len(total_idxs) != total:
        raise ValueError(f"total_idxs has {len(total_idxs)} elements, derive_cluster_sizing wants {total}")

    cursor = 0
    result: dict[str, list[int]] = {}
    for c in configs:
        ss = c.megatron_num_nodes * c.num_gpus_per_node
        result[c.name] = list(total_idxs[cursor : cursor + ss])
        cursor += ss

    if colocate:
        result["rollout"] = list(total_idxs)
    else:
        result["rollout"] = list(total_idxs[cursor:])
    return result
