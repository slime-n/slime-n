import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor_group import RayTrainGroup
from .rollout import RolloutManager

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    num_gpus = 0
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0
    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]

    result = {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }

    result["critic"] = result["actor"] if args.use_critic else None

    return result


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, role="actor"):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        role=role,
    )


def create_training_models(args, pgs, rollout_manager):
    actor_args = args
    if args.megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")

    actor_model = allocate_train_group(
        args=actor_args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
    )

    critic_model = None
    if args.use_critic:
        from slime.utils.arguments import parse_megatron_role_args

        critic_args = (
            parse_megatron_role_args(args, args.megatron_config_path, role="critic")
            if args.megatron_config_path is not None
            else args
        )
        critic_model = allocate_train_group(
            args=critic_args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
        )
        critic_start_rollout_ids = ray.get(critic_model.async_init(critic_model.args, role="critic", with_ref=False))

    actor_start_rollout_ids = ray.get(
        actor_model.async_init(
            actor_args,
            role="actor",
            with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
            with_opd_teacher=actor_args.use_opd and actor_args.opd_type == "megatron",
        )
    )
    # TODO how to decide rollout start id when critic is involved? For now we just require user to specify it via args.
    if args.use_critic:
        start_rollout_ids = critic_start_rollout_ids
    else:
        start_rollout_ids = actor_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1

    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    actor_model.set_rollout_manager(rollout_manager)
    if args.use_critic:
        critic_model.set_rollout_manager(rollout_manager)

    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    rollout_manager = RolloutManager.options(
        num_cpus=1,
        num_gpus=0,
    ).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch


# ════════════════════════════════════════════════════════════════════════════
# Multi-policy additions (Step 7 — multi-actor placement groups)
#
# Pure additions — the existing create_placement_groups / create_rollout_manager
# above are unchanged. The legacy single-policy train.py path stays bit-for-bit
# identical.
#
# create_rollout_manager_multi is a stub that points at Step 4 (RolloutManager
# changes) where the real implementation lands. Step 7 ships placement groups
# only.
# ════════════════════════════════════════════════════════════════════════════


def create_placement_groups_multi(args, policy_configs):
    """Carve a single global placement group into per-policy actor slices + a
    rollout slice. Pure additive entry point used by train_multi_policy.py.

    Cluster sizes derived from policy_configs (config.yaml):
      actor_gpus   = sum(c.megatron_num_nodes * c.num_gpus_per_node for c in cfgs)
      rollout_gpus = sum(c.sglang_num_nodes   * c.num_gpus_per_node for c in cfgs)
      total = max(actor_gpus, rollout_gpus)  if args.colocate
            = actor_gpus + rollout_gpus      otherwise

    Returns dict keyed by policy name + "rollout":
      result["<policy_name>"] = (pg, bundle_indices, gpu_ids)  for that policy's actor
      result["rollout"]       = (pg, bundle_indices, gpu_ids)  for sglang engines

    With --colocate, the rollout slice is the entire pool (sglang shares GPUs
    with actors via fractional Ray resources). Without --colocate, rollout gets
    the contiguous range right after the last actor slice.
    """
    actor_gpus = sum(c.megatron_num_nodes * c.num_gpus_per_node for c in policy_configs)
    rollout_gpus = sum(c.sglang_num_nodes * c.num_gpus_per_node for c in policy_configs)

    if args.colocate:
        total = max(actor_gpus, rollout_gpus)
    else:
        total = actor_gpus + rollout_gpus

    logger.info(
        f"create_placement_groups_multi: {len(policy_configs)} policies, "
        f"actor_gpus={actor_gpus}, rollout_gpus={rollout_gpus}, "
        f"total={total}, colocate={args.colocate}"
    )

    pg, idxs, gpus = _create_placement_group(total)

    cursor = 0
    result: dict[str, tuple] = {}
    for c in policy_configs:
        ss = c.megatron_num_nodes * c.num_gpus_per_node
        result[c.name] = (
            pg,
            list(idxs[cursor : cursor + ss]),
            list(gpus[cursor : cursor + ss]),
        )
        cursor += ss

    if args.colocate:
        result["rollout"] = (pg, list(idxs), list(gpus))
    else:
        result["rollout"] = (pg, list(idxs[cursor:]), list(gpus[cursor:]))

    return result


def create_training_models_multi(args, pgs, rollout_manager, policy_configs):
    """Multi-policy version of create_training_models. Mirrors the legacy
    function's responsibilities for N independent trainable Megatron actors.

    For each policy:
      - project PolicyConfig onto a Namespace (config_to_namespace)
      - allocate the RayTrainGroup
      - register the policy with rollout_manager (1:1 sglang server binding)
      - async_init the train group, collect start_rollout_ids
      - set_rollout_manager so the train group can post weight updates

    Then reconciles start_rollout_ids across policies (min + warning on
    divergence), sets args.start_rollout_id, and loads any resumed rollout
    buffer (rollout_global_dataset parity with legacy create_training_models).

    Returns dict[name, PolicyHandle].
    """
    from slime.utils.policy_config import PolicyHandle, config_to_namespace

    # ── Build N RayTrainGroups + register each with the rollout manager ──
    handles: dict[str, PolicyHandle] = {}
    actor_gpu_offset = 0
    for cfg in policy_configs:
        args_p = config_to_namespace(cfg, args)
        args_p.actor_gpu_offset = actor_gpu_offset
        actor_gpu_offset += cfg.megatron_num_nodes * cfg.num_gpus_per_node
        # Allocate a RayTrainGroup only for policies that have a Megatron actor.
        # For engine-only policies (m✗ s✓ frozen standalone SGLang engine),
        # RayTrainGroup(num_nodes=0) would spawn zero workers and downstream
        # async_init returns [] — crashing the start_rollout_id reconciliation.
        if cfg.has_megatron():
            train_group = allocate_train_group(
                args=args_p,
                num_nodes=cfg.megatron_num_nodes,
                num_gpus_per_node=cfg.num_gpus_per_node,
                pg=pgs[cfg.name],
                role=cfg.role,
            )
        else:
            train_group = None
        handles[cfg.name] = PolicyHandle(config=cfg, args=args_p, train_group=train_group)
        # Register every engine-hosting policy with the rollout manager so it
        # spins up an SGLang server and routes inference requests to it. This
        # includes trainable paired pipelines (m✓ s✓) and frozen standalone
        # engines (m✗ s✓ judge / RM / OPD SGLang teacher). Megatron-only frozen
        # producers (m✓ s✗) skip registration — they have no engine, and
        # _get_server would fail. Weight-push remains gated on cfg.trainable in
        # train_multi_policy.py so frozen engines never get a weight broadcast.
        if cfg.has_engine():
            ray.get(rollout_manager.register_policy.remote(cfg.name, cfg.sglang_server, args_p))

    # ── async_init each + reconcile start_rollout_ids across policies ──
    # async_init kwargs mirror legacy create_training_models so OPD-megatron
    # and ref-model toggles work per-policy via PolicyConfig.overrides. Engine-
    # only policies (train_group is None) have no actor to initialize and don't
    # carry a start_rollout_id.
    starts: dict[str, int] = {}
    for name, h in handles.items():
        if h.train_group is None:
            continue
        ids = ray.get(
            h.train_group.async_init(
                h.args,
                role=h.config.role,
                with_ref=h.args.kl_coef != 0 or h.args.use_kl_loss,
                with_opd_teacher=getattr(h.args, "use_opd", False) and getattr(h.args, "opd_type", None) == "megatron",
            )
        )
        if len(set(ids)) != 1:
            raise RuntimeError(f"{name}: workers disagree on start_rollout_id: {ids}")
        starts[name] = ids[0]
        h.train_group.set_rollout_manager(rollout_manager)

    chosen = min(starts.values())
    if any(s != chosen for s in starts.values()):
        logger.warning(
            f"start_rollout_ids diverged across policies: {starts}; using min={chosen}. "
            f"User is responsible for ensuring this is intended."
        )
    if args.start_rollout_id is None:
        args.start_rollout_id = chosen

    # Legacy parity: load resumed rollout buffer when --rollout-global-dataset is set
    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return handles


def create_rollout_manager_multi(args, pg, sglang_config):
    """Multi-policy rollout manager. Thin bridge: serializes the per-policy
    SglangConfig back to a temp YAML, sets args.sglang_config to that path, and
    delegates to the existing single-policy create_rollout_manager.

    This keeps slime/ray/rollout.py:_resolve_sglang_config completely untouched —
    we go through the same code path as a manually-written --sglang-config YAML.

    The driver (train_multi_policy.py) sets args.rollout_num_gpus from
    derive_cluster_sizing before calling here, so the assertion at
    _resolve_sglang_config (rollout.py:1126) `actual == expected` is satisfied:
      actual   = sglang_config.total_num_gpus = sum(server_groups[].num_gpus)
      expected = args.rollout_num_gpus        = sum(sglang_num_nodes × num_gpus_per_node)
    These are equal by config.yaml's validate_policy_config (sglang placement
    consistency check).

    Per-policy registration (RolloutManager.register_policy) is NOT done here —
    that's create_training_models_multi's job (it has the full per-policy
    Namespace). Calling register_policy here would double-register and trigger
    the "already bound" check.
    """
    import tempfile

    import yaml

    # Project SglangConfig back to the YAML schema upstream expects.
    yaml_dict: dict = {"sglang": []}
    for m in sglang_config.models:
        entry: dict = {"name": m.name, "update_weights": m.update_weights, "server_groups": []}
        if m.model_path is not None:
            entry["model_path"] = m.model_path
        if m.num_gpus_per_engine is not None:
            entry["num_gpus_per_engine"] = m.num_gpus_per_engine
        for g in m.server_groups:
            gd: dict = {"worker_type": g.worker_type, "num_gpus": g.num_gpus}
            if g.num_gpus_per_engine is not None:
                gd["num_gpus_per_engine"] = g.num_gpus_per_engine
            if g.overrides:
                gd["overrides"] = g.overrides
            entry["server_groups"].append(gd)
        yaml_dict["sglang"].append(entry)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_dict, f)
        args.sglang_config = f.name

    return create_rollout_manager(args, pg)
