"""Level-1 tests for the multi-policy config layer.

Pure-Python: no Ray, no GPUs, no slime internals beyond SglangConfig (which is
upstream and self-contained). Run with:

    python -m pytest tests/test_multi_policy_config.py -v

or directly:

    python tests/test_multi_policy_config.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile

import pytest
import yaml

# Make sure the worktree's slime/ is importable when running directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from slime.utils.policy_config import (
    PolicyConfig,
    _validate_shared_buffer_consistency,
    _validate_unique_names,
    _validate_unique_sglang_servers,
    build_sglang_config_from_policies,
    derive_cluster_sizing,
    derive_policy_slices,
    parse_policy_configs,
    validate_policy_config,
)


EXAMPLE_CONFIG = os.path.join(_REPO_ROOT, "examples", "multi_policy_multi_agent", "config.yaml")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _minimal_actor(**overrides) -> PolicyConfig:
    """Build a minimal valid PolicyConfig; tests override specific fields."""
    base = dict(
        name="solver",
        role="actor",
        hf_checkpoint="/x",
        sglang_server="solver",
        buffer_mode="split",
        num_gpus_per_node=8,
        megatron_num_nodes=1,
        sglang_num_nodes=1,
        sglang={
            "update_weights": True,
            "num_gpus_per_engine": 8,
            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
        },
    )
    base.update(overrides)
    return PolicyConfig(**base)


def _write_yaml(data: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(data, f)
    return path


# ────────────────────────────────────────────────────────────────────────────
# Parser end-to-end on the actual example
# ────────────────────────────────────────────────────────────────────────────


class TestExampleConfig:
    def test_parses_three_policies(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        assert [c.name for c in cfgs] == ["solver", "rewriter", "selector"]

    def test_each_is_actor(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        assert all(c.role == "actor" for c in cfgs)

    def test_megatron_fields_flattened(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        solver = cfgs[0]
        assert solver.tensor_model_parallel_size == 1
        assert solver.expert_model_parallel_size == 1
        assert solver.lr == 1.0e-6
        assert solver.optimizer_cpu_offload is True
        assert solver.advantage_estimator == "grpo"
        assert solver.n_samples_per_prompt == 4

    def test_sglang_kept_as_dict_with_default_model_path(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        for c in cfgs:
            assert isinstance(c.sglang, dict)
            assert c.sglang["model_path"] == c.hf_checkpoint
            assert c.sglang["update_weights"] is True

    def test_sglang_server_defaults_to_policy_name(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        for c in cfgs:
            assert c.sglang_server == c.name

    def test_buffer_mode_split(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        assert all(c.buffer_mode == "split" for c in cfgs)

    def test_placement_fields(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        for c in cfgs:
            assert c.num_gpus_per_node == 1
            assert c.megatron_num_nodes == 1
            assert c.sglang_num_nodes == 1


# ────────────────────────────────────────────────────────────────────────────
# Per-entry validator
# ────────────────────────────────────────────────────────────────────────────


class TestValidatePolicyConfig:
    def test_minimal_actor_passes(self):
        validate_policy_config(_minimal_actor())  # no raise

    def test_critic_role_rejected(self):
        cfg = _minimal_actor(role="critic")
        with pytest.raises(ValueError, match="only role='actor'"):
            validate_policy_config(cfg)

    def test_unknown_role_rejected(self):
        cfg = _minimal_actor(role="wizard")
        with pytest.raises(ValueError, match="only role='actor'"):
            validate_policy_config(cfg)

    def test_missing_sglang_server_rejected(self):
        cfg = _minimal_actor(sglang_server=None)
        with pytest.raises(ValueError, match="actor requires sglang_server"):
            validate_policy_config(cfg)

    def test_missing_hf_checkpoint_rejected(self):
        cfg = _minimal_actor(hf_checkpoint="")
        with pytest.raises(ValueError, match="hf_checkpoint required"):
            validate_policy_config(cfg)

    def test_bad_buffer_mode_rejected(self):
        cfg = _minimal_actor(buffer_mode="duplex")
        with pytest.raises(ValueError, match="buffer_mode must be"):
            validate_policy_config(cfg)

    def test_sglang_placement_mismatch_rejected(self):
        # 1 node × 8 gpus_per_node = 8, but server_groups sums to 4
        cfg = _minimal_actor(
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 4,
                "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
            }
        )
        with pytest.raises(ValueError, match="must equal sum of sglang.server_groups"):
            validate_policy_config(cfg)

    def test_sglang_placement_match_two_groups(self):
        # 1 node × 8 gpus_per_node = 8 = 4 + 4
        cfg = _minimal_actor(
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 4,
                "server_groups": [
                    {"worker_type": "prefill", "num_gpus": 4},
                    {"worker_type": "decode", "num_gpus": 4},
                ],
            }
        )
        validate_policy_config(cfg)  # no raise


# ────────────────────────────────────────────────────────────────────────────
# Cross-policy validators
# ────────────────────────────────────────────────────────────────────────────


class TestCrossPolicyValidators:
    def test_unique_names(self):
        cfgs = [
            _minimal_actor(name="a", sglang_server="a"),
            _minimal_actor(name="a", sglang_server="b"),  # duplicate name
        ]
        with pytest.raises(ValueError, match="duplicate policy names"):
            _validate_unique_names(cfgs)

    def test_unique_sglang_servers(self):
        cfgs = [
            _minimal_actor(name="a", sglang_server="X"),
            _minimal_actor(name="b", sglang_server="X"),  # collision
        ]
        with pytest.raises(ValueError, match="cannot push to the same"):
            _validate_unique_sglang_servers(cfgs)

    def test_shared_buffer_estimator_must_match(self):
        cfgs = [
            _minimal_actor(name="a", sglang_server="a", buffer_mode="shared", advantage_estimator="grpo"),
            _minimal_actor(name="b", sglang_server="b", buffer_mode="shared", advantage_estimator="gspo"),
        ]
        with pytest.raises(ValueError, match="advantage_estimator"):
            _validate_shared_buffer_consistency(cfgs)

    def test_shared_buffer_n_samples_must_match(self):
        cfgs = [
            _minimal_actor(name="a", sglang_server="a", buffer_mode="shared", n_samples_per_prompt=4),
            _minimal_actor(name="b", sglang_server="b", buffer_mode="shared", n_samples_per_prompt=8),
        ]
        with pytest.raises(ValueError, match="n_samples_per_prompt"):
            _validate_shared_buffer_consistency(cfgs)

    def test_split_buffer_no_constraint(self):
        cfgs = [
            _minimal_actor(
                name="a", sglang_server="a", buffer_mode="split", advantage_estimator="grpo", n_samples_per_prompt=4
            ),
            _minimal_actor(
                name="b", sglang_server="b", buffer_mode="split", advantage_estimator="gspo", n_samples_per_prompt=8
            ),
        ]
        _validate_shared_buffer_consistency(cfgs)  # no raise


# ────────────────────────────────────────────────────────────────────────────
# parse_policy_configs error paths (full-file YAML)
# ────────────────────────────────────────────────────────────────────────────


class TestParserErrors:
    def test_missing_top_level_policies(self):
        path = _write_yaml({"foo": []})
        with pytest.raises(ValueError, match="top-level 'policies' list"):
            parse_policy_configs(path)

    def test_policies_not_list(self):
        path = _write_yaml({"policies": "not a list"})
        with pytest.raises(ValueError, match="top-level 'policies' list"):
            parse_policy_configs(path)

    def test_duplicate_names_at_parse_time(self):
        path = _write_yaml(
            {
                "policies": [
                    {
                        "name": "a",
                        "role": "actor",
                        "hf_checkpoint": "/x",
                        "num_gpus_per_node": 8,
                        "megatron_num_nodes": 1,
                        "sglang_num_nodes": 1,
                        "sglang": {
                            "update_weights": True,
                            "num_gpus_per_engine": 8,
                            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                        },
                    },
                    {
                        "name": "a",
                        "role": "actor",
                        "hf_checkpoint": "/y",
                        "num_gpus_per_node": 8,
                        "megatron_num_nodes": 1,
                        "sglang_num_nodes": 1,
                        "sglang": {
                            "update_weights": True,
                            "num_gpus_per_engine": 8,
                            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                        },
                    },
                ]
            }
        )
        with pytest.raises(ValueError, match="duplicate policy names"):
            parse_policy_configs(path)


# ────────────────────────────────────────────────────────────────────────────
# Cluster sizing
# ────────────────────────────────────────────────────────────────────────────


class TestClusterSizing:
    def test_example_config_sizes(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=True)
        assert (actor, rollout, total) == (3, 3, 3)
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=False)
        assert (actor, rollout, total) == (3, 3, 6)

    def test_split_sizes(self):
        # actor: 1×4 + 2×4 = 12; rollout: 1×4 + 1×4 = 8
        cfgs = [
            _minimal_actor(
                name="a",
                sglang_server="a",
                num_gpus_per_node=4,
                megatron_num_nodes=1,
                sglang_num_nodes=1,
                sglang={
                    "update_weights": True,
                    "num_gpus_per_engine": 4,
                    "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                },
            ),
            _minimal_actor(
                name="b",
                sglang_server="b",
                num_gpus_per_node=4,
                megatron_num_nodes=2,
                sglang_num_nodes=1,
                sglang={
                    "update_weights": True,
                    "num_gpus_per_engine": 4,
                    "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                },
            ),
        ]
        for c in cfgs:
            validate_policy_config(c)
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=True)
        assert actor == 12 and rollout == 8 and total == 12
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=False)
        assert actor == 12 and rollout == 8 and total == 20


# ────────────────────────────────────────────────────────────────────────────
# build_sglang_config_from_policies projection
# ────────────────────────────────────────────────────────────────────────────


class TestBuildSglangConfig:
    def test_three_models_named_after_policies(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sglang_config = build_sglang_config_from_policies(cfgs)
        assert [m.name for m in sglang_config.models] == ["solver", "rewriter", "selector"]

    def test_each_model_is_updatable(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sglang_config = build_sglang_config_from_policies(cfgs)
        assert all(m.update_weights is True for m in sglang_config.models)

    def test_each_model_has_one_server_group_with_1_gpu(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sglang_config = build_sglang_config_from_policies(cfgs)
        for m in sglang_config.models:
            assert len(m.server_groups) == 1
            g = m.server_groups[0]
            assert g.num_gpus == 1
            assert g.worker_type == "regular"

    def test_server_args_folded_into_overrides(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sglang_config = build_sglang_config_from_policies(cfgs)
        for m in sglang_config.models:
            ov = m.server_groups[0].overrides
            # Server-args (mem_fraction_static, chunked_prefill_size, etc.) should be folded in
            assert ov["mem_fraction_static"] == 0.2
            assert ov["chunked_prefill_size"] == 8192
            assert ov["max_running_requests"] == 32
            # attention_backend is intentionally not set in EXAMPLE_CONFIG → sglang's default
            assert "attention_backend" not in ov
            # Model-level fields must NOT have leaked into overrides
            assert "num_gpus_per_engine" not in ov
            assert "update_weights" not in ov
            assert "server_groups" not in ov
            assert "model_path" not in ov

    def test_per_group_overrides_win(self):
        cfg = _minimal_actor(
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 8,
                "mem_fraction_static": 0.7,  # model-level
                "server_groups": [
                    {
                        "worker_type": "regular",
                        "num_gpus": 8,
                        "overrides": {"mem_fraction_static": 0.9},  # per-group override
                    }
                ],
            }
        )
        sglang_config = build_sglang_config_from_policies([cfg])
        assert sglang_config.models[0].server_groups[0].overrides["mem_fraction_static"] == 0.9


# ────────────────────────────────────────────────────────────────────────────
# Placement-slice math (pure-function version of create_placement_groups_multi)
# ────────────────────────────────────────────────────────────────────────────


class TestDerivePolicySlices:
    def test_three_policies_colocate(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        # Colocate: total=3, all three policies share the 3-GPU pool with the rollout
        slices = derive_policy_slices(cfgs, list(range(3)), colocate=True)
        assert slices["solver"] == list(range(0, 1))
        assert slices["rewriter"] == list(range(1, 2))
        assert slices["selector"] == list(range(2, 3))
        assert slices["rollout"] == list(range(3))  # rollout shares the whole pool

    def test_three_policies_no_colocate(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        # No colocate: total=6, actors get 0..2, rollout gets 3..5
        slices = derive_policy_slices(cfgs, list(range(6)), colocate=False)
        assert slices["solver"] == list(range(0, 1))
        assert slices["rewriter"] == list(range(1, 2))
        assert slices["selector"] == list(range(2, 3))
        assert slices["rollout"] == list(range(3, 6))

    def test_disjoint_actor_slices(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        slices = derive_policy_slices(cfgs, list(range(6)), colocate=False)
        actor_idxs = set(slices["solver"]) | set(slices["rewriter"]) | set(slices["selector"])
        assert len(actor_idxs) == 3  # no overlap between actor slices

    def test_two_policies_disjoint(self):
        cfgs = [
            _minimal_actor(
                name="a",
                sglang_server="a",
                num_gpus_per_node=4,
                megatron_num_nodes=1,
                sglang_num_nodes=1,
                sglang={
                    "update_weights": True,
                    "num_gpus_per_engine": 4,
                    "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                },
            ),
            _minimal_actor(
                name="b",
                sglang_server="b",
                num_gpus_per_node=4,
                megatron_num_nodes=1,
                sglang_num_nodes=1,
                sglang={
                    "update_weights": True,
                    "num_gpus_per_engine": 4,
                    "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                },
            ),
        ]
        for c in cfgs:
            validate_policy_config(c)
        slices = derive_policy_slices(cfgs, list(range(16)), colocate=False)
        assert slices["a"] == list(range(0, 4))
        assert slices["b"] == list(range(4, 8))
        assert slices["rollout"] == list(range(8, 16))

    def test_heterogeneous_actor_sizes(self):
        # Policy a: 1 node × 4 = 4 GPUs; policy b: 2 nodes × 4 = 8 GPUs
        cfgs = [
            _minimal_actor(
                name="a",
                sglang_server="a",
                num_gpus_per_node=4,
                megatron_num_nodes=1,
                sglang_num_nodes=1,
                sglang={
                    "update_weights": True,
                    "num_gpus_per_engine": 4,
                    "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                },
            ),
            _minimal_actor(
                name="b",
                sglang_server="b",
                num_gpus_per_node=4,
                megatron_num_nodes=2,
                sglang_num_nodes=1,
                sglang={
                    "update_weights": True,
                    "num_gpus_per_engine": 4,
                    "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                },
            ),
        ]
        for c in cfgs:
            validate_policy_config(c)
        # actor=12, rollout=8, total no-colocate=20
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=False)
        assert (actor, rollout, total) == (12, 8, 20)
        slices = derive_policy_slices(cfgs, list(range(20)), colocate=False)
        assert slices["a"] == list(range(0, 4))
        assert slices["b"] == list(range(4, 12))
        assert slices["rollout"] == list(range(12, 20))

    def test_wrong_idx_length_rejected(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        with pytest.raises(ValueError, match="total_idxs has"):
            derive_policy_slices(cfgs, list(range(10)), colocate=True)


# ────────────────────────────────────────────────────────────────────────────
# config_to_namespace (PolicyHandle's projection)
# ────────────────────────────────────────────────────────────────────────────


class TestConfigToNamespace:
    def _base_args(self, **kw):
        from argparse import Namespace

        defaults = dict(
            colocate=True,
            num_rollout=100,
            rollout_batch_size=32,
            offload_train=False,
            offload_rollout=False,
            check_weight_update_equal=False,
            use_fault_tolerance=False,
            save_interval=20,
            eval_interval=None,
            skip_eval_before_train=False,
            rollout_global_dataset=False,
            start_rollout_id=None,
        )
        defaults.update(kw)
        return Namespace(**defaults)

    def test_all_policy_fields_copied(self):
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(
            name="proposer", sglang_server="proposer", lr=5e-6, kl_coef=0.001, advantage_estimator="grpo"
        )
        ns = config_to_namespace(cfg, self._base_args())
        assert ns.name == "proposer"
        assert ns.lr == 5e-6
        assert ns.kl_coef == 0.001
        assert ns.advantage_estimator == "grpo"
        assert ns.tensor_model_parallel_size == cfg.tensor_model_parallel_size

    def test_policy_name_set(self):
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(name="proposer", sglang_server="proposer")
        ns = config_to_namespace(cfg, self._base_args())
        assert ns.policy_name == "proposer"

    def test_base_args_preserved(self):
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor()
        ns = config_to_namespace(cfg, self._base_args(num_rollout=5000, save_interval=100))
        assert ns.num_rollout == 5000
        assert ns.save_interval == 100
        assert ns.colocate is True

    def test_policy_field_overrides_base_arg_with_same_name(self):
        """If base_args and PolicyConfig both have a field, PolicyConfig wins."""
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(lr=9.99)
        # base_args also has lr — but config_to_namespace overwrites it
        ns = config_to_namespace(cfg, self._base_args(lr=1e-3))
        assert ns.lr == 9.99


# ────────────────────────────────────────────────────────────────────────────
# Parser edge cases
# ────────────────────────────────────────────────────────────────────────────


class TestParserEdgeCases:
    def test_role_defaults_to_actor(self):
        path = _write_yaml(
            {
                "policies": [
                    {
                        "name": "a",
                        # role omitted — should default to "actor"
                        "hf_checkpoint": "/x",
                        "num_gpus_per_node": 8,
                        "megatron_num_nodes": 1,
                        "sglang_num_nodes": 1,
                        "sglang": {
                            "update_weights": True,
                            "num_gpus_per_engine": 8,
                            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                        },
                    }
                ]
            }
        )
        cfgs = parse_policy_configs(path)
        assert cfgs[0].role == "actor"

    def test_buffer_mode_defaults_to_split(self):
        path = _write_yaml(
            {
                "policies": [
                    {
                        "name": "a",
                        "role": "actor",
                        "hf_checkpoint": "/x",
                        # buffer_mode omitted
                        "num_gpus_per_node": 8,
                        "megatron_num_nodes": 1,
                        "sglang_num_nodes": 1,
                        "sglang": {
                            "update_weights": True,
                            "num_gpus_per_engine": 8,
                            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                        },
                    }
                ]
            }
        )
        cfgs = parse_policy_configs(path)
        assert cfgs[0].buffer_mode == "split"

    def test_ref_load_optional(self):
        path = _write_yaml(
            {
                "policies": [
                    {
                        "name": "a",
                        "role": "actor",
                        "hf_checkpoint": "/x",
                        "ref_load": "/ref",  # provided
                        "num_gpus_per_node": 8,
                        "megatron_num_nodes": 1,
                        "sglang_num_nodes": 1,
                        "sglang": {
                            "update_weights": True,
                            "num_gpus_per_engine": 8,
                            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                        },
                    }
                ]
            }
        )
        assert parse_policy_configs(path)[0].ref_load == "/ref"

    def test_ref_load_default_none(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        # Example config has ref_load commented out
        assert all(c.ref_load is None for c in cfgs)

    def test_unknown_top_level_field_silently_dropped(self):
        # Top-level fields are picked by name in the parser; unknown ones are simply
        # not forwarded to PolicyConfig. Documenting actual behavior.
        path = _write_yaml(
            {
                "policies": [
                    {
                        "name": "a",
                        "role": "actor",
                        "hf_checkpoint": "/x",
                        "wizardly_field": True,  # silently ignored
                        "num_gpus_per_node": 8,
                        "megatron_num_nodes": 1,
                        "sglang_num_nodes": 1,
                        "sglang": {
                            "update_weights": True,
                            "num_gpus_per_engine": 8,
                            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                        },
                    }
                ]
            }
        )
        cfgs = parse_policy_configs(path)
        assert len(cfgs) == 1
        assert not hasattr(cfgs[0], "wizardly_field")

    def test_unknown_megatron_field_rejected(self):
        # Fields inside megatron: are **-spread into PolicyConfig — unknowns raise TypeError.
        path = _write_yaml(
            {
                "policies": [
                    {
                        "name": "a",
                        "role": "actor",
                        "hf_checkpoint": "/x",
                        "num_gpus_per_node": 8,
                        "megatron_num_nodes": 1,
                        "sglang_num_nodes": 1,
                        "megatron": {"wizardly_field": True},  # not a PolicyConfig field
                        "sglang": {
                            "update_weights": True,
                            "num_gpus_per_engine": 8,
                            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                        },
                    }
                ]
            }
        )
        with pytest.raises(TypeError):
            parse_policy_configs(path)

    def test_megatron_block_optional(self):
        # If megatron: is missing, all megatron fields use PolicyConfig defaults
        path = _write_yaml(
            {
                "policies": [
                    {
                        "name": "a",
                        "role": "actor",
                        "hf_checkpoint": "/x",
                        "num_gpus_per_node": 8,
                        "megatron_num_nodes": 1,
                        "sglang_num_nodes": 1,
                        "sglang": {
                            "update_weights": True,
                            "num_gpus_per_engine": 8,
                            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                        },
                    }
                ]
            }
        )
        cfgs = parse_policy_configs(path)
        # Defaults from PolicyConfig
        assert cfgs[0].tensor_model_parallel_size == 1
        assert cfgs[0].lr == 1.0e-6
        assert cfgs[0].advantage_estimator == "grpo"


# ────────────────────────────────────────────────────────────────────────────
# build_sglang_config_from_policies — split prefill+decode
# ────────────────────────────────────────────────────────────────────────────


class TestBuildSglangSplit:
    def test_prefill_decode_split(self):
        cfg = _minimal_actor(
            num_gpus_per_node=8,
            megatron_num_nodes=1,
            sglang_num_nodes=1,
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 4,
                "mem_fraction_static": 0.7,
                "server_groups": [
                    {"worker_type": "prefill", "num_gpus": 4, "num_gpus_per_engine": 4},
                    {"worker_type": "decode", "num_gpus": 4, "num_gpus_per_engine": 4},
                ],
            },
        )
        validate_policy_config(cfg)  # 1×8 == 4+4
        sg = build_sglang_config_from_policies([cfg])
        m = sg.models[0]
        assert len(m.server_groups) == 2
        assert [g.worker_type for g in m.server_groups] == ["prefill", "decode"]
        # Both groups inherit model-level mem_fraction_static
        for g in m.server_groups:
            assert g.overrides["mem_fraction_static"] == 0.7

    def test_no_sglang_block_rejected(self):
        cfg = PolicyConfig(name="a", role="actor", hf_checkpoint="/x", sglang_server="a", num_gpus_per_node=8)
        # No `sglang` sub-block → build raises (validate_policy_config skips since cfg.sglang is None)
        with pytest.raises(ValueError, match="missing 'sglang' sub-block"):
            build_sglang_config_from_policies([cfg])


# ────────────────────────────────────────────────────────────────────────────
# Full integration smoke (parser → validate → sizing → slices → sglang config)
# ────────────────────────────────────────────────────────────────────────────


class TestEndToEndPipeline:
    def test_full_pipeline_on_example_yaml(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)

        # Cluster sizing
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=True)
        assert (actor, rollout, total) == (3, 3, 3)

        # Slicing
        slices = derive_policy_slices(cfgs, list(range(total)), colocate=True)
        assert set(slices.keys()) == {"solver", "rewriter", "selector", "rollout"}

        # Sglang projection
        sg = build_sglang_config_from_policies(cfgs)
        assert [m.name for m in sg.models] == ["solver", "rewriter", "selector"]

        # Cross-validation: each policy's slice size matches its actor footprint
        for c in cfgs:
            assert len(slices[c.name]) == c.megatron_num_nodes * c.num_gpus_per_node


# ────────────────────────────────────────────────────────────────────────────
# Determinism / idempotence / roundtrip
# ────────────────────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_parse_twice_equal(self):
        a = parse_policy_configs(EXAMPLE_CONFIG)
        b = parse_policy_configs(EXAMPLE_CONFIG)
        assert a == b

    def test_dataclass_asdict_roundtrip(self):
        cfg = _minimal_actor(name="proposer", lr=5e-6, kl_coef=0.001)
        as_dict = dataclasses.asdict(cfg)
        cfg2 = PolicyConfig(**as_dict)
        assert cfg == cfg2

    def test_validate_idempotent(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        for cfg in cfgs:
            validate_policy_config(cfg)
            validate_policy_config(cfg)  # second call must not raise

    def test_derive_cluster_sizing_idempotent(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        a = derive_cluster_sizing(cfgs, colocate=True)
        b = derive_cluster_sizing(cfgs, colocate=True)
        assert a == b


# ────────────────────────────────────────────────────────────────────────────
# Cross-policy invariants in the actual example config
# ────────────────────────────────────────────────────────────────────────────


class TestExampleInvariants:
    def test_all_three_share_base_model(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        ckpts = {c.hf_checkpoint for c in cfgs}
        assert len(ckpts) == 1, f"all three policies should branch from one base, got {ckpts}"

    def test_num_gpus_per_node_consistent(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        ngs = {c.num_gpus_per_node for c in cfgs}
        assert len(ngs) == 1, f"node-level GPU count should be uniform, got {ngs}"

    def test_save_dirs_per_policy_distinct(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        saves = [c.save for c in cfgs]
        assert len(saves) == len(set(saves)), f"save dirs must be distinct: {saves}"

    def test_n_samples_per_prompt_matches_num_parallel(self):
        """Bug 1 fix regression: rollout_with_multi_agents.py num_parallel must match
        each policy's n_samples_per_prompt in config.yaml so GRPO group-norm reshape
        stays on the fast path. Both are 4 for the smoke-test sizing
        (engines see ~64 requests each per rollout)."""
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        for cfg in cfgs:
            assert cfg.n_samples_per_prompt == 4, (
                f"{cfg.name}: n_samples_per_prompt={cfg.n_samples_per_prompt}, "
                f"should be 4 to match MULTI_AGENT_CONFIGS['num_parallel']"
            )

    def test_selector_n_samples_not_one(self):
        """Bug 2 fix regression: selector.n_samples_per_prompt must be > 1 so GRPO
        group-norm produces non-zero advantage."""
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        selector = next(c for c in cfgs if c.name == "selector")
        assert selector.n_samples_per_prompt > 1, (
            "selector with n_samples_per_prompt=1 makes GRPO group-norm degenerate "
            "(within-group variance is 0). Bug 2 fix requires N parallel selectors."
        )


# ────────────────────────────────────────────────────────────────────────────
# Launcher ↔ config consistency (catches the GPU sizing bug we found earlier)
# ────────────────────────────────────────────────────────────────────────────


class TestLauncherConsistency:
    LAUNCHER_PATH = os.path.join(
        _REPO_ROOT,
        "examples",
        "multi_policy_multi_agent",
        "run-qwen3-0.6B-multi-policy-multi-agent.sh",
    )
    ARGUMENTS_PATH = os.path.join(_REPO_ROOT, "slime", "utils", "arguments.py")

    def test_launcher_num_gpus_matches_derived(self):
        """The launcher's NUM_GPUS must equal derive_cluster_sizing(colocate=True)
        because the launcher passes --colocate. If config.yaml changes (e.g. add a
        4th policy), the launcher must be updated to match."""
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        actor, rollout, total_colocate = derive_cluster_sizing(cfgs, colocate=True)

        with open(self.LAUNCHER_PATH) as f:
            launcher = f.read()

        # Find NUM_GPUS=<n> declaration
        import re

        m = re.search(r"^NUM_GPUS=(\d+)", launcher, re.MULTILINE)
        assert m, "launcher must declare NUM_GPUS=<int>"
        launcher_num_gpus = int(m.group(1))
        assert launcher_num_gpus == total_colocate, (
            f"launcher NUM_GPUS={launcher_num_gpus} but config-derived total_colocate"
            f"={total_colocate}. Update launcher to match config.yaml."
        )

    def test_launcher_uses_colocate(self):
        with open(self.LAUNCHER_PATH) as f:
            launcher = f.read()
        # NUM_GPUS=3 only matches the colocate sizing; without --colocate it'd be 6
        assert "--colocate" in launcher, "launcher passes --colocate to train_multi_policy.py"

    def test_launcher_passes_config_flag(self):
        with open(self.LAUNCHER_PATH) as f:
            launcher = f.read()
        assert "--config" in launcher
        assert "config.yaml" in launcher

    def test_arguments_registers_config_flag(self):
        with open(self.ARGUMENTS_PATH) as f:
            src = f.read()
        assert '"--config"' in src
        assert "train_multi_policy.py" in src


# ────────────────────────────────────────────────────────────────────────────
# Static checks on the example's Python source (regression for bug fixes)
# Uses ast — no module imports, robust to missing deps (transformers, etc.)
# ────────────────────────────────────────────────────────────────────────────


class TestExampleSourceStatic:
    AGENT_SYSTEM = os.path.join(_REPO_ROOT, "examples", "multi_policy_multi_agent", "agent_system.py")
    ROLLOUT_FN = os.path.join(_REPO_ROOT, "examples", "multi_policy_multi_agent", "rollout_with_multi_agents.py")

    def test_agent_system_imports_get_model_url(self):
        """Edit 1 (URL routing): agent_system.py must import get_model_url from sglang_rollout."""
        with open(self.AGENT_SYSTEM) as f:
            src = f.read()
        assert "from slime.rollout.sglang_rollout import get_model_url" in src

    def test_agent_system_uses_get_model_url_for_routing(self):
        """Edit 1: generate_response routes to get_model_url(args, key) not the
        single hardcoded sglang_router_ip:port."""
        with open(self.AGENT_SYSTEM) as f:
            src = f.read()
        assert "url = get_model_url(args, key)" in src
        assert "get_model_url(args, key)}/generate" not in src
        # The single-router hardcode must be gone
        assert "sglang_router_ip" not in src
        assert "sglang_router_port" not in src

    def test_agent_system_tags_policy_name(self):
        """Edit 2 (sample tagging): each Sample must get sample.policy_name = key
        before being appended to results_dict so the manager can route."""
        with open(self.AGENT_SYSTEM) as f:
            src = f.read()
        assert "sample.policy_name = key" in src

    def test_agent_system_has_select_worker(self):
        """Edit 3 (selector parallel): a select_worker function must exist
        alongside solver_worker and rewrite_worker."""
        import ast

        with open(self.AGENT_SYSTEM) as f:
            tree = ast.parse(f.read())
        fn_names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))}
        assert "select_worker" in fn_names
        assert "solver_worker" in fn_names
        assert "rewrite_worker" in fn_names

    def test_agent_system_no_singleton_assert(self):
        """Edit 3: the old `assert len(args.results_dict["selector"]) == 1` blocks
        N parallel selectors. Must be removed."""
        with open(self.AGENT_SYSTEM) as f:
            src = f.read()
        # The exact assertion text shouldn't appear anywhere
        assert 'len(args.results_dict["selector"]) == 1' not in src

    def test_agent_system_uses_mean_selector_reward_gate(self):
        """Edit 3: with N selectors, the global reward shaping must use the mean,
        not selector[0].reward."""
        with open(self.AGENT_SYSTEM) as f:
            src = f.read()
        assert "mean_selector_reward" in src or "mean(s.reward" in src

    def test_rollout_fn_num_parallel_is_4(self):
        """Bug 1 fix: num_parallel must equal n_samples_per_prompt across all
        policies, which is 4 in the current smoke-test config.yaml."""
        import ast

        with open(self.ROLLOUT_FN) as f:
            tree = ast.parse(f.read())
        # Find the MULTI_AGENT_CONFIGS dict assignment
        configs_dict = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "MULTI_AGENT_CONFIGS":
                        configs_dict = node.value
                        break
        assert configs_dict is not None, "MULTI_AGENT_CONFIGS not found"
        # Walk the dict literal to find num_parallel
        num_parallel = None
        for k, v in zip(configs_dict.keys, configs_dict.values, strict=True):
            if isinstance(k, ast.Constant) and k.value == "num_parallel":
                num_parallel = v.value if isinstance(v, ast.Constant) else None
                break
        assert num_parallel == 4, f"num_parallel must be 4, got {num_parallel}"

    def test_rollout_fn_points_at_multi_policy_module(self):
        """The custom_multi_agent_function_path must point at the new multi_policy_multi_agent
        package, not the original multi_agent."""
        with open(self.ROLLOUT_FN) as f:
            src = f.read()
        assert "examples.multi_policy_multi_agent.agent_system.run_agent_system" in src


class TestMultiPolicyWiringStatic:
    TRAIN_MULTI = os.path.join(_REPO_ROOT, "train_multi_policy.py")
    ROLLOUT = os.path.join(_REPO_ROOT, "slime", "ray", "rollout.py")
    TRAIN_ACTOR = os.path.join(_REPO_ROOT, "slime", "ray", "train_actor.py")

    def test_driver_sets_global_hf_checkpoint_from_policy_config(self):
        with open(self.TRAIN_MULTI) as f:
            src = f.read()
        assert "args.hf_checkpoint = policy_configs[0].hf_checkpoint" in src

    def test_driver_sets_megatron_total_gpus(self):
        with open(self.TRAIN_MULTI) as f:
            src = f.read()
        assert "args.megatron_total_gpus = actor_gpus" in src

    def test_rollout_helpers_prefer_megatron_total_gpus(self):
        with open(self.ROLLOUT) as f:
            src = f.read()
        assert 'getattr(args, "megatron_total_gpus", None)' in src

    def test_train_actor_passes_policy_name_to_parallel_config_registration(self):
        with open(self.TRAIN_ACTOR) as f:
            src = f.read()
        assert 'policy_name = getattr(self.args, "policy_name", None)' in src
        assert "policy_name=policy_name" in src


# ────────────────────────────────────────────────────────────────────────────
# SglangConfig projection — non-trivial types (lists, strings, bools)
# ────────────────────────────────────────────────────────────────────────────


class TestSglangProjectionTypes:
    def test_list_field_preserved(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            cuda_bs = m.server_groups[0].overrides["cuda_graph_bs"]
            assert isinstance(cuda_bs, list)
            assert cuda_bs[0] == 1
            assert 32 in cuda_bs  # max_running_requests=32 is captured

    def test_string_field_preserved(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            assert m.server_groups[0].overrides["log_level"] == "info"

    def test_int_field_preserved(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            ov = m.server_groups[0].overrides
            assert ov["chunked_prefill_size"] == 8192
            assert ov["max_running_requests"] == 32

    def test_float_field_preserved(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            assert m.server_groups[0].overrides["mem_fraction_static"] == 0.2

    def test_model_path_passed_through(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            assert m.model_path == "/root/Qwen3-0.6B"

    def test_num_gpus_per_engine_passed_through(self):
        cfgs = parse_policy_configs(EXAMPLE_CONFIG)
        sg = build_sglang_config_from_policies(cfgs)
        for m in sg.models:
            assert m.num_gpus_per_engine == 1


# ────────────────────────────────────────────────────────────────────────────
# Edge cases: empty / null / single-policy
# ────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_policies_list(self):
        path = _write_yaml({"policies": []})
        cfgs = parse_policy_configs(path)
        assert cfgs == []

    def test_empty_policies_list_cluster_sizing(self):
        actor, rollout, total = derive_cluster_sizing([], colocate=True)
        assert (actor, rollout, total) == (0, 0, 0)

    def test_empty_yaml_file(self):
        path = _write_yaml({})
        with pytest.raises(ValueError, match="top-level 'policies' list"):
            parse_policy_configs(path)

    def test_single_policy_run_works(self):
        """N=1 should be a valid degenerate case (essentially equivalent to single-policy
        but going through the multi-policy code path)."""
        path = _write_yaml(
            {
                "policies": [
                    {
                        "name": "actor",
                        "role": "actor",
                        "hf_checkpoint": "/m",
                        "num_gpus_per_node": 8,
                        "megatron_num_nodes": 1,
                        "sglang_num_nodes": 1,
                        "sglang": {
                            "update_weights": True,
                            "num_gpus_per_engine": 8,
                            "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                        },
                    }
                ]
            }
        )
        cfgs = parse_policy_configs(path)
        assert len(cfgs) == 1
        actor, rollout, total = derive_cluster_sizing(cfgs, colocate=True)
        assert (actor, rollout, total) == (8, 8, 8)


# ────────────────────────────────────────────────────────────────────────────
# PolicyHandle smoke (pure dataclass — no Ray)
# ────────────────────────────────────────────────────────────────────────────


class TestPolicyHandleDataclass:
    def test_construct_with_mock_train_group(self):
        """PolicyHandle is a pure dataclass in slime.utils.policy_config — no Ray dep."""
        from argparse import Namespace
        from slime.utils.policy_config import PolicyHandle

        cfg = _minimal_actor()
        ns = Namespace(policy_name="solver", lr=1e-6)
        handle = PolicyHandle(config=cfg, args=ns, train_group=object())
        assert handle.config.name == "solver"
        assert handle.args.policy_name == "solver"

    def test_three_fields_only(self):
        """Schema check: PolicyHandle has exactly {config, args, train_group}.
        Adding/removing fields is a deliberate API change — fail this test on drift."""
        import dataclasses

        from slime.utils.policy_config import PolicyHandle

        names = {f.name for f in dataclasses.fields(PolicyHandle)}
        assert names == {"config", "args", "train_group"}

    def test_mutable_dataclass(self):
        """PolicyHandle is intentionally mutable — driver code may swap train_group
        on recovery. Do not flip this to frozen=True without auditing call sites."""
        from argparse import Namespace
        from slime.utils.policy_config import PolicyHandle

        h = PolicyHandle(config=_minimal_actor(), args=Namespace(), train_group="a")
        h.train_group = "b"  # must not raise
        assert h.train_group == "b"

    def test_built_from_config_to_namespace(self):
        """Common-path integration: config_to_namespace(cfg, base) → PolicyHandle.args.
        Verifies the projection PolicyConfig → Namespace lands the fields the driver
        relies on (policy_name, hf_checkpoint, role, sglang_server)."""
        from argparse import Namespace
        from slime.utils.policy_config import PolicyHandle, config_to_namespace

        cfg = _minimal_actor()  # name="solver"
        base = Namespace(kl_coef=0.0, use_kl_loss=False, lr=1e-6)
        args_p = config_to_namespace(cfg, base)
        h = PolicyHandle(config=cfg, args=args_p, train_group=object())

        assert h.args.policy_name == "solver"
        assert h.args.hf_checkpoint == cfg.hf_checkpoint
        assert h.args.role == "actor"
        assert h.args.sglang_server == cfg.sglang_server
        # Globals from base_args still present
        assert h.args.kl_coef == 0.0
        assert h.args.lr == 1e-6

    def test_per_policy_namespaces_are_independent(self):
        """Two PolicyHandles built from the same base_args have independent
        namespaces — mutating one's args must not affect the other (essential
        for SPIRAL-style runs where each policy needs its own loss_type, lr, etc.)."""
        from argparse import Namespace
        from slime.utils.policy_config import config_to_namespace

        cfg_a = _minimal_actor()
        cfg_b = dataclasses.replace(_minimal_actor(), name="rewriter", sglang_server="rewriter")
        base = Namespace(kl_coef=0.0, lr=1e-6)

        ns_a = config_to_namespace(cfg_a, base)
        ns_b = config_to_namespace(cfg_b, base)

        ns_a.lr = 2e-6
        assert ns_b.lr == 1e-6  # not mutated by ns_a write
        assert base.lr == 1e-6  # base also not mutated


# ────────────────────────────────────────────────────────────────────────────
# config_to_namespace projection — covered separately because it's the bridge
# between PolicyConfig and the per-policy Namespace the driver hands to Ray
# ────────────────────────────────────────────────────────────────────────────


class TestConfigToNamespaceProjection:
    def test_global_fields_inherited_from_base(self):
        from argparse import Namespace
        from slime.utils.policy_config import config_to_namespace

        base = Namespace(num_rollout=10, save_interval=5, custom_path="/x")
        ns = config_to_namespace(_minimal_actor(), base)
        assert ns.num_rollout == 10
        assert ns.save_interval == 5
        assert ns.custom_path == "/x"

    def test_policy_fields_overlay_global(self):
        """When a name collides between global args and PolicyConfig, the policy
        field wins (e.g. global hf_checkpoint default replaced by per-policy)."""
        from argparse import Namespace
        from slime.utils.policy_config import config_to_namespace

        base = Namespace(hf_checkpoint="/global/path")
        cfg = _minimal_actor()  # cfg.hf_checkpoint != "/global/path"
        ns = config_to_namespace(cfg, base)
        assert ns.hf_checkpoint == cfg.hf_checkpoint
        assert ns.hf_checkpoint != "/global/path"

    def test_policy_name_set_from_cfg_name(self):
        """policy_name is appended even though it's not a PolicyConfig field —
        downstream Megatron code reads args.policy_name for weight-update routing."""
        from argparse import Namespace
        from slime.utils.policy_config import config_to_namespace

        ns = config_to_namespace(_minimal_actor(), Namespace())
        assert ns.policy_name == "solver"

    def test_legacy_actor_sizing_aliases_are_policy_specific(self):
        """Existing Megatron/update-weight code still reads actor_num_* fields.
        The per-policy namespace must not inherit stale global sizing."""
        from argparse import Namespace
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(megatron_num_nodes=2, num_gpus_per_node=4)
        base = Namespace(actor_num_nodes=99, actor_num_gpus_per_node=99, num_gpus_per_node=99, world_size=99)
        ns = config_to_namespace(cfg, base)
        assert ns.actor_num_nodes == 2
        assert ns.actor_num_gpus_per_node == 4
        assert ns.num_gpus_per_node == 4
        assert ns.world_size == 8

    def test_does_not_mutate_base_args(self):
        from argparse import Namespace
        from slime.utils.policy_config import config_to_namespace

        base = Namespace(hf_checkpoint="/global", kl_coef=0.5)
        original = vars(base).copy()
        _ = config_to_namespace(_minimal_actor(), base)
        assert vars(base) == original

    def test_does_not_mutate_cfg(self):
        from argparse import Namespace
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor()
        before = dataclasses.asdict(cfg)
        _ = config_to_namespace(cfg, Namespace())
        assert dataclasses.asdict(cfg) == before


# ────────────────────────────────────────────────────────────────────────────
# Per-policy weight-load fallback — the critical chain that lets a policy
# load weights from HF when no Megatron torch_dist checkpoint exists yet
# ────────────────────────────────────────────────────────────────────────────


class TestWeightLoadFallback:
    def _ns(self, **overrides):
        from argparse import Namespace

        defaults = dict(no_load_optim=False, no_load_rng=False, finetune=False, ref_load=None, start_rollout_id=None)
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_bridge_with_load_none_falls_back_to_hf_checkpoint(self):
        """Mirrors upstream slime_validate_args: bridge mode + load None +
        ref_load None → ns.load = cfg.hf_checkpoint. mbridge then resolves
        the hub id at AutoBridge.from_hf_pretrained time."""
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(megatron_to_hf_mode="bridge", load=None, ref_load=None, hf_checkpoint="Qwen/Qwen3-0.6B")
        ns = config_to_namespace(cfg, self._ns())
        assert ns.load == "Qwen/Qwen3-0.6B"
        assert ns.start_rollout_id == 0

    def test_bridge_with_real_megatron_ckpt_keeps_load(self, tmp_path):
        """If `load` points at a real torch_dist Megatron ckpt, bridge mode
        respects it (resumes instead of HF-loading)."""
        from slime.utils.policy_config import config_to_namespace

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "latest_checkpointed_iteration.txt").write_text("100")
        cfg = _minimal_actor(megatron_to_hf_mode="bridge", load=str(ckpt))
        ns = config_to_namespace(cfg, self._ns())
        assert ns.load == str(ckpt)

    def test_bridge_uses_ref_load_when_set(self):
        """Bridge mode + load None + ref_load set → ns.load = ref_load."""
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(megatron_to_hf_mode="bridge", load=None, ref_load="/path/ref")
        ns = config_to_namespace(cfg, self._ns())
        assert ns.load == "/path/ref"

    def test_raw_mode_with_no_ckpt_marks_finetune(self):
        """Default raw mode + no real ckpt → mirror slime_validate_args:
        no_load_optim=True, no_load_rng=True, finetune=True, start_rollout_id=0."""
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(megatron_to_hf_mode="raw", load="/does/not/exist")
        ns = config_to_namespace(cfg, self._ns())
        assert ns.no_load_optim is True
        assert ns.no_load_rng is True
        assert ns.finetune is True
        assert ns.start_rollout_id == 0


class TestEpsClipHighDefaulting:
    """slime_validate_args sets args.eps_clip_high = args.eps_clip when None.
    config_to_namespace must mirror this per-policy or compute_policy_loss
    will hit `1 + None` TypeError downstream."""

    def _ns(self, **overrides):
        from argparse import Namespace

        defaults = dict(no_load_optim=False, no_load_rng=False, finetune=False, ref_load=None, start_rollout_id=None)
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_eps_clip_high_defaults_to_eps_clip_when_none(self):
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(eps_clip=0.2, eps_clip_high=None)
        ns = config_to_namespace(cfg, self._ns())
        assert ns.eps_clip_high == 0.2

    def test_explicit_eps_clip_high_preserved(self):
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(eps_clip=0.2, eps_clip_high=0.28)
        ns = config_to_namespace(cfg, self._ns())
        assert ns.eps_clip_high == 0.28


class TestPerPolicyDerivedDefaults:
    """Other slime_validate_args defaults that depend on per-policy values
    and would silently miscompute on the per-policy ns without mirroring."""

    def _ns(self, **overrides):
        from argparse import Namespace

        defaults = dict(no_load_optim=False, no_load_rng=False, finetune=False, ref_load=None, start_rollout_id=None)
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_grpo_std_normalization_off_when_n_samples_is_1(self):
        """n_samples_per_prompt=1 means the GRPO group has size 1; std is 0
        → group-norm divides by zero. Upstream forces grpo_std_normalization
        off; mirror it per-policy."""
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(n_samples_per_prompt=1, grpo_std_normalization=True)
        ns = config_to_namespace(cfg, self._ns())
        assert ns.grpo_std_normalization is False

    def test_grpo_std_normalization_kept_when_n_samples_gt_1(self):
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(n_samples_per_prompt=4, grpo_std_normalization=True)
        ns = config_to_namespace(cfg, self._ns())
        assert ns.grpo_std_normalization is True

    def test_log_probs_max_tokens_per_gpu_falls_back(self):
        """When use_dynamic_batch_size is on, log_probs_max_tokens_per_gpu
        defaults to max_tokens_per_gpu — same dynamic-batch budget as the
        train forward."""
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(use_dynamic_batch_size=True, max_tokens_per_gpu=8192, log_probs_max_tokens_per_gpu=None)
        ns = config_to_namespace(cfg, self._ns())
        assert ns.log_probs_max_tokens_per_gpu == 8192

    def test_log_probs_max_tokens_per_gpu_explicit_preserved(self):
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(use_dynamic_batch_size=True, max_tokens_per_gpu=8192, log_probs_max_tokens_per_gpu=2048)
        ns = config_to_namespace(cfg, self._ns())
        assert ns.log_probs_max_tokens_per_gpu == 2048

    def test_log_probs_max_tokens_per_gpu_no_op_without_dynamic_batch(self):
        """Without use_dynamic_batch_size, slime never reads this knob; leave
        it None (matches upstream behavior — no defaulting branch fires)."""
        from slime.utils.policy_config import config_to_namespace

        cfg = _minimal_actor(use_dynamic_batch_size=False, max_tokens_per_gpu=8192, log_probs_max_tokens_per_gpu=None)
        ns = config_to_namespace(cfg, self._ns())
        assert ns.log_probs_max_tokens_per_gpu is None


class TestFrozenEnginePolicyShape:
    """Coverage for the m✗ s✓ trainable=false policy shape (frozen standalone
    SGLang engine, e.g. OPD SGLang teacher / judge / RM).

    The framework historically equated `cfg.trainable` with "hosts an engine".
    This class verifies the four widened predicate sites — `has_engine()` /
    `has_megatron()` helpers, `derive_cluster_sizing`, the validator's engine
    placement check, and `build_sglang_config_from_policies` — handle the new
    shape, while preserving behavior for paired-trainable and Megatron-only
    frozen shapes.
    """

    def _frozen_sglang_teacher(self, **overrides) -> PolicyConfig:
        base = dict(
            name="teacher_sglang",
            role="actor",
            trainable=False,
            hf_checkpoint="/x",
            sglang_server="teacher_sglang",
            buffer_mode="split",
            num_gpus_per_node=1,
            megatron_num_nodes=0,
            sglang_num_nodes=1,
            sglang={
                "update_weights": False,
                "num_gpus_per_engine": 1,
                "server_groups": [{"worker_type": "regular", "num_gpus": 1}],
            },
        )
        base.update(overrides)
        return PolicyConfig(**base)

    def _megatron_only_frozen_teacher(self, **overrides) -> PolicyConfig:
        # Mirrors examples/multi_policy_opd_megatron/config.yaml teacher shape.
        base = dict(
            name="teacher_megatron",
            role="actor",
            trainable=False,
            hf_checkpoint="/x",
            sglang_server="teacher_megatron",
            buffer_mode="split",
            num_gpus_per_node=1,
            megatron_num_nodes=1,
            sglang_num_nodes=0,
            sglang=None,
        )
        base.update(overrides)
        return PolicyConfig(**base)

    # ── helpers ──
    def test_has_engine_trainable_paired(self):
        cfg = _minimal_actor()
        assert cfg.has_engine() is True
        assert cfg.has_megatron() is True

    def test_has_engine_megatron_only_frozen(self):
        cfg = self._megatron_only_frozen_teacher()
        assert cfg.has_engine() is False  # sglang_num_nodes=0 → no engine
        assert cfg.has_megatron() is True

    def test_has_engine_sglang_only_frozen(self):
        cfg = self._frozen_sglang_teacher()
        assert cfg.has_engine() is True  # sglang block + nodes>0
        assert cfg.has_megatron() is False  # megatron_num_nodes=0

    # ── validate_policy_config ──
    def test_validator_accepts_sglang_only_frozen(self):
        validate_policy_config(self._frozen_sglang_teacher())  # no raise

    def test_validator_rejects_megatron_less_trainable(self):
        # megatron_num_nodes=0 + trainable=True is logically impossible.
        cfg = self._frozen_sglang_teacher(trainable=True)
        with pytest.raises(ValueError, match="megatron_num_nodes=0 requires trainable=false"):
            validate_policy_config(cfg)

    def test_validator_rejects_no_gpus_at_all(self):
        # megatron_num_nodes=0 + sglang_num_nodes=0 (or no sglang block) is a
        # policy with zero GPUs — meaningless.
        cfg = self._frozen_sglang_teacher(sglang_num_nodes=0, sglang=None)
        with pytest.raises(ValueError, match="no SGLang engine"):
            validate_policy_config(cfg)

    def test_validator_catches_sglang_placement_on_frozen_engine(self):
        # Previously this was skipped for trainable=False; now applies whenever
        # has_engine() is true. 1 node × 1 gpu = 1, server_groups sums to 2.
        cfg = self._frozen_sglang_teacher(
            sglang={
                "update_weights": False,
                "num_gpus_per_engine": 1,
                "server_groups": [{"worker_type": "regular", "num_gpus": 2}],
            }
        )
        with pytest.raises(ValueError, match="must equal sum of sglang.server_groups"):
            validate_policy_config(cfg)

    def test_validator_still_skips_megatron_only_frozen(self):
        # m✓ s✗ frozen producer: no engine → placement check skipped.
        validate_policy_config(self._megatron_only_frozen_teacher())  # no raise

    # ── derive_cluster_sizing ──
    def test_cluster_sizing_counts_frozen_sglang_teacher(self):
        # Mirrors examples/multi_policy_opd_sglang/dev_config.yaml:
        #   student (m✓ s✓) + teacher_sglang (m✗ s✓ frozen)
        student = _minimal_actor(
            name="student",
            sglang_server="student",
            num_gpus_per_node=1,
            megatron_num_nodes=1,
            sglang_num_nodes=1,
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 1,
                "server_groups": [{"worker_type": "regular", "num_gpus": 1}],
            },
        )
        teacher = self._frozen_sglang_teacher()
        actor, rollout, total = derive_cluster_sizing([student, teacher], colocate=False)
        # actor: only student (teacher.megatron_num_nodes=0)
        # rollout: BOTH student and teacher engines now counted (was 1 before fix)
        assert actor == 1 and rollout == 2 and total == 3

    def test_cluster_sizing_megatron_only_frozen_unchanged(self):
        # Behavior parity for OPD-Megatron shape: frozen Megatron teacher
        # still counted in actor_gpus, not in rollout_gpus.
        student = _minimal_actor(
            name="student",
            sglang_server="student",
            num_gpus_per_node=1,
            megatron_num_nodes=1,
            sglang_num_nodes=1,
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 1,
                "server_groups": [{"worker_type": "regular", "num_gpus": 1}],
            },
        )
        teacher = self._megatron_only_frozen_teacher()
        actor, rollout, total = derive_cluster_sizing([student, teacher], colocate=False)
        assert actor == 2 and rollout == 1 and total == 3

    # ── build_sglang_config_from_policies ──
    def test_sglang_config_includes_frozen_sglang_teacher(self):
        student = _minimal_actor(
            name="student",
            sglang_server="student",
            num_gpus_per_node=1,
            megatron_num_nodes=1,
            sglang_num_nodes=1,
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 1,
                "server_groups": [{"worker_type": "regular", "num_gpus": 1}],
            },
        )
        teacher = self._frozen_sglang_teacher()
        sglang_config = build_sglang_config_from_policies([student, teacher])
        names = [m.name for m in sglang_config.models]
        assert names == ["student", "teacher_sglang"]
        # Frozen engine carries update_weights=False through to its ModelConfig.
        teacher_model = next(m for m in sglang_config.models if m.name == "teacher_sglang")
        assert teacher_model.update_weights is False

    def test_sglang_config_still_skips_megatron_only_frozen(self):
        # Behavior parity for OPD-Megatron shape: teacher_megatron has no
        # sglang block and is omitted from SglangConfig.models.
        student = _minimal_actor(
            name="student",
            sglang_server="student",
            num_gpus_per_node=1,
            megatron_num_nodes=1,
            sglang_num_nodes=1,
            sglang={
                "update_weights": True,
                "num_gpus_per_engine": 1,
                "server_groups": [{"worker_type": "regular", "num_gpus": 1}],
            },
        )
        teacher = self._megatron_only_frozen_teacher()
        sglang_config = build_sglang_config_from_policies([student, teacher])
        assert [m.name for m in sglang_config.models] == ["student"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
