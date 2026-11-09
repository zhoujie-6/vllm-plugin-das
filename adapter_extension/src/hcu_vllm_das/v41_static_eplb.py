# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""V4.1 static placement; parser and loader fanout ported from static_eplb.py.

The v2 artifact format remains owned by the existing offline EPLB workflow.
"""
from __future__ import annotations

import functools

import hashlib

import json

from collections.abc import Mapping

from dataclasses import dataclass

from functools import lru_cache

from pathlib import Path

from types import MethodType

from typing import Any

import torch

_OFFLINE_EPLB_FORMAT = "vllm_offline_eplb_physical_to_logical_by_model"

_RECORD_KINDS = frozenset({"initial", "proposal"})

def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"Static EPLB {name} must be an integer >= {minimum}"
        )
    return value

@dataclass(frozen=True, slots=True)
class StaticEplbPlan:
    """A validated, immutable physical-to-logical expert placement."""

    model_key: str
    model_name: str
    source_path: str
    source_sha256: str
    record_kind: str
    _map_values: tuple[tuple[int, ...], ...]
    num_logical_experts: int
    num_physical_experts: int
    num_redundant_experts: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_key, str) or not self.model_key:
            raise ValueError("Static EPLB model_key must be nonempty")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("Static EPLB model_name must be nonempty")
        if (
            not isinstance(self.source_path, str)
            or not Path(self.source_path).is_absolute()
        ):
            raise ValueError("Static EPLB source_path must be absolute")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_sha256
            )
        ):
            raise ValueError(
                "Static EPLB source_sha256 must be a SHA-256 hex digest"
            )
        if (
            not isinstance(self.record_kind, str)
            or self.record_kind not in _RECORD_KINDS
        ):
            raise ValueError("Static EPLB record_kind must be initial or proposal")
        _integer(self.num_logical_experts, "num_logical_experts", 1)
        _integer(self.num_physical_experts, "num_physical_experts", 1)
        _integer(self.num_redundant_experts, "num_redundant_experts")
        if (
            self.num_logical_experts + self.num_redundant_experts
            != self.num_physical_experts
        ):
            raise ValueError(
                "Static EPLB physical/logical/redundant counts disagree"
            )
        if type(self._map_values) is not tuple or not self._map_values:
            raise ValueError(
                "Static EPLB rows must be a nonempty immutable tuple"
            )

        required = set(range(self.num_logical_experts))
        for row in self._map_values:
            if type(row) is not tuple or len(row) != self.num_physical_experts:
                raise ValueError(
                    "Static EPLB row shape or immutability mismatch"
                )
            if any(
                type(expert_id) is not int
                or not 0 <= expert_id < self.num_logical_experts
                for expert_id in row
            ):
                raise ValueError("Static EPLB invalid logical expert ID")
            if set(row) != required:
                raise ValueError("Static EPLB row misses logical experts")

    @property
    def physical_to_logical_map(self) -> torch.Tensor:
        """Return an isolated CPU tensor copy of the immutable placement."""

        return torch.tensor(self._map_values, dtype=torch.int64, device="cpu")

    def layer_map(self, layer_idx: int) -> tuple[int, ...]:
        return self._map_values[layer_idx]

    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.model_key,
            self.model_name,
            self.source_sha256,
            (len(self._map_values), self.num_physical_experts),
            self.num_logical_experts,
            self.num_physical_experts,
            self.num_redundant_experts,
        )

def _required_field(payload: dict[str, Any], name: str) -> Any:
    if name not in payload:
        raise ValueError(f"Static EPLB model map is missing {name}")
    return payload[name]

@lru_cache(maxsize=16)
def _parse_plan(
    raw: bytes,
    canonical_path: str,
    model_key: str,
    expected_model_name: str,
    expected_shape: tuple[int, int],
    num_logical_experts: int,
    num_redundant_experts: int,
    require_proposal: bool,
) -> StaticEplbPlan:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Static EPLB file must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Static EPLB top-level payload must be an object")
    if type(payload.get("version")) is not int or payload["version"] != 2:
        raise ValueError("Static EPLB file must declare version 2")
    if payload.get("format") != _OFFLINE_EPLB_FORMAT:
        raise ValueError(
            f"Static EPLB file must declare format {_OFFLINE_EPLB_FORMAT!r}"
        )
    model_maps = payload.get("model_maps")
    if not isinstance(model_maps, dict):
        raise ValueError("Static EPLB model_maps must be an object")
    if model_key not in model_maps:
        raise ValueError(f"Static EPLB missing model key {model_key!r}")
    selected = model_maps[model_key]
    if not isinstance(selected, dict):
        raise ValueError("Static EPLB model map must be an object")

    record_kind = _required_field(selected, "record_kind")
    if not isinstance(record_kind, str) or record_kind not in _RECORD_KINDS:
        raise ValueError(
            "Static EPLB record_kind must be initial or proposal"
        )
    if require_proposal and record_kind != "proposal":
        raise ValueError(
            f"Static EPLB model key {model_key!r} has no calibration proposal"
        )
    model_name = _required_field(selected, "model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Static EPLB model_name must be a nonempty string")
    if model_name != expected_model_name:
        raise ValueError("Static EPLB declared model_name mismatch")

    expected_metadata = {
        "model_class": model_key,
        "num_moe_layers": expected_shape[0],
        "num_logical_experts": num_logical_experts,
        "num_physical_experts": expected_shape[1],
        "num_redundant_experts": num_redundant_experts,
    }
    for name, expected in expected_metadata.items():
        value = _required_field(selected, name)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Static EPLB declared {name} mismatch")

    rows = _required_field(selected, "physical_to_logical_map")
    if (
        not isinstance(rows, list)
        or len(rows) != expected_shape[0]
        or any(not isinstance(row, list) for row in rows)
    ):
        raise ValueError("Static EPLB map shape mismatch")

    return StaticEplbPlan(
        model_key=model_key,
        model_name=model_name,
        source_path=canonical_path,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        record_kind=record_kind,
        _map_values=tuple(tuple(row) for row in rows),
        num_logical_experts=num_logical_experts,
        num_physical_experts=expected_shape[1],
        num_redundant_experts=num_redundant_experts,
    )

def load_static_eplb_plan(
    path: str | Path,
    *,
    model_key: str,
    expected_model_name: str,
    expected_shape: tuple[int, int],
    num_logical_experts: int,
    num_redundant_experts: int,
    require_proposal: bool = True,
) -> StaticEplbPlan:
    """Read and validate a version-2 offline plan.

    File bytes are read on every call. Parsing is cached only by those exact
    bytes, so replacing a file while preserving its size and mtime cannot
    return a stale placement.
    """

    if not isinstance(model_key, str) or not model_key:
        raise ValueError("Static EPLB model_key must be nonempty")
    if not isinstance(expected_model_name, str) or not expected_model_name:
        raise ValueError("Static EPLB expected_model_name must be nonempty")
    if type(expected_shape) is not tuple or len(expected_shape) != 2:
        raise ValueError(
            "Static EPLB expected_shape must contain two dimensions"
        )
    for dimension in expected_shape:
        _integer(dimension, "shape", 1)
    _integer(num_logical_experts, "num_logical_experts", 1)
    _integer(num_redundant_experts, "num_redundant_experts")
    if type(require_proposal) is not bool:
        raise ValueError("Static EPLB require_proposal must be bool")
    if num_logical_experts + num_redundant_experts != expected_shape[1]:
        raise ValueError(
            "Static EPLB physical/logical/redundant counts disagree"
        )

    canonical = Path(path).expanduser().resolve(strict=True)
    return _parse_plan(
        canonical.read_bytes(),
        str(canonical),
        model_key,
        expected_model_name,
        expected_shape,
        num_logical_experts,
        num_redundant_experts,
        require_proposal,
    )

def validate_static_eplb_artifact(
    path: str | Path,
    *,
    required_models: Mapping[str, int],
    expected_model_name: str,
    num_logical_experts: int,
    num_redundant_experts: int,
) -> str:
    """Validate a stable, proposal-complete multi-model placement file."""

    if not isinstance(required_models, Mapping) or not required_models:
        raise ValueError("Static EPLB required_models must be nonempty")
    if not isinstance(expected_model_name, str) or not expected_model_name:
        raise ValueError("Static EPLB expected_model_name must be nonempty")
    logical = _integer(num_logical_experts, "num_logical_experts", 1)
    redundant = _integer(num_redundant_experts, "num_redundant_experts")
    physical = logical + redundant
    canonical = Path(path).expanduser().resolve(strict=True)
    raw = canonical.read_bytes()
    for model_key, expected_layers in required_models.items():
        if not isinstance(model_key, str) or not model_key:
            raise ValueError("Static EPLB required model key must be nonempty")
        layers = _integer(
            expected_layers,
            f"{model_key} num_moe_layers",
            1,
        )
        _parse_plan(
            raw,
            str(canonical),
            model_key,
            expected_model_name,
            (layers, physical),
            logical,
            redundant,
            True,
        )
    if canonical.read_bytes() != raw:
        raise ValueError("Static EPLB file changed during validation")
    return hashlib.sha256(raw).hexdigest()

def verify_static_plan_across_ep_ranks(plan: StaticEplbPlan) -> None:
    """Fail unless every EP rank loaded the exact same plan fingerprint."""

    if not torch.distributed.is_initialized():
        return
    from vllm.distributed import get_ep_group

    group = get_ep_group()
    fingerprint = plan.fingerprint()
    fingerprints: list[tuple[object, ...] | None] = [None] * group.world_size
    torch.distributed.all_gather_object(
        fingerprints,
        fingerprint,
        group=group.cpu_group,
    )
    if any(value != fingerprint for value in fingerprints):
        raise RuntimeError(
            "Static EPLB plan fingerprint mismatch across EP ranks"
        )

def load_static_logical_expert(
    routed_experts: object,
    original_weight_loader,
    *,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    logical_expert_id: int,
    return_success: bool,
):
    """Load one logical checkpoint expert into every mapped physical slot."""

    row = routed_experts._vllm_hcu_static_eplb_row
    if type(logical_expert_id) is not int or logical_expert_id not in row:
        raise ValueError(
            f"Static EPLB invalid logical expert ID {logical_expert_id!r}"
        )
    if isinstance(loaded_weight, torch.Tensor) and loaded_weight.ndim >= 3:
        raise ValueError(
            "Static EPLB fused expert tensors must be split by the current loader"
        )
    if (
        getattr(
            getattr(routed_experts, "quant_method", None),
            "use_global_sf",
            False,
        )
        and "input_scale" in weight_name
    ):
        return original_weight_loader(
            routed_experts,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=logical_expert_id,
            return_success=return_success,
        )

    loaded_any = False
    for physical_expert_id, mapped_logical_id in enumerate(row):
        if mapped_logical_id != logical_expert_id:
            continue
        loaded = original_weight_loader(
            routed_experts,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=physical_expert_id,
            return_success=True,
        )
        loaded_any = bool(loaded) or loaded_any
    return loaded_any if return_success else None

def _static_loader(original):
    @functools.wraps(original)
    def load(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        logical_experts = self.moe_config.num_logical_experts
        fused_shared_experts = (
            self.expert_map_manager.num_fused_shared_experts
        )
        if (
            type(expert_id) is int
            and logical_experts
            <= expert_id
            < logical_experts + fused_shared_experts
        ):
            physical_tail_id = (
                len(self._vllm_hcu_static_eplb_row)
                + expert_id
                - logical_experts
            )
            return original(
                self,
                param,
                loaded_weight,
                weight_name,
                shard_id,
                physical_tail_id,
                return_success=return_success,
            )
        return load_static_logical_expert(
            self,
            original,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            logical_expert_id=expert_id,
            return_success=return_success,
        )

    return load

def _static_expert_mapping(
    self,
    ckpt_gate_proj_name=None,
    ckpt_down_proj_name=None,
    ckpt_up_proj_name=None,
    include_fused=False,
):
    return self.build_expert_params_mapping(
        ckpt_gate_proj_name or self.ckpt_gate_proj_name,
        ckpt_down_proj_name or self.ckpt_down_proj_name,
        ckpt_up_proj_name or self.ckpt_up_proj_name,
        self.moe_config.num_logical_experts
        + self.expert_map_manager.num_fused_shared_experts,
        num_redundant_experts=0,
        routed_experts_prefix="",
        lora_base_layer_prefix=self.lora_base_layer_prefix,
        include_fused=include_fused,
    )

# Static dispatch policy deliberately stays outside the dynamic EPLB worker.
# Existing offline tooling produces version-2 placement artifacts.
def static_plan(config):
    from .environment import static_expert_map_path
    import json
    path = static_expert_map_path()
    if not path:
        return None
    if config.parallel_config.enable_eplb:
        raise ValueError("Static expert map cannot be combined with dynamic --enable-eplb")
    raw = Path(path).expanduser().resolve(strict=True).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    cached = getattr(config, "_hcu_v41_static_plan", None)
    if cached is not None:
        if cached.source_sha256 != sha or cached.source_path != str(Path(path).expanduser().resolve()):
            raise ValueError("Static expert map changed after model construction")
        return cached
    data = json.loads(raw)
    key = "DeepseekV41ForCausalLM"
    record = data.get("model_maps", {}).get(key, {})
    redundant = _integer(record.get("num_redundant_experts"), "num_redundant_experts")
    hf = config.model_config.hf_config
    plan = load_static_eplb_plan(
        path, model_key=key, expected_model_name=config.model_config.model,
        expected_shape=(hf.num_hidden_layers, hf.n_routed_experts + redundant),
        num_logical_experts=hf.n_routed_experts, num_redundant_experts=redundant)
    if plan.source_sha256 != sha:
        raise ValueError("Static expert map changed during construction")
    if plan.num_physical_experts % 8:
        raise ValueError("Static expert count must divide EP8")
    config._hcu_v41_static_plan = plan
    return plan


def _layer_index(prefix, plan):
    import re
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
    if match is None or int(match[1]) >= len(plan._map_values):
        raise ValueError("Static V41 expert map requires a target layer prefix")
    return int(match[1])


def bind_weight_loaders(layer, plan, prefix):
    row = plan.layer_map(_layer_index(prefix, plan))
    if layer.moe_config.num_experts != len(row):
        raise ValueError("Static physical expert allocation disagrees with placement")
    layer._vllm_hcu_static_eplb_row = row
    original = layer.weight_loader.__func__
    loader = MethodType(_static_loader(original), layer)
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        getattr(layer, name).weight_loader = loader
    layer.get_expert_mapping = MethodType(_static_expert_mapping, layer)


def routing_table(plan, prefix, device):
    row = plan.layer_map(_layer_index(prefix, plan))
    # Round-robin among replicas in physical order. No routing weights change.
    replicas = [[p for p, logical in enumerate(row) if logical == i]
                for i in range(plan.num_logical_experts)]
    width = max(map(len, replicas))
    ids = torch.tensor([r + [-1] * (width - len(r)) for r in replicas],
                       dtype=torch.int64, device=device)
    counts = torch.tensor(list(map(len, replicas)), dtype=torch.int64, device=device)
    return ids, counts


def map_routes(topk_ids, table):
    ids, counts = table
    valid = topk_ids >= 0
    logical = topk_ids.clamp_min(0).long()
    # Include source EP rank to avoid identical replica choices on PCP ranks.
    from vllm.distributed import get_ep_group
    rank = get_ep_group().rank_in_group
    offsets = torch.arange(topk_ids.numel(), device=topk_ids.device,
                           dtype=torch.int64).view(topk_ids.shape) + rank
    physical = ids[logical, offsets % counts[logical]]
    return torch.where(valid, physical, -1)
