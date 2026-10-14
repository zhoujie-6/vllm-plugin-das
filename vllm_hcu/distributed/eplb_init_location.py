# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Static initial EPLB expert placement (``--init-expert-location``).

Mirrors SGLang's ``--init-expert-location``.  Two payload formats are
accepted, as ``.pt`` (``torch.load``), ``.json`` or an inline JSON string:

* ``{"logical_count": [..., num_layers, num_logical_experts]}`` -- the dump of
  SGLang's ``expert_distribution_recorder`` (``stat`` mode).  Leading
  dimensions (buffered steps) are summed.  The placement is computed with
  vLLM's configured EPLB policy, the same policy used by runtime rebalancing.
* ``{"physical_to_logical_map": [num_layers, num_physical_experts]}`` -- an
  explicit placement, applied as-is.

vLLM loads expert weights in the trivial layout (physical ``i`` holds logical
``i % num_routed_experts``).  The new placement is therefore realized once,
right after ``EplbState.add_model``, through the regular EPLB weight shuffle
(``rearrange_expert_weights_inplace``) and map commit, so every rank ends up
in exactly the state a runtime rebalance would have produced.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

# vLLM only installs handlers on the "vllm" logger tree.
logger = init_logger("vllm.hcu.eplb")


def load_expert_location_payload(location: str) -> Mapping[str, Any]:
    """Read an ``--init-expert-location`` value into a mapping."""

    if location.endswith(".pt"):
        payload = torch.load(location, weights_only=True, map_location="cpu")
    elif location.endswith(".json"):
        payload = json.loads(Path(location).read_text())
    else:
        payload = json.loads(location)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"init_expert_location {location!r} must contain a mapping, "
            f"got {type(payload).__name__}"
        )
    if "physical_to_logical_map" not in payload and "logical_count" not in payload:
        raise ValueError(
            f"init_expert_location {location!r} has neither "
            f"'physical_to_logical_map' nor 'logical_count' (keys={list(payload)})"
        )
    return payload


def _strip_non_moe_layers(
    table: torch.Tensor, num_moe_layers: int, name: str
) -> torch.Tensor:
    """Accept a per-decoder-layer table whose leading dense layers are empty."""

    num_layers = table.shape[0]
    if num_layers == num_moe_layers:
        return table
    extra = num_layers - num_moe_layers
    if extra > 0 and not table[:extra].any():
        logger.info(
            "init_expert_location: dropping %d leading non-MoE layer(s) from %s",
            extra,
            name,
        )
        return table[extra:]
    raise ValueError(
        f"init_expert_location {name} covers {num_layers} layers but the model "
        f"has {num_moe_layers} MoE layers"
    )


def logical_load_from_payload(
    payload: Mapping[str, Any], num_moe_layers: int, num_logical_experts: int
) -> torch.Tensor:
    """Return the expert load as ``[num_moe_layers, num_logical_experts]``."""

    load = torch.as_tensor(payload["logical_count"]).to(torch.int64)
    if load.dim() < 2:
        raise ValueError(
            f"logical_count must have at least 2 dims, got shape {tuple(load.shape)}"
        )
    if load.dim() > 2:
        load = load.reshape(-1, *load.shape[-2:]).sum(dim=0)
    if load.shape[1] != num_logical_experts:
        raise ValueError(
            f"logical_count has {load.shape[1]} experts but the model has "
            f"{num_logical_experts} logical experts"
        )
    if (load < 0).any():
        raise ValueError("logical_count must be non-negative")
    load = _strip_non_moe_layers(load, num_moe_layers, "logical_count")
    if not load.any():
        raise ValueError("logical_count is all zeros")
    return load


def _validate_physical_to_logical(
    mapping: torch.Tensor,
    num_moe_layers: int,
    num_physical_experts: int,
    num_logical_experts: int,
) -> torch.Tensor:
    mapping = torch.as_tensor(mapping).to(torch.int64)
    expected = (num_moe_layers, num_physical_experts)
    if tuple(mapping.shape) != expected:
        raise ValueError(
            f"physical_to_logical_map must have shape {expected}, "
            f"got {tuple(mapping.shape)}"
        )
    if ((mapping < 0) | (mapping >= num_logical_experts)).any():
        raise ValueError("physical_to_logical_map has out-of-range expert ids")
    covered = torch.zeros(num_moe_layers, num_logical_experts, dtype=torch.bool)
    covered.scatter_(1, mapping, True)
    if not covered.all():
        raise ValueError(
            "physical_to_logical_map leaves some logical experts without a replica"
        )
    return mapping


def max_rank_load_ratio(
    physical_to_logical_map: torch.Tensor, logical_load: torch.Tensor, ep_size: int
) -> float:
    """Summed over layers: max rank load / mean rank load (1.0 is perfect)."""

    mapping = physical_to_logical_map.to(torch.long)
    load = logical_load.to(torch.float64)
    replicas = torch.zeros_like(load).scatter_add_(
        1, mapping, torch.ones_like(mapping, dtype=load.dtype)
    )
    per_replica = torch.gather(load / replicas.clamp_min(1), 1, mapping)
    per_rank_per_layer = per_replica.reshape(mapping.shape[0], ep_size, -1).sum(-1)
    # MoE layers are serialized by the combine barrier, so the cost is the sum
    # over layers of the slowest rank, not the slowest rank's total.
    layer_max = per_rank_per_layer.max(dim=1).values.sum()
    layer_mean = per_rank_per_layer.mean(dim=1).sum()
    return float(layer_max / layer_mean) if layer_mean > 0 else 1.0


def apply_initial_expert_location(state: Any, model_state: Any, location: str) -> None:
    """Rearrange one registered model to the placement given by ``location``.

    Must be called collectively on every EP rank, after ``add_model`` and
    before the async EPLB worker starts.
    """

    from vllm.distributed.eplb import eplb_state as eplb_state_module
    from vllm.distributed.eplb.rebalance_execute import (
        rearrange_expert_weights_inplace,
    )
    from vllm.distributed.parallel_state import get_ep_group, get_node_count

    model = model_state.model
    ep_group = get_ep_group().device_group
    ep_size = ep_group.size()
    is_main_rank = ep_group.rank() == 0
    old_map = model_state.physical_to_logical_map.cpu()

    payload = load_expert_location_payload(location)
    logical_load: torch.Tensor | None = None
    if "physical_to_logical_map" in payload:
        new_map = _validate_physical_to_logical(
            payload["physical_to_logical_map"],
            model.num_moe_layers,
            model.num_physical_experts,
            model.num_logical_experts,
        )
        if "logical_count" in payload:
            logical_load = logical_load_from_payload(
                payload, model.num_moe_layers, model.num_logical_experts
            )
    else:
        logical_load = logical_load_from_payload(
            payload, model.num_moe_layers, model.num_logical_experts
        )
        num_nodes = get_node_count()
        if ep_size % num_nodes != 0:
            num_nodes = 1
        new_map = state.policy.rebalance_experts(
            logical_load,
            model.num_physical_experts,
            model.num_expert_groups,
            num_nodes,
            ep_size,
            old_map,
        ).to(torch.int64)

    if is_main_rank:
        if logical_load is not None:
            logger.info(
                "init_expert_location %s: predicted MoE rank imbalance "
                "(sum of per-layer max/mean) %.3f -> %.3f",
                location,
                max_rank_load_ratio(old_map, logical_load, ep_size),
                max_rank_load_ratio(new_map, logical_load, ep_size),
            )
        else:
            logger.info("init_expert_location %s: applying explicit mapping", location)

    start = time.perf_counter()
    rearrange_expert_weights_inplace(
        old_map,
        new_map,
        model.expert_weights,
        model_state.expert_buffer,
        ep_group,
        model_state.communicator,
        False,
        None,
    )
    eplb_state_module._commit_eplb_maps(model_state, new_physical_to_logical_map=new_map)
    torch.accelerator.synchronize()
    if is_main_rank:
        logger.info(
            "init_expert_location: rearranged experts of %s in %.2f s",
            model_state.model_name,
            time.perf_counter() - start,
        )


__all__ = [
    "apply_initial_expert_location",
    "load_expert_location_payload",
    "logical_load_from_payload",
    "max_rank_load_ratio",
]
