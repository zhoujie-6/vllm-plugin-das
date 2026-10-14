# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.distributed import eplb_init_location as init_location
from vllm_hcu.patch.config import HcuFeatureConfig
from vllm_hcu.patch.worker.framework_opt import patch_eplb_init_location

_SGLANG_DUMP = Path(
    "/home/zhoujie6/dpsk_vllm_sglang/"
    "expert_distribution_recorder_1781606718.3198578.pt"
)


def test_logical_count_sums_steps_and_strips_dense_layers(tmp_path) -> None:
    counts = torch.zeros(3, 4, 8, dtype=torch.int32)
    counts[:, 1:, :] = torch.arange(8, dtype=torch.int32)
    path = tmp_path / "dump.pt"
    torch.save({"rank": 0, "logical_count": counts}, path)

    payload = init_location.load_expert_location_payload(str(path))
    load = init_location.logical_load_from_payload(payload, 3, 8)

    assert load.dtype == torch.int64
    assert load.shape == (3, 8)
    assert torch.equal(load[0], 3 * torch.arange(8))


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"logical_count": torch.ones(5, 8)}, "covers 5 layers"),
        ({"logical_count": torch.ones(3, 6)}, "6 experts"),
        ({"logical_count": torch.zeros(3, 8)}, "all zeros"),
        ({"logical_count": -torch.ones(3, 8)}, "non-negative"),
    ],
)
def test_logical_count_rejects_mismatches(payload, match) -> None:
    with pytest.raises(ValueError, match=match):
        init_location.logical_load_from_payload(payload, 3, 8)


def test_payload_formats(tmp_path) -> None:
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"physical_to_logical_map": [[0, 1]]}))
    assert "physical_to_logical_map" in init_location.load_expert_location_payload(
        str(path)
    )
    inline = init_location.load_expert_location_payload('{"logical_count": [[1]]}')
    assert inline["logical_count"] == [[1]]
    with pytest.raises(ValueError, match="neither"):
        init_location.load_expert_location_payload('{"rank": 0}')


def test_explicit_mapping_validation() -> None:
    good = torch.tensor([[0, 1, 2, 3, 0, 1]])
    assert torch.equal(
        init_location._validate_physical_to_logical(good, 1, 6, 4), good
    )
    with pytest.raises(ValueError, match="without a replica"):
        init_location._validate_physical_to_logical(
            torch.tensor([[0, 1, 2, 2, 0, 1]]), 1, 6, 4
        )
    with pytest.raises(ValueError, match="out-of-range"):
        init_location._validate_physical_to_logical(
            torch.tensor([[0, 1, 2, 3, 4, 1]]), 1, 6, 4
        )
    with pytest.raises(ValueError, match="shape"):
        init_location._validate_physical_to_logical(good, 2, 6, 4)


def test_max_rank_load_ratio_splits_replica_load() -> None:
    load = torch.tensor([[6, 2]])
    # rank0 holds expert 0, rank1 holds expert 1.
    assert init_location.max_rank_load_ratio(torch.tensor([[0, 1]]), load, 2) == 1.5
    # Both experts replicated on both ranks: 3+1 vs 3+1.
    assert init_location.max_rank_load_ratio(
        torch.tensor([[0, 1, 0, 1]]), load, 2
    ) == 1.0


@pytest.mark.skipif(not _SGLANG_DUMP.exists(), reason="SGLang dump not present")
def test_sglang_dump_balances_with_vllm_default_policy() -> None:
    from vllm.distributed.eplb.eplb_state import EplbState
    from vllm.distributed.eplb.policy.default import DefaultEplbPolicy

    payload = init_location.load_expert_location_payload(str(_SGLANG_DUMP))
    load = init_location.logical_load_from_payload(payload, 43, 256)
    trivial = torch.tensor(
        EplbState.build_initial_global_physical_to_logical_map(256, 64)
    ).expand(43, -1)
    new_map = DefaultEplbPolicy.rebalance_experts(load, 320, 1, 1, 8, trivial)

    assert new_map.shape == (43, 320)
    before = init_location.max_rank_load_ratio(trivial, load, 8)
    after = init_location.max_rank_load_ratio(new_map, load, 8)
    assert after < 1.1 < before


def _fake_eplb_module(calls: list) -> ModuleType:
    module = ModuleType(patch_eplb_init_location.TARGET_MODULE)

    class EplbState:
        def __init__(self) -> None:
            self.model_states = {}

        def add_model(self, model, model_config):
            calls.append(("add", model))
            self.model_states[model_config.compute_hash()] = model

    module.EplbState = EplbState
    module._commit_eplb_maps = lambda *args, **kwargs: None
    return module


def _model_config(name: str) -> SimpleNamespace:
    return SimpleNamespace(compute_hash=lambda: name)


@pytest.mark.parametrize("location", [None, "/tmp/dump.pt"])
def test_adapter_applies_only_to_main_model(monkeypatch, location) -> None:
    calls: list = []
    module = _fake_eplb_module(calls)
    assert patch_eplb_init_location.apply_to_module(module) is True
    assert patch_eplb_init_location.apply_to_module(module) is False

    main = _model_config("main")
    vllm_config = SimpleNamespace(
        model_config=main,
        additional_config={
            "hcu": HcuFeatureConfig(init_expert_location=location).to_dict()
        },
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config", lambda: vllm_config
    )
    monkeypatch.setattr(
        init_location,
        "apply_initial_expert_location",
        lambda state, model_state, loc: calls.append(("apply", model_state, loc)),
    )

    state = module.EplbState()
    state.add_model("draft", _model_config("draft"))
    state.add_model("target", main)

    expected = [("add", "draft"), ("add", "target")]
    if location is not None:
        expected.append(("apply", "target", location))
    assert calls == expected
