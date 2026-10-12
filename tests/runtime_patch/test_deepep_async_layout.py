"""CPU contracts for the DeepEP layout -> dispatch -> receiver event chain."""
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.fused_moe.deepep_runtime import ht_do_dispatch
from vllm_hcu.platforms import envs as henvs


@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("async_prepare", [False, True])
@pytest.mark.parametrize("dbo", [False, True])
def test_ht_layout_dependency_and_receiver(monkeypatch, master, enabled,
                                           async_prepare, dbo):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", str(int(master)))
    monkeypatch.setattr(henvs, "VLLM_HCU_DEEPEP_HT_ASYNC_LAYOUT", enabled)
    compute_event, layout_event, result_event = object(), object(), object()
    calls = []
    class Buffer:
        capture = object()
        def get_dispatch_layout(self, **kwargs):
            assert kwargs["previous_event"] is compute_event
            calls.append(("layout", kwargs))
            return None, None, None, None, layout_event
        def dispatch(self, **kwargs):
            calls.append(("dispatch", kwargs))
            return kwargs["x"], kwargs["topk_idx"], kwargs["topk_weights"], [1], "handle", result_event
    instance = SimpleNamespace(
        buffer=Buffer(), async_prepare=async_prepare, handles={},
        _get_dispatch_config=lambda: "config",
        _receiver=lambda *args, **kwargs: (args, kwargs),
    )
    module = SimpleNamespace(
        dbo_get_previous_event=lambda capture: compute_event,
        dbo_yield_and_switch_from_compute_to_comm=lambda: None,
        dbo_enabled=lambda: dbo, dbo_current_ubatch_id=lambda: 0,
        dbo_switch_to_compute_sync=lambda: None,
    )
    config = SimpleNamespace(use_int8_w8a8=False, use_fp8_w8a8=False,
                             is_per_act_token=True, is_block_quantized=False)
    ids = torch.zeros(2, 1, dtype=torch.int64)
    receiver = ht_do_dispatch(module, instance, torch.ones(2, 128), None,
                              ids, torch.ones(2, 1), 1, None, config, False)
    asynchronous = master and enabled and async_prepare and not dbo
    assert calls[0][1]["async_finish"] == asynchronous
    assert calls[1][1]["previous_event"] is (layout_event if asynchronous else compute_event)
    assert calls[1][1]["async_finish"] == (async_prepare and not dbo)
    assert instance.handles[0] == "handle"
    args, kwargs = receiver()
    assert args[0] is result_event  # receiver still joins dispatch completion
    assert kwargs["defer_input_quant"] is False
