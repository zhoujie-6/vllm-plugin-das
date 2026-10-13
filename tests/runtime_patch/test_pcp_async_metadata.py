# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_hcu.v1 import pcp_manager as runtime


def manager(device="cuda"):
    return SimpleNamespace(device=device, _metadata_upload_sources=[])


@pytest.mark.parametrize("enabled", [True, False])
def test_cpu_metadata_values_dtype_and_empty(monkeypatch, enabled):
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_PCP_ASYNC_METADATA", enabled)
    m = manager("cpu")
    for array in [np.arange(7, dtype=np.int64), np.array([True, False]),
                  np.array([], dtype=np.int64)]:
        output = runtime.HcuPCPManager._upload_metadata(m, array)
        torch.testing.assert_close(output, torch.from_numpy(array))
    assert not m._metadata_upload_sources


@pytest.mark.parametrize("child,master,expected_async", [(True, True, True),
                                                        (False, True, False),
                                                        (True, False, False)])
def test_cuda_metadata_snapshot_and_rollback(monkeypatch, child, master, expected_async):
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_PCP_ASYNC_METADATA", child)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1" if master else "0")
    calls = []
    # Use real CPU tensors, recording CUDA transfer requests instead of launching DMA.
    def pin(self):
        calls.append("pin")
        return self.clone()
    def transfer(self, device, **kwargs):
        calls.append((device, kwargs))
        return self.clone()
    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    monkeypatch.setattr(torch.Tensor, "to", transfer)
    m = manager()
    source = np.arange(8, dtype=np.int64)
    output = runtime.HcuPCPManager._upload_metadata(m, source)
    source[:] = -1
    assert torch.equal(output, torch.arange(8))
    if expected_async:
        assert calls == ["pin", ("cuda", {"non_blocking": True})]
        assert len(m._metadata_upload_sources) == 1
        assert torch.equal(m._metadata_upload_sources[0], torch.arange(8))
    else:
        assert calls == [("cuda", {})]
        assert not m._metadata_upload_sources


def test_local_query_rows_reuses_uploaded_snapshot(monkeypatch):
    monkeypatch.setattr(runtime.henvs, "VLLM_HCU_PCP_ASYNC_METADATA", True)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    transfers = []
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self.clone())
    def transfer(self, device, **kwargs):
        transfers.append(kwargs)
        return self.clone()
    monkeypatch.setattr(torch.Tensor, "to", transfer)
    m = manager()
    m._local_token_indices_np = np.array([0, 2, 3], dtype=np.int64)
    m._local_real_positions_np = np.array([0, 1, 4], dtype=np.int64)
    m._local_query_rows_cache = {}
    m._upload_metadata = lambda array: runtime.HcuPCPManager._upload_metadata(m, array)
    first = runtime.HcuPCPManager.local_query_rows(m, 0, 3)
    assert torch.equal(first[0], torch.tensor([0, 2]))
    assert torch.equal(first[1], torch.tensor([0, 1]))
    m._local_token_indices_np[:] = -1
    second = runtime.HcuPCPManager.local_query_rows(m, 0, 3)
    assert second is first
    assert transfers == [{"non_blocking": True}] * 2
    assert len(m._metadata_upload_sources) == 2
