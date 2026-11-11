# SPDX-License-Identifier: Apache-2.0
"""Real TCPStore/default-PG reuse, using CPU Gloo and synthetic host metadata."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import socket
import sys
import types
import unittest


def worker(rank, port, manual_peer):
    import torch
    import torch.distributed as dist

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("hcu_autonet_rendezvous_target", root / "vllm_hcu/distributed/auto_network.py")
    network = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = network
    spec.loader.exec_module(network)
    for key in network._MANUAL:
        os.environ.pop(key, None)
    os.environ.pop("VLLM_HCU_AUTO_NETWORK", None)
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    read = network._read
    network._read = lambda path, default="": f"host-{rank}" if path.name == "boot_id" else read(path, default)
    network.discover_host = lambda: {
        "gpus": [{"bus": "0000:01:00.0", "numa": 0, "path": "/sys/devices/pci/gpu"}],
        "names": [f"fake_nic_{rank}"], "ipc_capable": False,
        "ports": [{"nic": f"fake_nic_{rank}", "index": 0, "port": 1, "gid": 3,
                   "kind": "roce", "fabric": "roce:10.0.0.0/24", "numa": 0,
                   "path": "/sys/devices/pci/nic", "speed": 100}]}
    if manual_peer and rank == 1:
        os.environ["NCCL_IB_HCA"] = "=user_nic"
    parallel = types.SimpleNamespace(
        nnodes=2, enable_elastic_ep=False, distributed_executor_backend="external_launcher",
        master_addr="127.0.0.1", master_port=port, distributed_timeout_seconds=30,
        enable_expert_parallel=False, all2all_backend="allgather_reducescatter")
    init = f"tcp://127.0.0.1:{port}"
    lifetime = network.prepare_network(types.SimpleNamespace(parallel_config=parallel), 2, rank, init, 0)
    if manual_peer:
        assert os.environ.get("NCCL_IB_HCA") == ("=user_nic" if rank == 1 else None)
    else:
        assert os.environ["NCCL_IB_HCA"] == f"=fake_nic_{rank}:1"
    # Rank zero must reuse the preflight's server instead of failing EADDRINUSE.
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2)
    del lifetime
    value = torch.tensor([rank + 1.0])
    dist.all_reduce(value)
    assert value.item() == 3.0
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for CPU distributed tests")
class RendezvousTests(unittest.TestCase):
    def run_pair(self, manual):
        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(worker, args=(port, manual), nprocs=2, join=True)

    def test_default_group_reuses_negotiation_store(self):
        self.run_pair(False)

    def test_one_manual_peer_preserves_both_environments(self):
        self.run_pair(True)


if __name__ == "__main__":
    unittest.main()
