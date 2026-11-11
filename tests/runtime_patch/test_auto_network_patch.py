# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

FOLDER = Path(__file__).resolve().parents[2] / "vllm_hcu/patch/platform/framework_opt"
PACKAGE = "hcu_network_adapter_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(FOLDER)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + ".patch_auto_network", FOLDER / "patch_auto_network.py")
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)


class PatchTests(unittest.TestCase):
    def target(self, trace=None):
        module = types.ModuleType(adapter.TARGET_MODULE)
        def initialize(world_size=-1, rank=-1, distributed_init_method="env://", local_rank=-1,
                       backend="nccl", timeout=None):
            if trace is not None:
                trace.append(("upstream", world_size, rank, distributed_init_method, local_rank, backend))
            return "upstream_result"
        module.init_distributed_environment = initialize
        return module

    def test_idempotence_and_stale_marker(self):
        module = self.target()
        self.assertTrue(adapter.apply_to_module(module))
        self.assertFalse(adapter.apply_to_module(module))
        module.init_distributed_environment = lambda: None
        with self.assertRaisesRegex(RuntimeError, "stale"):
            adapter.apply_to_module(module)

    def test_changed_upstream_signature_fails_before_patching(self):
        module = self.target()
        original = lambda wrong: None
        module.init_distributed_environment = original
        with self.assertRaisesRegex(RuntimeError, "signature"):
            adapter.apply_to_module(module)
        self.assertIs(module.init_distributed_environment, original)

    def call_initialization(self, backend):
        trace = []
        module = self.target(trace)
        network = types.ModuleType("vllm_hcu.distributed.auto_network")
        network.prepare_network = lambda *args: trace.append(("preflight",) + args[1:])
        config = types.ModuleType("vllm.config")
        config.get_current_vllm_config_or_none = lambda: types.SimpleNamespace(parallel_config=types.SimpleNamespace(nnodes=2))
        vllm = types.ModuleType("vllm")
        vllm.config = config
        torch = types.ModuleType("torch")
        dist = types.ModuleType("torch.distributed")
        dist.is_initialized = lambda: False
        torch.distributed = dist
        modules = {"vllm": vllm, "vllm.config": config, "torch": torch,
                   "torch.distributed": dist, "vllm_hcu.distributed.auto_network": network}
        with patch.dict(sys.modules, modules):
            adapter.apply_to_module(module)
            result = module.init_distributed_environment(4, 1, "tcp://master:1", 3, backend=backend)
        self.assertEqual(result, "upstream_result")
        return trace

    def test_preflight_precedes_upstream_and_preserves_arguments(self):
        trace = self.call_initialization("nccl")
        self.assertEqual(trace, [("preflight", 4, 1, "tcp://master:1", 3),
                                 ("upstream", 4, 1, "tcp://master:1", 3, "nccl")])

    def test_cpu_coordinator_is_not_reconfigured(self):
        self.assertEqual(self.call_initialization("gloo"),
                         [("upstream", 4, 1, "tcp://master:1", 3, "gloo")])

    def test_late_gpu_worker_alias_is_updated(self):
        module = self.target()
        worker = types.ModuleType("vllm.v1.worker.gpu_worker")
        worker.init_distributed_environment = module.init_distributed_environment
        with patch.dict(sys.modules, {worker.__name__: worker}):
            adapter.apply_to_module(module)
            self.assertIs(worker.init_distributed_environment, module.init_distributed_environment)


if __name__ == "__main__":
    unittest.main()
