# SPDX-License-Identifier: Apache-2.0
"""CPU-only regressions, including asymmetric hosts and storage-only overlap."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("hcu_auto_network_test_target", ROOT / "vllm_hcu/distributed/auto_network.py")
n = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = n
SPEC.loader.exec_module(n)


def host(name="a", numa_domains=4, names=None):
    names = names or [f"mlx5_{i}" for i in range(10)]
    pairs = [0, 0, 1, 1, 2, 2, 3, 3]
    gpus = [{"bus": f"0000:{i + 32:02x}:00.0", "numa": pair if numa_domains == 4 else pair // 2,
             "path": f"/sys/devices/root{pair if i % 2 == 0 else 20 + i}/gpu{i}"}
            for i, pair in enumerate(pairs)]
    ports = []
    for index in range(10):
        if index == 7:
            continue
        rail = index if index < 6 else index - 2
        pair = rail // 2
        ports.append({"nic": names[index], "index": index, "port": 1, "gid": 3, "kind": "roce",
                      "fabric": f"roce:13.13.{rail + 1}.0/24" if index != 6 else "roce:12.12.0.0/16",
                      "numa": (pair if numa_domains == 4 else pair // 2) if index != 6 else 1,
                      "path": f"/sys/devices/root{pair}/nic" if index != 6 else "/sys/devices/storage/nic",
                      "speed": 200})
    return {"id": name, "gpus": gpus, "ports": ports, "names": names,
            "slots": list(range(8)), "deepep": True, "layered_capable": True, "ipc_capable": True}


class PlanTests(unittest.TestCase):
    def tearDown(self):
        n._OWNED_ENV.clear()

    def test_two_and_four_numa_domains_use_eight_business_ports(self):
        plans = n.build_plans([host("two", 2), host("four", 4)])
        expected = ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4", "mlx5_5", "mlx5_8", "mlx5_9"]
        for plan in plans.values():
            self.assertEqual([plan["mapping"][i]["nic"] for i in range(8)], expected)
            self.assertEqual(plan["env"]["ROCSHMEM_ALLOWED_IBV_DEVICES"], ",".join(expected))
            self.assertEqual(plan["env"]["ROCSHMEM_GDA_RAIL_ONLY"], "1")
            self.assertEqual(plan["topo"].splitlines()[-1].split()[2], "7")

    def test_names_are_local_not_hardcoded(self):
        plans = n.build_plans([host(), host("b", names=[f"device{i}" for i in range(10)])])
        self.assertIn("device8:1", plans["b"]["env"]["NCCL_IB_HCA"])
        self.assertEqual(plans["a"]["mapping"][7]["fabric"], plans["b"]["mapping"][7]["fabric"])

    def test_only_common_storage_network_is_selected_on_asymmetric_pair(self):
        a, b = host(), host("b")
        b["ports"] = [dict(b["ports"][6], nic="only_active", index=0)]
        b["names"] = ["only_active"]
        plans = n.build_plans([a, b])
        self.assertEqual(plans["a"]["env"]["NCCL_IB_HCA"], "=mlx5_6:1")
        self.assertEqual(plans["b"]["env"]["NCCL_IB_HCA"], "=only_active:1")
        self.assertNotIn("ROCSHMEM_GDA_RAIL_ONLY", plans["a"]["env"])

    def test_equal_speed_storage_port_does_not_replace_compute_port(self):
        plans = n.build_plans([host("a", 2), host("b", 2)])
        for plan in plans.values():
            self.assertNotIn("mlx5_6", plan["env"]["ROCSHMEM_ALLOWED_IBV_DEVICES"])
            self.assertEqual(plan["mapping"][7]["nic"], "mlx5_9")

    def test_ipv6_only_roce_selects_matching_nccl_family(self):
        a, b = host(), host("b")
        for h in (a, b):
            h["ports"] = [dict(h["ports"][0], fabric="roce:2001:db8::/64", gid=5)]
        plan = n.build_plans([a, b])["a"]
        self.assertEqual(plan["env"]["NCCL_IB_ADDR_FAMILY"], "AF_INET6")
        self.assertEqual(plan["mapping"][0]["gid"], 5)

    def test_same_nic_names_do_not_imply_same_network(self):
        a, b = host(), host("b")
        for port in b["ports"]:
            port["fabric"] = port["fabric"].replace("13.13", "14.14").replace("12.12", "15.15")
        with self.assertRaisesRegex(n.NetworkConfigurationError, "no common"):
            n.build_plans([a, b])

    def test_reversed_nic_order_cannot_silently_cross_roce_rails(self):
        a, b = host(), host("b")
        b["ports"].reverse()
        b["names"].reverse()
        plans = n.build_plans([a, b])
        for i in range(8):
            self.assertEqual(plans["a"]["mapping"][i]["fabric"], plans["b"]["mapping"][i]["fabric"])
        self.assertEqual(len(plans["a"]["common_fabrics"]), 1)

    def test_common_ib_fabric_keeps_multiple_nics_without_roce_layered_flags(self):
        a, b = host(), host("b")
        for h in (a, b):
            for port in h["ports"]:
                port.update(fabric="ib:fe80::/64", kind="ib", gid=0)
        plans = n.build_plans([a, b])
        self.assertNotIn("NCCL_CROSS_NIC", plans["a"]["env"])
        self.assertNotIn("ROCSHMEM_GDA_RAIL_ONLY", plans["a"]["env"])
        self.assertGreater(len(plans["a"]["env"]["NCCL_IB_HCA"].split(",")), 1)

    def test_unsupported_layered_binary_fails_before_collectives(self):
        a, b = host(), host("b")
        b["layered_capable"] = False
        with self.assertRaisesRegex(n.NetworkConfigurationError, "supporting"):
            n.build_plans([a, b])

    def test_missing_local_peer_access_does_not_enable_rail_only(self):
        a, b = host(), host("b")
        b["ipc_capable"] = False
        with self.assertRaisesRegex(n.NetworkConfigurationError, "peer access"):
            n.build_plans([a, b])

    def test_unused_visible_gpus_are_included_in_topology(self):
        a, b = host(), host("b")
        a["deepep"] = b["deepep"] = False
        a["slots"] = b["slots"] = [0, 1]
        plans = n.build_plans([a, b])
        self.assertEqual(len(plans["a"]["topo"].splitlines()), 8)
        self.assertNotIn("ROCSHMEM_ALLOWED_IBV_DEVICES", plans["a"]["env"])

    def test_nccl_only_can_use_socket_when_no_common_rdma_exists(self):
        a, b = host(), host("b")
        a["deepep"] = b["deepep"] = False
        a["ports"] = []
        plans = n.build_plans([a, b])
        self.assertEqual(plans["a"]["env"], {"NCCL_IB_DISABLE": "1"})

    def test_explicit_socket_interface_is_not_overwritten(self):
        plan = n.build_plans([host(), host("b")])["a"]
        env = {"NCCL_SOCKET_IFNAME": "=custom0"}
        with patch.object(n, "_route_interface", return_value="mgmt0"):
            applied = n.apply_plan(plan, 7, "master", env)
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "=custom0")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "mgmt0")
        self.assertTrue(Path(applied["ROCSHMEM_TOPO_FILE_FORCE"]).is_file())

    def test_discovery_excludes_down_ports_and_uses_verbs_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gpu = root / "bus/pci/devices/gpu0"
            gpu.mkdir(parents=True)
            (gpu / "numa_node").write_text("0")
            for nic, state in (("net9", "1: DOWN"), ("net2", "4: ACTIVE")):
                device = root / "class/infiniband" / nic
                pci = device / "device"
                pci.mkdir(parents=True)
                (pci / "numa_node").write_text("0")
                port = device / "ports/1"
                (port / "gids").mkdir(parents=True)
                (port / "state").write_text(state)
                (port / "rate").write_text("200 Gb/sec")
                (port / "gids/3").write_text("::ffff:10.0.0.1")
            with patch.object(n, "_gid_fabric", return_value=("roce:10.0.0.0/24", "roce")):
                result = n.discover_host(["gpu0"], ["net9", "net2"], root)
            self.assertEqual(len(result["ports"]), 1)
            self.assertEqual(result["ports"][0]["nic"], "net2")
            self.assertEqual(result["ports"][0]["index"], 1)

    def test_gid0_link_local_roce_is_not_treated_as_shared_fabric(self):
        with tempfile.TemporaryDirectory() as tmp:
            port = Path(tmp)
            (port / "gids").mkdir()
            (port / "gids/0").write_text("fe80::1")
            (port / "link_layer").write_text("Ethernet")
            (port / "gid_attrs/types").mkdir(parents=True)
            (port / "gid_attrs/types/0").write_text("IB/RoCE v1")
            self.assertIsNone(n._gid_fabric(port, 0))


class OverrideTests(unittest.TestCase):
    def tearDown(self):
        n._OWNED_ENV.clear()

    def test_complete_nccl_socket_preflight_without_rdma(self):
        metadata = {key: value for key, value in host().items() if key in {"gpus", "ports", "names", "ipc_capable"}}
        metadata["ports"] = []
        peer = dict(metadata, id="remote", rank=1, local_rank=0, manual=[], disabled=False,
                    deepep=False, layered_capable=False)
        data = {"1": json.dumps(peer).encode()}
        store = types.SimpleNamespace(set=lambda k, v: data.__setitem__(k, v.encode()), get=lambda k: data[k])
        dist = types.ModuleType("torch.distributed")
        dist.rendezvous = lambda *a, **kw: iter([(store, 0, 2)])
        dist.PrefixStore = lambda name, value: value
        torch = types.ModuleType("torch")
        torch.distributed = dist
        parallel = types.SimpleNamespace(nnodes=2, enable_elastic_ep=False, distributed_executor_backend="external_launcher",
                                         master_addr="localhost", distributed_timeout_seconds=None,
                                         enable_expert_parallel=False, all2all_backend="allgather_reducescatter")
        with patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}), \
                patch.dict(os.environ, {}, clear=True), \
                patch.object(n, "discover_host", return_value=metadata), \
                patch.object(n, "_route_interface", return_value=None):
            n.prepare_network(types.SimpleNamespace(parallel_config=parallel), 2, 0, "tcp://localhost:1", 0)
            self.assertEqual(os.environ["NCCL_IB_DISABLE"], "1")

    def test_any_manual_peer_makes_the_entire_cohort_preserve_environment(self):
        peer = {"id": "remote", "rank": 1, "local_rank": 0, "manual": ["NCCL_IB_HCA"],
                "disabled": False, "deepep": True}
        store = types.SimpleNamespace()
        data = {"1": json.dumps(peer).encode()}
        store.set = lambda key, value: data.__setitem__(key, value.encode())
        store.get = lambda key: data[key]
        dist = types.ModuleType("torch.distributed")
        dist.rendezvous = lambda *a, **kw: iter([(store, 0, 2)])
        dist.PrefixStore = lambda name, value: value
        torch = types.ModuleType("torch")
        torch.distributed = dist
        p = types.SimpleNamespace(nnodes=2, enable_elastic_ep=False, distributed_executor_backend="mp",
                                 data_parallel_rank=0, world_size_across_dp=2, master_addr="localhost",
                                 master_port=12345, distributed_timeout_seconds=None,
                                 enable_expert_parallel=True, all2all_backend="deepep_auto")
        with patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}), \
                patch.dict(os.environ, {"UNRELATED_SECRET": "not-for-exchange"}, clear=True), \
                patch.object(n, "discover_host", return_value=host()), \
                patch.object(n, "layered_capable", return_value=True), \
                patch.object(n, "build_plans") as planner:
            n.prepare_network(types.SimpleNamespace(parallel_config=p), 2, 0, "tcp://localhost:12345", 0)
            planner.assert_not_called()
            self.assertNotIn("NCCL_IB_HCA", os.environ)
            self.assertNotIn("not-for-exchange", data["0"].decode())


if __name__ == "__main__":
    unittest.main()
