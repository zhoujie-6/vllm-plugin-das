# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Negotiate HCU transport defaults before NCCL or RocSHMEM initializes.

Discovery is CPU-only. No model, collective, IP address or interface is
hardcoded. Explicit transport settings make the whole launch opt out, so a
partially customized deployment is never silently combined with an auto plan.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
import importlib.util
import ipaddress
import json
import logging
import mmap
import os
from pathlib import Path
import re
import socket
import struct
import tempfile
from datetime import timedelta

logger = logging.getLogger(__name__)
_TOPO_DIRS: list[tempfile.TemporaryDirectory] = []
_OWNED_ENV: dict[str, str] = {}
_MANUAL = (
    "ROCSHMEM_TOPO_FILE_FORCE", "ROCSHMEM_ALLOWED_IBV_DEVICES",
    "ROCSHMEM_IB_GID_INDEX", "ROCSHMEM_IB_PORT", "ROCSHMEM_REQUESTED_DEV",
    "ROCSHMEM_BACKEND", "ROCSHMEM_GDA_RAIL_ONLY", "ROCSHMEM_DISABLE_IPC",
    "ROCSHMEM_IPC_MNVL", "DEEPEP_ENABLE_LL_LAYERED_OPT",
    "DEEPEP_ENABLE_LL_LAYERED_DEVELOP", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
    "NCCL_IB_DISABLE", "NCCL_NET", "NCCL_CROSS_NIC", "NCCL_TOPO_FILE",
    "NCCL_NET_PLUGIN", "NCCL_IB_ADDR_FAMILY", "NCCL_IB_ADDR_RANGE",
    "NCCL_IB_ROCE_VERSION_NUM", "NCCL_IB_MERGE_NICS", "ROCSHMEM_DISABLE_AUTOTOPO",
)


class NetworkConfigurationError(RuntimeError):
    pass


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return default


@functools.lru_cache(maxsize=1)
def _hip_library():
    for name in ("amdhip64", "galaxyhip"):
        library = ctypes.util.find_library(name)
        if library:
            hip = ctypes.CDLL(library)
            return hip
    else:
        raise NetworkConfigurationError("HIP runtime library was not found")


def gpu_bus_ids() -> list[str]:
    """HIP enumeration respects the worker's visible-device mask."""
    hip = _hip_library()
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    count = ctypes.c_int()
    if hip.hipGetDeviceCount(ctypes.byref(count)) != 0:
        raise NetworkConfigurationError("HIP device enumeration failed")
    result = []
    for index in range(count.value):
        buffer = ctypes.create_string_buffer(64)
        if hip.hipDeviceGetPCIBusId(buffer, len(buffer), index) != 0:
            raise NetworkConfigurationError(f"Cannot read PCI address of GPU {index}")
        result.append(buffer.value.decode().lower())
    return result


def _full_peer_access(count: int) -> bool:
    hip = _hip_library()
    hip.hipDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    for source in range(count):
        for target in range(count):
            if source == target:
                continue
            access = ctypes.c_int()
            if hip.hipDeviceCanAccessPeer(ctypes.byref(access), source, target) or not access.value:
                return False
    return True


def verbs_device_names() -> list[str]:
    library = ctypes.util.find_library("ibverbs")
    if library is None:
        return []
    verbs = ctypes.CDLL(library)
    verbs.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    verbs.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    verbs.ibv_get_device_name.argtypes = [ctypes.c_void_p]
    verbs.ibv_get_device_name.restype = ctypes.c_char_p
    verbs.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    count = ctypes.c_int()
    devices = verbs.ibv_get_device_list(ctypes.byref(count))
    if not devices:
        return []
    try:
        return [verbs.ibv_get_device_name(devices[i]).decode() for i in range(count.value)]
    finally:
        verbs.ibv_free_device_list(devices)


def _ipv4_network(interface: str, address: str) -> str | None:
    import fcntl

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            data = fcntl.ioctl(sock.fileno(), 0x891B, struct.pack("256s", interface.encode()[:15]))
        mask = socket.inet_ntoa(data[20:24])
        return str(ipaddress.IPv4Network((address, mask), strict=False))
    except (OSError, ValueError):
        return None


def _gid_fabric(port: Path, index: int) -> tuple[str, str] | None:
    text = _read(port / "gids" / str(index))
    if not text:
        return None
    gid = ipaddress.IPv6Address(text)
    if int(gid) == 0:
        return None
    if _read(port / "link_layer") == "InfiniBand":
        return "ib:" + str(ipaddress.IPv6Network((gid, 64), strict=False)), "ib"
    # Link-local GIDs do not identify which independent RoCE rail they belong to.
    if _read(port / "gid_attrs/types" / str(index)) != "RoCE v2":
        return None
    interface = _read(port / "gid_attrs/ndevs" / str(index))
    if gid.ipv4_mapped:
        network = _ipv4_network(interface, str(gid.ipv4_mapped))
        return ("roce:" + network, "roce") if network else None
    if gid.is_link_local:
        return None
    for row in _read(Path("/proc/net/if_inet6")).splitlines():
        fields = row.split()
        if len(fields) == 6 and fields[-1] == interface and int(fields[0], 16) == int(gid):
            network = ipaddress.IPv6Network((gid, int(fields[2], 16)), strict=False)
            return "roce:" + str(network), "roce"
    return None


def discover_host(
    buses: list[str] | None = None,
    names: list[str] | None = None,
    sysfs: Path = Path("/sys"),
) -> dict:
    inspect_peers = buses is None
    buses = gpu_bus_ids() if buses is None else buses
    names = verbs_device_names() if names is None else names
    gpus = []
    for bus in buses:
        path = sysfs / "bus/pci/devices" / bus
        gpus.append({"bus": bus, "path": str(path.resolve()), "numa": int(_read(path / "numa_node", "-1"))})
    ports = []
    for index, name in enumerate(names):
        device = sysfs / "class/infiniband" / name
        pci = (device / "device").resolve()
        # RocSHMEM's current provider uses port 1. Do not silently select a port
        # that the provider cannot open through its GPU-to-NIC file format.
        port = device / "ports/1"
        if not _read(port / "state").startswith("4:"):
            continue
        gids = sorted((port / "gids").glob("*"), key=lambda p: int(p.name))
        choices = [(int(p.name), _gid_fabric(port, int(p.name))) for p in gids]
        choices = [(i, f) for i, f in choices if f]
        if not choices:
            continue
        # Prefer an IPv4 RoCEv2 primary address, matching NCCL's default family.
        choices.sort(key=lambda item: (":" in item[1][0].removeprefix("roce:"), item[0]))
        gid, (fabric, kind) = choices[0]
        rate = re.match(r"[0-9.]+", _read(port / "rate", "0"))
        ports.append({"nic": name, "index": index, "port": 1, "gid": gid,
                      "fabric": fabric, "kind": kind, "path": str(pci),
                      "numa": int(_read(pci / "numa_node", "-1")),
                      "speed": float(rate[0]) if rate else 0})
    if not gpus:
        raise NetworkConfigurationError("No HIP GPUs are visible")
    return {"gpus": gpus, "ports": ports, "names": names,
            "ipc_capable": _full_peer_access(len(gpus)) if inspect_peers else False}


def _common_order(left: list[str], right: list[str]) -> list[str]:
    """NCCL filters HCAs in verbs order, not in NCCL_IB_HCA string order."""
    table = [[[] for _ in range(len(right) + 1)] for _ in range(len(left) + 1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            table[i][j] = (table[i - 1][j - 1] + [a] if a == b else
                           max(table[i - 1][j], table[i][j - 1], key=len))
    return table[-1][-1]


def _shared_path(left: str, right: str) -> int:
    a, b = Path(left).parts, Path(right).parts
    shared = 0
    for x, y in zip(a, b):
        if x != y:
            break
        shared += 1
    return shared


def _score(gpu: dict, port: dict, used: dict) -> tuple:
    numa = int(gpu["numa"] < 0 or gpu["numa"] != port["numa"])
    return (numa, -_shared_path(gpu["path"], port["path"]), -port.get("gpu_proximity", 0),
            -port["speed"], used.get(port["nic"], 0), port["index"])


def build_plans(hosts: list[dict]) -> dict[str, dict]:
    """Choose common fabrics jointly; never infer compatibility from NIC names."""
    sequences = []
    for host in hosts:
        for port in host["ports"]:
            port["gpu_proximity"] = max(_shared_path(gpu["path"], port["path"]) for gpu in host["gpus"])
        sequences.append(list(dict.fromkeys(p["fabric"] for p in host["ports"])))
    shared_fabrics = set.intersection(*(set(s) for s in sequences))
    common = sequences[0]
    for sequence in sequences[1:]:
        common = _common_order(common, sequence)
    if not common:
        if not any(host["deepep"] for host in hosts):
            logger.warning("No common RDMA fabric: NCCL will use its Socket transport")
            return {host["id"]: {"env": {"NCCL_IB_DISABLE": "1"}, "topo": "",
                    "mapping": {slot: {"nic": "Socket", "gid": None} for slot in host["slots"]},
                    "restricted": False, "common_fabrics": []} for host in hosts}
        raise NetworkConfigurationError(
            "Nodes have no common ordered RDMA fabric. Check interface addresses/routes, "
            "or supply explicit transport settings; auto configuration will not guess a route.")
    ib = [fabric for fabric in common if fabric.startswith("ib:")]
    if ib:
        common = ib  # Do not mix a storage RoCE network into a common IB fabric.
    else:
        ipv4 = [f for f in common if ipaddress.ip_network(f.removeprefix("roce:")).version == 4]
        if ipv4:
            common = ipv4  # A process has one NCCL GID address-family setting.
    slots = [host["slots"] for host in hosts]
    if any(len(s) != len(slots[0]) for s in slots):
        raise NetworkConfigurationError("Automatic multi-node transport requires equal workers per node")
    used = {host["id"]: {} for host in hosts}
    assignments = {host["id"]: {} for host in hosts}
    for position in range(len(slots[0])):
        choices = []
        for fabric in common:
            ports = []
            for host in hosts:
                slot = host["slots"][position]
                candidates = [p for p in host["ports"] if p["fabric"] == fabric]
                ports.append(min(candidates, key=lambda p: _score(host["gpus"][slot], p, used[host["id"]])))
            scores = [_score(h["gpus"][h["slots"][position]], p, used[h["id"]]) for h, p in zip(hosts, ports)]
            aggregate = tuple(sum(s[i] for s in scores) for i in range(len(scores[0]) - 1))
            choices.append((aggregate, common.index(fabric), ports))
        _, _, selected = min(choices, key=lambda item: item[:2])
        for host, port in zip(hosts, selected):
            assignments[host["id"]][host["slots"][position]] = port
            counts = used[host["id"]]
            counts[port["nic"]] = counts.get(port["nic"], 0) + 1
    selected_fabrics = {p["fabric"] for mapping in assignments.values() for p in mapping.values()}
    restricted = len(selected_fabrics) > 1 and not ib
    if restricted and any(host["deepep"] for host in hosts):
        if len(slots[0]) != 8 or any(not host["layered_capable"] or not host.get("ipc_capable") for host in hosts):
            raise NetworkConfigurationError(
                "Independent RoCE rails with DeepEP require eight workers per node and "
                "full local GPU peer access and a DeepEP/RocSHMEM build supporting "
                "layered IPC and same-rail RDMA.")
    plans = {}
    for host in hosts:
        mapping = assignments[host["id"]]
        selected_names = {p["nic"] for p in mapping.values()}
        names = [name for name in host["names"] if name in selected_names]
        selected_ports = [next(p for p in mapping.values() if p["nic"] == name) for name in names]
        # Complete the file for visible but unused GPUs as well: topology printers
        # and RocSHMEM NUMA queries may enumerate them during initialization.
        for slot, gpu in enumerate(host["gpus"]):
            if slot not in mapping:
                mapping[slot] = min(selected_ports, key=lambda p: _score(gpu, p, used[host["id"]]))
        topo = "".join(f'{gpu["bus"]} {mapping[i]["nic"]} {names.index(mapping[i]["nic"])}\n'
                       for i, gpu in enumerate(host["gpus"]))
        env = {"NCCL_IB_HCA": "=" + ",".join(f'{p["nic"]}:{p["port"]}' for p in selected_ports)}
        if restricted:
            env["NCCL_CROSS_NIC"] = "0"
            # Independent L3 rails must not be merged into one NCCL virtual NIC.
            env["NCCL_IB_MERGE_NICS"] = "0"
        if not ib:
            family = ipaddress.ip_network(common[0].removeprefix("roce:")).version
            env["NCCL_IB_ADDR_FAMILY"] = "AF_INET" if family == 4 else "AF_INET6"
        if host["deepep"]:
            env["ROCSHMEM_ALLOWED_IBV_DEVICES"] = ",".join(names)
            if restricted:
                env.update(ROCSHMEM_GDA_RAIL_ONLY="1", DEEPEP_ENABLE_LL_LAYERED_OPT="1")
        plans[host["id"]] = {"env": env, "topo": topo, "mapping": mapping,
                              "restricted": restricted, "common_fabrics": common,
                              "ordering_omitted": sorted(shared_fabrics - set(common)) if not ib else []}
    return plans


@functools.lru_cache(maxsize=1)
def layered_capable() -> bool:
    """Check installed capabilities without importing DeepEP or opening a GPU."""
    spec = importlib.util.find_spec("deep_ep")
    if spec is None or not spec.origin:
        return False
    root = Path(spec.origin).parent
    if "DEEPEP_ENABLE_LL_LAYERED_OPT" not in _read(root / "buffer.py"):
        return False
    for library in root.glob("*cpp*.so"):
        with library.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if data.find(b"ROCSHMEM_GDA_RAIL_ONLY") >= 0:
                return True
    return False


def _route_interface(master: str) -> str | None:
    import fcntl

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect((master, 9))  # UDP connect selects a route; sends no packet.
            source = route.getsockname()[0]
        for device in Path("/sys/class/net").iterdir():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    data = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", device.name.encode()[:15]))
                if socket.inet_ntoa(data[20:24]) == source:
                    return device.name
            except OSError:
                continue
    except OSError:
        pass
    return None


def apply_plan(plan: dict, local_rank: int, master: str, environ=None) -> dict[str, str]:
    environ = os.environ if environ is None else environ
    values = dict(plan["env"])
    if "ROCSHMEM_ALLOWED_IBV_DEVICES" in values:
        folder = tempfile.TemporaryDirectory(prefix="vllm-hcu-network-", dir="/dev/shm" if Path("/dev/shm").is_dir() else None)
        _TOPO_DIRS.append(folder)
        path = Path(folder.name) / "topo.config"
        path.write_text(plan["topo"])
        values["ROCSHMEM_TOPO_FILE_FORCE"] = str(path)
        values["ROCSHMEM_IB_GID_INDEX"] = str(plan["mapping"][local_rank]["gid"])
    interface = _route_interface(master)
    if interface:
        values.update(NCCL_SOCKET_IFNAME="=" + interface, GLOO_SOCKET_IFNAME=interface)
    for name, value in values.items():
        if name not in environ:
            _OWNED_ENV[name] = value
        environ.setdefault(name, value)
    return {name: environ[name] for name in values}


def prepare_network(config, world_size, rank, init_method, local_rank):
    """Use the same public rendezvous as vLLM, before its first NCCL init.

    Keep the returned generator/store alive until init_distributed_environment
    returns. TCP rendezvous uses a multi-tenant server, which the upstream
    default process group reuses on rank zero (including eager/split-group init).
    """
    import torch.distributed as dist

    parallel = config.parallel_config
    if parallel.nnodes <= 1 or getattr(parallel, "enable_elastic_ep", False):
        return None
    if parallel.distributed_executor_backend == "ray":
        logger.info("HCU automatic network setup is not applied to Ray's per-worker device masks")
        return None
    if parallel.distributed_executor_backend != "external_launcher":
        rank = parallel.data_parallel_rank * world_size + rank
        world_size = parallel.world_size_across_dp
        address = parallel.master_addr
        if ":" in address and not address.startswith("["):
            address = f"[{address}]"
        init_method = f"tcp://{address}:{parallel.master_port}"
    if not init_method.startswith(("tcp://", "env://")):
        raise NetworkConfigurationError("Automatic multi-node networking requires TCP/env rendezvous")
    timeout = timedelta(seconds=parallel.distributed_timeout_seconds or 300)
    rendezvous = dist.rendezvous(init_method, rank=rank, world_size=world_size, timeout=timeout)
    store, _, _ = next(rendezvous)
    prefix = dist.PrefixStore("vllm_hcu_auto_network_v1", store)
    for name, value in _OWNED_ENV.items():
        if os.environ.get(name) == value:
            os.environ.pop(name)
    _OWNED_ENV.clear()
    manual = [name for name in _MANUAL if name in os.environ]
    disabled = os.environ.get("VLLM_HCU_AUTO_NETWORK", "1").lower() in {"0", "false", "off"}
    deepep = bool(parallel.enable_expert_parallel and str(parallel.all2all_backend).startswith("deepep"))
    host = {"id": _read(Path("/proc/sys/kernel/random/boot_id"), socket.gethostname()),
            "rank": rank, "local_rank": local_rank, "manual": manual, "disabled": disabled,
            "deepep": deepep, "layered_capable": False}
    try:
        if not manual and not disabled:
            host.update(discover_host())
            host["layered_capable"] = layered_capable() if deepep else False
    except Exception as exc:
        host["error"] = f"{type(exc).__name__}: {exc}"
    prefix.set(str(rank), json.dumps(host))
    peers = [json.loads(prefix.get(str(i))) for i in range(world_size)]
    if any(p["manual"] or p["disabled"] for p in peers):
        if rank == 0:
            logger.info("HCU automatic network setup skipped: a worker has explicit transport settings or opted out")
        return rendezvous, store
    errors = [p["error"] for p in peers if "error" in p]
    if errors:
        raise NetworkConfigurationError("HCU network discovery failed: " + "; ".join(sorted(set(errors))))
    nodes = {}
    for peer in peers:
        node = nodes.setdefault(peer["id"], dict(peer, slots=[]))
        node["slots"].append(peer["local_rank"])
    for node in nodes.values():
        node["slots"] = sorted(node["slots"])
        if len(node["slots"]) != len(set(node["slots"])):
            raise NetworkConfigurationError("Workers on one host have duplicate local GPU ranks")
        if any(i < 0 or i >= len(node["gpus"]) for i in node["slots"]):
            raise NetworkConfigurationError("Local GPU rank is outside the visible HIP device list")
    plans = build_plans(list(nodes.values()))
    plan = plans[host["id"]]
    if rank == 0 and plan.get("ordering_omitted"):
        logger.warning("HCU network plan omits common rails to keep NCCL HCA ordering/address "
                       "family consistent: %s", plan["ordering_omitted"])
    values = apply_plan(plan, local_rank, parallel.master_addr)
    logger.info("HCU auto network rank=%s GPU=%s -> %s GID=%s; NCCL=%s; same-rail=%s",
                rank, nodes[host["id"]]["gpus"][local_rank]["bus"],
                plan["mapping"][local_rank]["nic"], plan["mapping"][local_rank]["gid"],
                values.get("NCCL_IB_HCA", "Socket"), plan["restricted"])
    return rendezvous, store
