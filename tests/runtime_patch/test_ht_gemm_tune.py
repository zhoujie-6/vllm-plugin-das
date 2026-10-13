# SPDX-License-Identifier: Apache-2.0
import torch
import pytest

from vllm_hcu.model_executor.layers.fused_moe import ht_gemm_tune as tune


@pytest.fixture
def setup(monkeypatch):
    tune._cache.clear()
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_DEEPEP_HT_GEMM_TUNE", "1")
    monkeypatch.setattr(tune, "_supported", lambda *args: True)
    monkeypatch.setattr(tune, "_sample", lambda ids: torch.where(ids >= 0)[0])
    calls = []
    a = (torch.arange(16).float().reshape(8, 2), torch.ones((8, 1)))
    b = (torch.ones((2, 2)), torch.ones((2, 1)))
    out = torch.empty((8, 2))
    ids = torch.tensor([0, 0, -1, -1, 1, 1, -1, -1], dtype=torch.int32)
    def gemm(a, b, output, indices, config=None):
        calls.append(config)
        output.copy_(a[0] @ b[0])
        return output
    def time(call):
        call()
        config = calls[-1]
        return 10.0 if config is None else (6.0 if config["BLOCK_M"] == 64 else 12.0)
    monkeypatch.setattr(tune, "_time", time)
    yield a, b, out, ids, gemm, calls
    tune._cache.clear()


def test_profile_selects_valid_faster_config_and_runtime_only_calls_once(setup):
    a, b, out, ids, fn, calls = setup
    with tune.ht_gemm_profile_scope():
        tune.run_ht_fp8_gemm(fn, a, b, out, ids)
    torch.testing.assert_close(out, a[0] @ b[0])
    assert calls[-1]["BLOCK_M"] == 64
    calls.clear()
    tune.run_ht_fp8_gemm(fn, a, b, out, ids)
    assert len(calls) == 1 and calls[0]["BLOCK_M"] == 64


@pytest.mark.parametrize("switch", ["child", "master", "unknown_shape", "unsupported"])
def test_runtime_fallback_never_times(setup, monkeypatch, switch):
    a, b, out, ids, fn, calls = setup
    if switch == "child": monkeypatch.setenv("VLLM_HCU_DEEPEP_HT_GEMM_TUNE", "0")
    if switch == "master": monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    if switch == "unsupported": monkeypatch.setattr(tune, "_supported", lambda *args: False)
    def forbidden(call):
        raise AssertionError("runtime must not calibrate")
    monkeypatch.setattr(tune, "_time", forbidden)
    tune.run_ht_fp8_gemm(fn, a, b, out, ids)
    assert calls == [None]


def test_faster_but_wrong_candidate_rejected(setup, monkeypatch):
    a, b, out, ids, fn, calls = setup
    def bad_fn(a, b, output, indices, config=None):
        fn(a, b, output, indices, config)
        if config is not None: output.add_(10)
        return output
    with tune.ht_gemm_profile_scope():
        tune.run_ht_fp8_gemm(bad_fn, a, b, out, ids)
    assert calls[-1] is None
    torch.testing.assert_close(out, a[0] @ b[0])


def test_timing_noise_retains_baseline(setup, monkeypatch):
    a, b, out, ids, fn, calls = setup
    def time(call):
        call()
        return 10.0 if calls[-1] is None else 9.8
    monkeypatch.setattr(tune, "_time", time)
    with tune.ht_gemm_profile_scope():
        tune.run_ht_fp8_gemm(fn, a, b, out, ids)
    assert calls[-1] is None


def test_scope_restored_after_failure(setup):
    assert not tune._profile.get()
    with pytest.raises(RuntimeError):
        with tune.ht_gemm_profile_scope():
            raise RuntimeError("profile failed")
    assert not tune._profile.get()


def test_cache_capacity_prevents_more_calibration(setup, monkeypatch):
    a, b, out, ids, fn, calls = setup
    monkeypatch.setattr(tune, "_MAX_KEYS", 0)
    with tune.ht_gemm_profile_scope():
        tune.run_ht_fp8_gemm(fn, a, b, out, ids)
    assert calls == [None]


def test_prefix_calibration_restores_full_output(setup):
    a, b, _, _, fn, calls = setup
    a = (torch.arange(16384).float().reshape(8192, 2), torch.ones((8192, 1)))
    out = torch.empty((8192, 2))
    ids = torch.zeros(8192, dtype=torch.int32)
    with tune.ht_gemm_profile_scope():
        tune.run_ht_fp8_gemm(fn, a, b, out, ids)
    assert len(tune._cache) == 2
    torch.testing.assert_close(out, a[0] @ b[0])


@pytest.mark.parametrize("fail", [False, True])
def test_model_profile_enters_and_restores_tuning_scope(fail):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).parents[2] / "vllm_hcu/v1/hcu_model_runner_v2.py"
    tree = ast.parse(path.read_text())
    method = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                  for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "profile_run")
    class Base:
        def profile_run(self):
            assert tune._profile.get()
            if fail: raise RuntimeError("profile failure")
            return 42
    node = ast.ClassDef(name="Runner", bases=[ast.Name(id="Base", ctx=ast.Load())],
                       keywords=[], body=[method], decorator_list=[])
    ns = {"Base": Base, "is_deepseek_v4": lambda config: True}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), ns)
    runner = ns["Runner"]()
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=8))
    if fail:
        with pytest.raises(RuntimeError): runner.profile_run()
    else:
        assert runner.profile_run() == 42
    assert not tune._profile.get()


def test_legacy_vendor_api_without_config_retains_original_call(setup):
    a, b, out, ids, fn, calls = setup
    def old_fn(a, b, output, indices):
        return fn(a, b, output, indices)
    with tune.ht_gemm_profile_scope():
        tune.run_ht_fp8_gemm(old_fn, a, b, out, ids)
    assert calls == [None]
    assert list(tune._cache.values()) == [None]
