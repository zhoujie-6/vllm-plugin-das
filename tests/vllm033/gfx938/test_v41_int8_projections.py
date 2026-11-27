"""Check per-group activation and per-output-channel scales independently."""
import pytest
import torch

from hcu_vllm_das.quantization.v41_int8_projections import grouped_projection


@pytest.mark.hcu
@pytest.mark.parametrize("rows", [0, 1, 33])
def test_grouped_wo_a_matches_integer_reference(rows):
    generator = torch.Generator(device="cuda").manual_seed(77)
    groups, rank, width = 8, 1024, 4096
    x = torch.randn(rows, groups, width, device="cuda", dtype=torch.bfloat16, generator=generator)
    # Exact power-of-two scales keep values at half-integer ties independent
    # of the GPU reciprocal approximation used by the activation quantizer.
    x[:, :, 0] = 127
    x *= (2.0 ** torch.arange(groups, device="cuda")).view(1, groups, 1)
    x[:, 0].zero_()
    weights = torch.randint(-127, 128, (groups * rank, width), device="cuda",
                            dtype=torch.int8, generator=generator)
    scales = torch.rand(groups * rank, 1, device="cuda", generator=generator) * 0.02
    actual = grouped_projection(x, weights, scales)
    assert actual.shape == (rows, groups, rank)
    assert actual.dtype == torch.bfloat16
    if rows:
        cols = [0, rank // 2, rank - 1]
        peak = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-10)
        codes = (x.float() * (127 / peak)).round().to(torch.int32)
        w = weights.view(groups, rank, width)[:, cols].int()
        accum = (codes.unsqueeze(2) * w).sum(-1, dtype=torch.int32)
        expected = accum.float() * (peak / 127) * scales.view(groups, rank)[:, cols]
        torch.testing.assert_close(actual[:, :, cols], expected.bfloat16(), rtol=0, atol=0)


@pytest.mark.hcu
def test_checkpoint_output_projections(tmp_path):
    """Read actual target and draft weights without allocating the full model."""
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace as NS

    from safetensors import safe_open
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment, destroy_model_parallel,
        init_distributed_environment, initialize_model_parallel,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention
    from vllm.models.deepseek_v41.amd.vl_model import _make_deepseek_v4_vl_weights_mapper
    from vllm.utils.torch_utils import set_default_torch_dtype
    from hcu_vllm_das.quantization.v41_int8_projections import configure_draft, configure_target

    model_dir = os.environ.get("DSV41_W4A8_CHECKPOINT")
    if not model_dir:
        pytest.skip("set DSV41_W4A8_CHECKPOINT to test an installed checkpoint")
    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = EngineArgs(model=str(model_dir), quantization="compressed-tensors",
                     enforce_eager=True, max_model_len=4096).create_engine_config()

    def read(key):
        with safe_open(str(model_dir / index[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def load(layer, prefix):
        w = read(prefix + ".weight")
        assert w.dtype == layer.weight.dtype
        scale = None
        if prefix + ".weight_scale" in index:
            scale = read(prefix + ".weight_scale")
            layer.weight_scale.weight_loader(layer.weight_scale, scale)
        layer.weight.weight_loader(layer.weight, w)
        layer.quant_method.process_weights_after_loading(layer)
        return w, scale

    def make_attention(prefix):
        wo_a = ColumnParallelLinear(4096, 8192, bias=False, quant_config=cfg.quant_config,
                                    prefix=prefix + ".wo_a")
        a = NS(wo_a=wo_a, n_local_heads=64, n_local_groups=8, o_lora_rank=1024,
               rope_head_dim=64, rotary_emb=None, _wo_b_after_wo_a=lambda z: z)
        a._o_proj = DeepseekV41ROCMAiterMLAAttention._o_proj.__get__(a)
        return a

    with set_current_vllm_config(cfg):
        init_distributed_environment(1, 0, f"file://{tmp_path}/dist", 0, backend="nccl")
        initialize_model_parallel()
        try:
            with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16), torch.inference_mode():
                a, d = (make_attention(p) for p in ("language_model.model.layers.0.attn",
                                                    "model.layers.40.attn"))
                lm = ParallelLMHead(129280, 5120, prefix="language_model.lm_head")
                mk = ParallelLMHead(129280, 256, prefix="model.markov_head.markov_w2", disable_tp=True)
                mk._retain_weight_for_gather = False
                mapper = _make_deepseek_v4_vl_weights_mapper("fp4", "weight_scale_inv")
                target = NS(language_model=NS(lm_head=lm, model=NS(layers=[NS(attn=a)])),
                            hf_to_vllm_mapper=mapper)
                draft = NS(model=NS(layers=[NS(attn=d)], markov_head=NS(markov_w2=mk)))
                configure_target(target, cfg)
                configure_draft(draft, cfg)

                if "head.weight_scale" in index:
                    assert mapper.map_name("head.weight_scale") == "language_model.lm_head.weight_scale"
                else:
                    assert target.language_model.lm_head is lm
                    assert draft.model.markov_head.markov_w2 is mk

                for head, key in ((target.language_model.lm_head, "head"),
                                  (draft.model.markov_head.markov_w2, "mtp.2.markov_head.head")):
                    w, scale = load(head, key)
                    x = torch.randint(-62, 63, (3, w.shape[1])).bfloat16()
                    x[:, 0] = 127  # Unit activation scale for the independent reference.
                    got = LogitsProcessor(129280)(head, x)[:, :72]
                    expected = x.float() @ w[:72].cuda().float().T
                    if scale is not None:
                        expected *= scale[:72].cuda().T
                    torch.testing.assert_close(got.float(), expected.bfloat16().float(), rtol=.009, atol=.03)

                for attn, key in ((a, "layers.0.attn.wo_a"), (d, "mtp.0.attn.wo_a")):
                    w, scale = load(attn.wo_a, key)
                    x = torch.randint(-62, 63, (2, 8, 4096)).bfloat16()
                    x[:, :, 0] = 127
                    got = attn._o_proj(x.view(2, 64, 512), None).view(2, 8, 1024)
                    expected = (x.float().unsqueeze(2) * w.view(8, 1024, 4096)[:, :5].cuda().float()).sum(-1)
                    if scale is not None:
                        expected *= scale.view(8, 1024)[:, :5].cuda()
                    torch.testing.assert_close(got[:, :, :5].float(), expected.bfloat16().float(),
                                               rtol=.009, atol=.03)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
