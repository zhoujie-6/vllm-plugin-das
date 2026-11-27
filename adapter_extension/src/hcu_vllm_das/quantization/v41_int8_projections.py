# SPDX-License-Identifier: Apache-2.0
"""Channel-INT8 output projections for the optional V4.1 W4A8 export."""
import re
from types import MethodType

import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)

from .ct_w4a8 import HcuDSV41CTW4A8Config


@triton.jit
def _grouped_mm(X, W, XS, WS, Y, M, G: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    g = tl.program_id(2)
    k = tl.arange(0, BK)
    x = X + (m[:, None] * G + g) * K + k[None, :]
    w = W + (g * N + n[:, None]) * K + k[None, :]
    acc = tl.full((BM, BN), 0, tl.int32)
    for tile in range(tl.cdiv(K, BK)):
        valid = tile * BK + k < K
        a = tl.load(x, (m[:, None] < M) & valid[None, :], 0)
        b = tl.load(w, (n[:, None] < N) & valid[None, :], 0)
        acc += tl.dot(a, tl.trans(b))
        x += BK
        w += BK
    xs = tl.load(XS + m * G + g, m < M, 0)
    ws = tl.load(WS + g * N + n, n < N, 0)
    y = acc.to(tl.float32) * xs[:, None] * ws[None, :]
    tl.store(Y + (m[:, None] * G + g) * N + n[None, :], y,
             (m[:, None] < M) & (n[None, :] < N))


def grouped_projection(x, weight, scale):
    from lightop.quant import per_token_quant_int8

    rows, groups, width = x.shape
    rank = weight.shape[0] // groups
    result = x.new_empty((rows, groups, rank))
    if rows:
        codes, scales = per_token_quant_int8(x.reshape(rows * groups, width).contiguous())
        _grouped_mm[(triton.cdiv(rows, 32), triton.cdiv(rank, 64), groups)](
            codes, weight, scales, scale, result, rows, groups, width, rank, 32, 64, 64)
    return result


class _GroupedInt8Method(CompressedTensorsLinearMethod):
    def process_weights_after_loading(self, layer):
        # Grouped GEMM reads the original [group * rank, width] checkpoint order.
        if (layer.weight.dtype != torch.int8 or not layer.weight.is_contiguous()
                or layer.weight_scale.shape != (layer.weight.shape[0], 1)
                or not layer.weight_scale.is_contiguous()):
            raise ValueError("Invalid channel-INT8 WO-A weights or scales")


def _int8_o_proj(self, attn_out, positions):
    # The AMD attention backend has already applied inverse RoPE to these rows.
    rows = attn_out.shape[0]
    x = attn_out[:, :self.n_local_heads].reshape(rows, self.n_local_groups, self.wo_a.weight.shape[1])
    z = grouped_projection(x, self.wo_a.weight, self.wo_a.weight_scale)
    return self._wo_b_after_wo_a(z.flatten(1))


def _install_grouped(layers, quant):
    for layer in layers:
        attn = getattr(layer, "attn", None)
        wo_a = getattr(attn, "wo_a", None)
        if not isinstance(getattr(wo_a, "scheme", None), CompressedTensorsW8A8Int8):
            continue
        if wo_a.scheme.strategy != "channel" or wo_a.scheme.is_static_input_scheme or not wo_a.scheme.input_symmetric:
            raise ValueError("V4.1 INT8 WO-A requires channel weights and dynamic symmetric token activations")
        wo_a.quant_method = _GroupedInt8Method(quant)
        attn._o_proj = MethodType(_int8_o_proj, attn)


def _quantized_head(head, quant):
    if not isinstance(head, ParallelLMHead) or not quant.get_scheme_dict(head, head.prefix):
        return head
    replacement = ParallelLMHead(head.num_embeddings, head.embedding_dim, bias=head.has_bias,
                                 params_dtype=head.params_dtype, org_num_embeddings=head.org_vocab_size,
                                 padding_size=head.padding_size, quant_config=quant, prefix=head.prefix,
                                 disable_tp=head.disable_tp)
    if not isinstance(getattr(replacement, "scheme", None), CompressedTensorsW8A8Int8):
        raise ValueError(f"Expected a CT W8A8 INT8 head at {head.prefix}")
    return replacement


def configure_target(model, cfg):
    quant = getattr(cfg, "quant_config", None)
    if not isinstance(quant, HcuDSV41CTW4A8Config):
        return
    llm = model.language_model
    if isinstance(getattr(llm, "lm_head", None), ParallelLMHead):
        head = _quantized_head(llm.lm_head, quant)
        if head is not llm.lm_head:
            llm.lm_head = head
            mapper = model.hf_to_vllm_mapper
            mapper.orig_to_new_regex = {**mapper.orig_to_new_regex,
                re.compile(r"^head\.weight_scale$"): "language_model.lm_head.weight_scale"}
    _install_grouped(llm.model.layers, quant)


def configure_draft(model, cfg):
    quant = getattr(cfg, "quant_config", None)
    if not isinstance(quant, HcuDSV41CTW4A8Config):
        return
    markov = model.model.markov_head
    head = _quantized_head(markov.markov_w2, quant)
    if head is not markov.markov_w2:
        head._retain_weight_for_gather = markov.markov_w2._retain_weight_for_gather
        head.is_w4a16_nvfp4 = False
        markov.markov_w2 = head
    # The speculator shares the target's complete LM head after draft loading.
    _install_grouped(model.model.layers, quant)
