# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 Channel-FP8 schema routing without changing official config owners.

The native V4.1 configuration is subclassed, rather than reconstructing its
constructor parameter list, so native FP8 policy and future optional arguments
remain owned by vLLM. Only the reviewed channel schema changes linear/expert
routing. CT continues to own its quantization methods and kernel selection.
"""
from __future__ import annotations

from collections.abc import Mapping

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
from vllm.platforms import current_platform

from hcu_vllm_das.dsv41_activation import QUANTIZATION_METHOD, check_health


def _field(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _is_v41(hf_config) -> bool:
    for config in (hf_config, _field(hf_config, "text_config")):
        if config is None:
            continue
        if _field(config, "model_type") in ("deepseek_v41", "deepseek_v41_text"):
            return True
        if "DeepseekV41ForCausalLM" in (_field(config, "architectures") or ()):
            return True
    return False


def _is_channel_schema(config) -> bool:
    return (
        isinstance(config, dict)
        and config.get("quant_method") in ("fp8", "deepseek_v4_fp8")
        and config.get("is_per_channel") is True
        and config.get("weight_block_size") is None
        and config.get("activation_scheme", "dynamic") == "dynamic"
    )


def _is_hcu() -> bool:
    return current_platform.is_rocm() and current_platform.device_name == "hcu"


def _ct_description() -> dict:
    # Allocate each nested dictionary per configuration. The checkpoint and
    # another model's parsed scheme must not be mutated by CT or a mapper.
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8, "type": "float", "symmetric": True,
                    "dynamic": False, "strategy": "channel", "group_size": None,
                },
                "input_activations": {
                    "num_bits": 8, "type": "float", "symmetric": True,
                    "dynamic": True, "strategy": "token", "group_size": None,
                },
            },
        },
        "ignore": [r"re:.*attn\.wo_a.*"],
    }


class HcuDSV41ChannelFP8Config(DeepseekV4FP8Config):
    """A distinct public configuration; native FP8 and CT remain unmodified."""

    @classmethod
    def get_name(cls):
        return QUANTIZATION_METHOD

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if (
            user_quant not in (None, "fp8", "deepseek_v4_fp8", QUANTIZATION_METHOD)
            or not _is_hcu()
            or not _is_v41(hf_config)
            or not _is_channel_schema(hf_quant_cfg)
        ):
            return None
        check_health()
        return QUANTIZATION_METHOD

    @classmethod
    def from_config(cls, config):
        if not _is_channel_schema(config):
            raise ValueError(
                "DSV4.1 Channel-FP8 requires channel weights and dynamic activations"
            )
        if not _is_hcu():
            raise ValueError("DSV4.1 Channel-FP8 routing requires the HCU platform")
        check_health()
        values = dict(config)
        values.setdefault("activation_scheme", "dynamic")
        return super().from_config(values)

    def _channel_config(self):
        config = getattr(self, "_hcu_channel_config", None)
        if config is None:
            config = CompressedTensorsConfig.from_config(_ct_description())
            self._hcu_channel_config = config
        # vLLM updates this mapping as models are constructed. Keep that
        # native state current without translating FP8 exclusions into CT's
        # different ignore syntax.
        config.packed_modules_mapping = self.packed_modules_mapping
        return config

    def get_quant_method(self, layer, prefix):
        check_health()
        if isinstance(layer, RoutedExperts) and is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return super().get_quant_method(layer, prefix)
        if isinstance(layer, (LinearBase, RoutedExperts)) and is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
            match_mode=self.ignored_layers_match_mode,
        ):
            return super().get_quant_method(layer, prefix)
        if isinstance(layer, LinearBase):
            if prefix.endswith(".wo_a"):
                return UnquantizedLinearMethod()
            channel_config = self._channel_config()
            method = channel_config.get_quant_method(layer, prefix)
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
                CompressedTensorsLinearMethod,
            )
            from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
                CompressedTensorsW8A8Fp8,
            )

            if isinstance(method, CompressedTensorsLinearMethod) and isinstance(
                layer.scheme, CompressedTensorsW8A8Fp8
            ):
                from .dsv41_channel_fp8_linear import HcuDSV41ChannelFP8LinearMethod

                return HcuDSV41ChannelFP8LinearMethod(channel_config)
            return method
        if isinstance(layer, RoutedExperts):
            channel_config = self._channel_config()
            channel_config._add_fused_moe_to_target_scheme_map()
            names = [
                f"{prefix}.0.{projection}"
                for projection in (
                    layer.ckpt_gate_proj_name,
                    layer.ckpt_up_proj_name,
                    layer.ckpt_down_proj_name,
                )
            ]
            schemes = [channel_config.get_scheme_dict(layer, name) for name in names]
            if not all(scheme == schemes[0] for scheme in schemes):
                raise ValueError("All Channel-FP8 MoE projections require one quantization scheme")
            scheme = schemes[0]
            if scheme is not None:
                weight_quant = scheme.get("weights")
                input_quant = scheme.get("input_activations")
                if (
                    weight_quant is not None
                    and input_quant is not None
                    and weight_quant.num_bits == 8
                    and input_quant.num_bits == 8
                    and weight_quant.strategy == "channel"
                    and input_quant.strategy == "token"
                    and weight_quant.type == "float"
                    and input_quant.type == "float"
                    and not weight_quant.dynamic
                    and input_quant.dynamic
                    and weight_quant.block_structure is None
                ):
                    if layer.moe_config.moe_backend == "deep_gemm_mega_moe":
                        from ..moe.v41_megamoe import HcuV41MegaMoEMethod
                        return HcuV41MegaMoEMethod(
                            weight_quant, input_quant, layer.moe_config, prefix)
                    from .channel_fp8_moe import HcuDSV41ChannelFP8MoEMethod

                    return HcuDSV41ChannelFP8MoEMethod(
                        weight_quant,
                        input_quant,
                        layer.moe_config,
                        prefix,
                    )
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import CompressedTensorsMoEMethod
            return CompressedTensorsMoEMethod.get_moe_method(channel_config, layer, prefix)
        return super().get_quant_method(layer, prefix)
