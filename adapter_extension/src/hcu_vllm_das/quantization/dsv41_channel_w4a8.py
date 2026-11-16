# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the native CT W4A8 registration."""
from .ct_w4a8 import HcuDSV41CTW4A8Config as HcuDSV41ChannelW4A8Config
from .dsv41_channel_fp8 import _field


def _channel_w4a8(scheme):
    weight = scheme.get("weights")
    act = scheme.get("input_activations")
    return (
        scheme.get("format") == "pack-quantized"
        and _field(weight, "type") == "int" and _field(weight, "num_bits") == 4
        and _field(weight, "symmetric") is True and not _field(weight, "dynamic")
        and _field(weight, "strategy") == "channel"
        and _field(weight, "group_size") in (None, -1)
        and not _field(weight, "actorder")
        and _field(act, "type") == "int" and _field(act, "num_bits") == 8
        and _field(act, "strategy") == "token" and _field(act, "dynamic") is True
        and _field(act, "symmetric") is True and not scheme.get("output_activations")
    )

