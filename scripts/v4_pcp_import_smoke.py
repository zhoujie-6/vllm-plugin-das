# SPDX-License-Identifier: Apache-2.0
"""Load the actual v0.25.1 V4 model through HCU callbacks; no weights required."""
from pathlib import Path
import os
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ['VLLM_PLUGINS'] = '__disabled__'
    import vllm_hcu
    vllm_hcu._prepare_general_plugin()
    from vllm.models.deepseek_v4.amd import model
    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_pcp_model
    assert patch_deepseek_v4_pcp_model.apply_to_module(model) is False
    from vllm_hcu.model_executor.layers import deepseek_v4_pcp  # noqa: F401
    from vllm_hcu.v1.attention import deepseek_v4_pcp as backend  # noqa: F401
    print(f'V4 PCP callbacks and operators loaded from {vllm_hcu.__file__}')


if __name__ == '__main__':
    main()
