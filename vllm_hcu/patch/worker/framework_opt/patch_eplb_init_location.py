# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Apply ``--init-expert-location`` right after the main model joins EPLB.

``EplbState.add_model`` builds the trivial placement that matches how expert
weights were loaded.  For the target (main) model, and only when the HCU
sidecar carries ``init_expert_location``, this adapter immediately rearranges
the experts to the static placement.  It runs inside ``Worker.load_model``,
whose ``set_current_vllm_config`` context provides the sidecar, and before the
async EPLB worker is started.  Draft models are left untouched.
"""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.distributed.eplb.eplb_state"
PATCH_ID = "worker.framework_opt.eplb.hcu_init_expert_location"
TARGETS = (f"{TARGET_MODULE}.EplbState.add_model",)

_MARKER = "_vllm_hcu_eplb_init_location_applied"
_WRAPPER = "_vllm_hcu_eplb_init_location_wrapper"


def _requested_location(model_config: object) -> str | None:
    from vllm.config import get_current_vllm_config

    from vllm_hcu.patch.config import get_hcu_config

    vllm_config = get_current_vllm_config()
    location = get_hcu_config(vllm_config).init_expert_location
    if location is None:
        return None
    main_config = getattr(vllm_config, "model_config", None)
    if main_config is None or (
        model_config is not main_config
        and model_config.compute_hash() != main_config.compute_hash()
    ):
        from vllm.logger import init_logger

        init_logger("vllm.hcu.eplb").info(
            "init_expert_location: skipping non-target EPLB model %s",
            getattr(model_config, "model", model_config),
        )
        return None
    return location


def apply_to_module(module: ModuleType) -> bool:
    eplb_state = load_exact_module(TARGET_MODULE, module)
    state_cls = require_class(eplb_state, "EplbState", f"{TARGET_MODULE}.EplbState")
    wrapped = ((state_cls, "add_model", TARGETS[0], _WRAPPER),)
    if already_applied(eplb_state, _MARKER, wrapped):
        return False

    original_add_model = require_callable(state_cls, "add_model", TARGETS[0])
    require_exact_signature(
        original_add_model,
        TARGETS[0],
        positional=("self", "model", "model_config"),
    )
    require_callable(
        eplb_state, "_commit_eplb_maps", f"{TARGET_MODULE}._commit_eplb_maps"
    )

    @functools.wraps(original_add_model)
    def hcu_add_model(self, model, model_config):
        result = original_add_model(self, model, model_config)
        location = _requested_location(model_config)
        if location is not None:
            from vllm_hcu.distributed.eplb_init_location import (
                apply_initial_expert_location,
            )

            # Same key as EplbState.add_model.
            model_state = self.model_states[model_config.compute_hash()]
            apply_initial_expert_location(self, model_state, location)
        return result

    setattr(hcu_add_model, _WRAPPER, True)
    setattr(state_cls, "_vllm_hcu_original_add_model", original_add_model)
    setattr(state_cls, "add_model", hcu_add_model)
    setattr(eplb_state, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
