# DeepSeek V4 online EPLB with deepep_auto

Online EPLB is enabled for DeepSeek V4 FP8 experts with the explicit
`deep_gemm` MoE backend. Other model/backend combinations retain the guard.
Both synchronous and asynchronous upstream EPLB modes are accepted.

```bash
--all2all-backend deepep_auto --moe-backend deep_gemm --enable-eplb \
--eplb-config '{"window_size":1000,"step_interval":1000,"num_redundant_experts":64,"use_async":true}'
```

Remove `expert_map_path`, `expert_map_record_path`, and `disable_rearrange`
for online operation. No offline artifact is required. Redundant expert slots
consume memory; ensure physical expert count is divisible by EP size.

The existing RoutedExperts weight registration includes HT weights, shared
channel scales, and independent LL packed weights when present. All participate
in the same EPLB transfer transaction. Shared layouts register one weight copy.
The existing async transfer buffers remain alive until completion, before
routing-map publication. Do not replace this registration with HT-only tensors.

Configuration, dual-layout transfer, buffer lifetime, and runtime regression
tests cover this path; a full multi-rank model run is still required to validate
a deployment and its workload accuracy.
