#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Full-graph (ACL Graph) e2e for the custom FIA (BlasST) dispatch.

UT cannot cover graph capture/update/replay semantics (task-update windows,
captured weak-ref addresses, per-step metadata refresh), so this test runs a
real engine: Qwen3-0.6B (head_dim 128, dense GQA, causal — eligible for the
custom path) with ``cudagraph_mode=FULL_DECODE_ONLY`` and
``custom_fia_config={enabled, full_graph}``, then compares greedy outputs
against the same engine with the custom path disabled. It also asserts the
custom gate actually admitted batches (i.e. the graph path was exercised),
so the comparison cannot pass vacuously if the gate silently falls back
everywhere.
"""

import pytest
from vllm import LLM, SamplingParams

import vllm_ascend.attention.attention_v1 as attn_module
from tests.e2e.conftest import cleanup_dist_env_and_memory

QWEN3 = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 32

PROMPTS = [
    "The capital of France is",
    "Explain the theory of relativity in simple terms:",
]


def _make_llm(custom_fia: bool) -> LLM:
    additional_config = None
    if custom_fia:
        additional_config = {
            "custom_fia_config": {
                "enabled": True,
                "full_graph": True,
            },
        }
    return LLM(
        model=QWEN3,
        max_model_len=1024,
        enforce_eager=False,
        gpu_memory_utilization=0.85,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
        additional_config=additional_config,
    )


def _greedy_tokens(llm: LLM):
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    return [[out.token_ids for out in req.outputs] for req in llm.generate(PROMPTS, params)]


def test_custom_fia_full_graph_matches_baseline():
    try:
        baseline = _greedy_tokens(_make_llm(custom_fia=False))
    finally:
        cleanup_dist_env_and_memory()

    gate_calls = {"admitted": 0, "checked": 0}
    orig_gate = attn_module.AscendAttentionBackendImpl._can_use_custom_fia

    def counting_gate(self, attn_metadata):
        admitted = orig_gate(self, attn_metadata)
        gate_calls["checked"] += 1
        gate_calls["admitted"] += int(admitted)
        return admitted

    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(attn_module.AscendAttentionBackendImpl,
                       "_can_use_custom_fia", counting_gate)
            custom = _greedy_tokens(_make_llm(custom_fia=True))
    finally:
        cleanup_dist_env_and_memory()

    assert gate_calls["admitted"] > 0, (
        "custom FIA gate never admitted a batch under full-graph decode — "
        "the output comparison below would pass vacuously")
    assert custom == baseline, (
        "custom FIA full-graph decode diverges from baseline: "
        f"baseline={baseline} custom={custom}")
