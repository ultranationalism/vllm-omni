# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DIFFUSION_CPU_OFFLOAD env var (PR #2276).

Both config construction paths must honor the env var, and an explicit
`enable_cpu_offload` kwarg must win over it.
"""

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine


def _from_kwargs(**kw) -> bool:
    return OmniDiffusionConfig.from_kwargs(model="x", **kw).enable_cpu_offload


def _stage_cfg(**kw) -> bool:
    stages = AsyncOmniEngine._create_default_diffusion_stage_cfg({"model": "x", **kw})
    return stages[0]["engine_args"]["enable_cpu_offload"]


@pytest.mark.parametrize("build", [_from_kwargs, _stage_cfg], ids=["from_kwargs", "stage_cfg"])
class TestDiffusionCpuOffloadEnv:
    def test_env_enables_offload(self, monkeypatch, build):
        monkeypatch.setenv("DIFFUSION_CPU_OFFLOAD", "1")
        assert build() is True

    def test_explicit_kwarg_overrides_env(self, monkeypatch, build):
        monkeypatch.setenv("DIFFUSION_CPU_OFFLOAD", "1")
        assert build(enable_cpu_offload=False) is False
