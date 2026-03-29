# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id


def get_stage_type(stage_cfg: Any) -> str:
    """Best-effort stage type resolver across dict/omegaconf/object configs."""
    if isinstance(stage_cfg, dict):
        return stage_cfg.get("stage_type", "llm")
    if hasattr(stage_cfg, "get"):
        try:
            return stage_cfg.get("stage_type", "llm")
        except Exception:
            pass
    return getattr(stage_cfg, "stage_type", "llm")


def parse_lora_request(lora_body: Any) -> tuple[LoRARequest | None, float | None]:
    """Parse a request-level LoRA object into a LoRARequest and optional scale.

    Raises:
        ValueError: If the object shape is invalid or required fields are missing.
    """
    if lora_body is None:
        return None, None

    if not isinstance(lora_body, dict):
        raise ValueError("Invalid lora field: expected an object.")

    lora_name = lora_body.get("name") or lora_body.get("lora_name") or lora_body.get("adapter")
    lora_path = (
        lora_body.get("local_path")
        or lora_body.get("path")
        or lora_body.get("lora_path")
        or lora_body.get("lora_local_path")
    )
    lora_scale = lora_body.get("scale")
    if lora_scale is None:
        lora_scale = lora_body.get("lora_scale")
    lora_int_id = lora_body.get("int_id")
    if lora_int_id is None:
        lora_int_id = lora_body.get("lora_int_id")
    if lora_int_id is None and lora_path:
        lora_int_id = stable_lora_int_id(str(lora_path))

    if not lora_name or not lora_path:
        raise ValueError("Invalid lora object: both name and path are required.")

    scale = float(lora_scale) if lora_scale is not None else None
    return LoRARequest(str(lora_name), int(lora_int_id), str(lora_path)), scale


def parse_lora_requests(
    lora_body: dict[str, Any] | list[dict[str, Any]] | None,
) -> tuple[list[LoRARequest], list[float]]:
    """Parse one or more LoRA objects into parallel lists of requests and scales.

    Handles three shapes:
      - ``None``  -> empty lists
      - ``dict``  -> single adapter, wrapped in a one-element list
      - ``list``  -> multiple adapters

    Returns:
        ``(lora_requests, lora_scales)`` with matching lengths.
    """
    if lora_body is None:
        return [], []

    items: list[dict[str, Any]]
    if isinstance(lora_body, dict):
        items = [lora_body]
    elif isinstance(lora_body, list):
        items = lora_body
    else:
        raise ValueError("Invalid lora field: expected a dict, list of dicts, or null.")

    requests: list[LoRARequest] = []
    scales: list[float] = []
    for item in items:
        req, scale = parse_lora_request(item)
        if req is None:
            continue
        requests.append(req)
        scales.append(scale if scale is not None else 1.0)

    return requests, scales
