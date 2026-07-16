from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

FILTER_WEIGHTS_BASENAME = "filter.pt"
FILTER_SPEC_BASENAME = "filter_spec.json"
LEGACY_WEIGHTS_GLOB = "model_weights_filter*.pth"


def _adapt_state_dict_for_predictor(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize legacy key layouts to StatePredictor.load_state_dict format."""
    if any(k.startswith("predictor.") for k in state_dict):
        return {k.removeprefix("predictor."): v for k, v in state_dict.items()}
    return state_dict


def model_config_to_filter_spec(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "filter_state_dim": int(model_config["state_dim"]),
        "filter_action_dim": int(model_config["action_dim"]),
        "filter_output_dim": int(model_config["output_dim"]),
        "filter_num_hidden": int(model_config["num_hidden"]),
        "filter_hidden_exp": int(model_config["hidden_exp"]),
        "filter_dropout": float(model_config["dropout"]),
    }


def filter_spec_to_model_config(filter_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_dim": int(filter_spec["filter_state_dim"]),
        "action_dim": int(filter_spec["filter_action_dim"]),
        "output_dim": int(filter_spec["filter_output_dim"]),
        "num_hidden": int(filter_spec["filter_num_hidden"]),
        "hidden_exp": int(filter_spec["filter_hidden_exp"]),
        "dropout": float(filter_spec["filter_dropout"]),
    }


def resolve_training_output_dir(output_path: str | Path | None) -> Path:
    if output_path is None:
        return Path(__file__).resolve().parent / "models"
    path = Path(output_path)
    if path.suffix in (".pt", ".pth"):
        return path.parent
    return path


def find_filter_weights_path(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Filter checkpoint directory not found: {checkpoint_dir}"
        )

    canonical = checkpoint_dir / FILTER_WEIGHTS_BASENAME
    if canonical.is_file():
        return canonical

    matches = sorted(
        checkpoint_dir.glob(LEGACY_WEIGHTS_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No filter weights found in {checkpoint_dir}. "
        f"Expected {FILTER_WEIGHTS_BASENAME} or {LEGACY_WEIGHTS_GLOB}."
    )


def read_filter_spec(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    spec_path = checkpoint_dir / FILTER_SPEC_BASENAME
    if spec_path.is_file():
        with open(spec_path, "r", encoding="utf-8") as fin:
            return json.load(fin)

    weights_path = find_filter_weights_path(checkpoint_dir)
    checkpoint = torch.load(weights_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Legacy filter checkpoint at {weights_path} is not a dict and "
            f"{FILTER_SPEC_BASENAME} is missing in {checkpoint_dir}."
        )
    model_config = checkpoint.get("model_config")
    if model_config is None:
        raise ValueError(
            f"Filter checkpoint at {weights_path} has no model_config and "
            f"{FILTER_SPEC_BASENAME} is missing in {checkpoint_dir}."
        )
    return model_config_to_filter_spec(model_config)


def validate_filter_spec_adapter_dims(
    filter_spec: dict[str, Any],
    *,
    state_dim: int,
    action_dim: int,
    output_dim: int,
    checkpoint_dir: str | Path,
) -> None:
    expected = {
        "filter_state_dim": int(state_dim),
        "filter_action_dim": int(action_dim),
        "filter_output_dim": int(output_dim),
    }
    mismatches = {
        key: (int(filter_spec[key]), expected[key])
        for key in expected
        if int(filter_spec[key]) != expected[key]
    }
    if mismatches:
        action_mismatch = mismatches.get("filter_action_dim")
        migration_hint = ""
        if action_mismatch is not None and action_mismatch[1] == 2:
            migration_hint = (
                " Runtime expects a 2D safety-filter action schema (SOI2, ID2). "
                "Legacy 3D checkpoints that include ID1 are incompatible; retrain "
                "and export a new filter checkpoint/ORT model with action_dim=2."
            )
        raise ValueError(
            "Filter checkpoint adapter-dimension mismatch at "
            f"{checkpoint_dir}: {mismatches}. "
            "State/action/output dims must match the runtime adapter."
            f"{migration_hint}"
        )


def save_filter_checkpoint(
    output_dir: str | Path,
    *,
    state_dict: dict[str, torch.Tensor],
    filter_spec: dict[str, Any],
    random_seed: int | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / FILTER_WEIGHTS_BASENAME
    spec_path = output_dir / FILTER_SPEC_BASENAME
    payload: dict[str, Any] = {
        "model_state_dict": state_dict,
        "model_config": filter_spec_to_model_config(filter_spec),
    }
    if random_seed is not None:
        payload["random_seed"] = int(random_seed)

    torch.save(payload, weights_path)
    with open(spec_path, "w", encoding="utf-8") as fout:
        json.dump(filter_spec, fout, indent=2)

    return weights_path, spec_path


def load_filter_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_adapter_dims: dict[str, int] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], Path]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Filter checkpoint directory not found: {checkpoint_dir}"
        )

    filter_spec = read_filter_spec(checkpoint_dir)
    if expected_adapter_dims is not None:
        validate_filter_spec_adapter_dims(
            filter_spec,
            state_dim=int(expected_adapter_dims["state"]),
            action_dim=int(expected_adapter_dims["action"]),
            output_dim=int(expected_adapter_dims["output"]),
            checkpoint_dir=checkpoint_dir,
        )

    weights_path = find_filter_weights_path(checkpoint_dir)
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported filter checkpoint format at {weights_path}")
    state_dict = _adapt_state_dict_for_predictor(state_dict)

    return state_dict, filter_spec, weights_path
