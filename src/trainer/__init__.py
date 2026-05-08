from __future__ import annotations

from typing import Any

from .base import TrainerBackend


def create_trainer(backend: str = "lightning", **kwargs) -> TrainerBackend:
    """Factory function to create a trainer backend.

    Args:
        backend: Either ``"lightning"`` (PyTorch Lightning Trainer) or
            ``"fabric"`` (Lightning Fabric manual loop).
        **kwargs: Forwarded to the backend constructor.

    Returns:
        A :class:`TrainerBackend` instance.
    """
    if backend == "lightning":
        from .lightning import LightningTrainerBackend

        return LightningTrainerBackend(**kwargs)
    elif backend == "fabric":
        from .fabric import FabricTrainerBackend

        return FabricTrainerBackend(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")


def normalize_state_dict(ckpt: Any) -> dict:
    """Extract a flat state_dict from various checkpoint formats.

    Handles:
        - ``{"state_dict": {...}}``         Lightning checkpoint
        - ``{"model_state_dict": {...}}``  custom save (this repo)
        - ``{"model": {...}}``              wrapped
        - bare ``state_dict`` dict

    Strips a leading ``"model."`` prefix from every key (Lightning saves
    keys as ``model.<original>`` because of the :class:`_LitWrapper`).
    """
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt

    if any(isinstance(k, str) and k.startswith("model.") for k in sd):
        sd = {
            (k[len("model.") :] if k.startswith("model.") else k): v
            for k, v in sd.items()
        }
    return sd


__all__ = ["TrainerBackend", "create_trainer", "normalize_state_dict"]
