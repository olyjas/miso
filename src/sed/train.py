from __future__ import annotations

"""SED training entry-point.

Usage::

    python -m src.sed.train --config configs/sed/ast_finetune.yaml [--data_dir ...]
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.soundscape_dataset import SoundscapeDataset
from src.metrics.sed import ClassificationMetrics
from src.sed.loss import get_loss_function
from src.sed.model import ASTModel
from src.trainer import create_trainer
from src.tse.net import _import_attr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dataset(config: dict, split: str) -> SoundscapeDataset:
    """Instantiate a :class:`SoundscapeDataset` in SED mode."""
    data_cfg = config["data"]
    return SoundscapeDataset(
        fg_dir=data_cfg["fg_dir"],
        noise_dir=data_cfg["noise_dir"],
        hrtf_list=data_cfg["hrtf_list"],
        split=split,
        sr=data_cfg.get("sr", 16000),
        duration=data_cfg.get("duration", 5),
        bg_dir=data_cfg.get("bg_dir"),
        num_fg_range=tuple(data_cfg.get("num_fg_range", [1, 5])),
        num_bg_range=tuple(data_cfg.get("num_bg_range", [1, 3])),
        num_noise_range=tuple(data_cfg.get("num_noise_range", [1, 1])),
        snr_range_fg=tuple(data_cfg.get("snr_range_fg", [5, 15])),
        snr_range_bg=tuple(data_cfg.get("snr_range_bg", [0, 10])),
        num_total_labels=config["model"].get("num_classes", 20),
        samples_per_epoch=data_cfg.get("samples_per_epoch", 20000),
        hrtf_type=data_cfg.get("hrtf_type", "CIPIC"),
        task="sed",
    )


def _collate_fn(batch):
    """Custom collate: stack mixture waveforms and label vectors."""
    inputs_list, targets_list = zip(*batch)

    mixtures = torch.stack([inp["mixture"] for inp in inputs_list])
    labels = torch.stack([inp["labels"] for inp in inputs_list])

    return {
        "mixture": mixtures,
        "labels": labels,
    }, targets_list


def _build_optimizer(model: ASTModel, config: dict) -> torch.optim.Optimizer:
    """Create optimizer with layer-wise learning rates."""
    train_cfg = config["training"]

    encoder_lr = train_cfg.get("encoder_lr", 1e-5)
    head_lr = train_cfg.get("head_lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.01)

    # Separate parameters into encoder and classifier groups
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    param_groups = []
    if encoder_params:
        param_groups.append(
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay}
        )
    if head_params:
        param_groups.append(
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
        )

    return torch.optim.AdamW(param_groups)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning-rate scheduler from config via dotted-path."""
    train_cfg = config["training"]
    return _import_attr(train_cfg["scheduler_name"])(
        optimizer,
        **train_cfg.get("scheduler_params", {}),
    )


def _build_loss_fn(config: dict, dataset, device):
    """Trainer-facing adapter: (outputs, targets, inputs) -> scalar loss.

    SED ground truth is the multi-hot ``inputs["labels"]`` (NOT
    ``targets["target"]`` which is the per-source waveform produced by
    SoundscapeDataset for TSE-compatible output).
    """
    base = get_loss_function(config, dataset=dataset, device=device)

    def adapter(outputs, targets, inputs):
        logits = outputs["output"]      # (B, num_classes)
        labels = inputs["labels"]       # (B, num_classes)  multi-hot float
        return base(logits, labels)

    return adapter


def _build_metrics_fn(config: dict):
    """Trainer-facing adapter: (outputs, targets, inputs) -> dict of scalars."""
    threshold = config.get("evaluation", {}).get("threshold", 0.5)

    def adapter(outputs, targets, inputs):
        # Use softmax/sigmoid scores when available, otherwise raw logits.
        scores = outputs.get("scores")
        if scores is None:
            scores = torch.sigmoid(outputs["output"])
        labels = inputs["labels"]
        cm = ClassificationMetrics()
        preds_np = scores.detach().cpu().numpy()
        tgts_np = labels.detach().cpu().numpy()
        return cm.compute_all(preds_np, tgts_np, threshold=threshold)

    return adapter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SED model (AST fine-tuning)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Override data root")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override data dir if provided
    if args.data_dir:
        config["data"]["fg_dir"] = str(Path(args.data_dir) / "scaper_fmt")
        config["data"]["noise_dir"] = str(
            Path(args.data_dir) / "noise_scaper_fmt"
        )

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Datasets
    logger.info("Building datasets...")
    train_ds = _build_dataset(config, "train")
    val_ds = _build_dataset(config, "val")

    data_cfg = config["data"]
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 8),
        collate_fn=_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 8),
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    # Model
    model_cfg = config["model"]
    model = ASTModel(
        model_name=model_cfg.get("name", "MIT/ast-finetuned-audioset-10-10-0.4593"),
        num_classes=model_cfg.get("num_classes", 20),
        freeze_encoder=model_cfg.get("freeze_encoder", True),
        sample_rate=model_cfg.get("sample_rate", 16000),
    )
    logger.info(
        "Model: %d trainable / %d total parameters",
        model.get_trainable_parameters(),
        model.get_total_parameters(),
    )

    # Optimizer & scheduler
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    # Loss / metrics adapters (3-arg trainer interface)
    loss_fn = _build_loss_fn(config, dataset=train_ds, device=device)
    metrics_fn = _build_metrics_fn(config)

    # Trainer
    backend = config["training"].get("backend", "lightning")
    trainer = create_trainer(backend)

    # Train
    logger.info("Starting training...")
    result = trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        metrics_fn=metrics_fn,
    )

    logger.info("Training complete. Result: %s", result)


if __name__ == "__main__":
    main()
