from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from .config import apply_overrides, load_config, save_config
from .data import build_dataset
from .flow import ResFlowProcess
from .metrics import mae, psnr, ssim
from .model import build_model
from .utils import (
    atomic_link_or_copy,
    atomic_torch_save,
    checkpoint_state,
    distributed_setup,
    seed_everything,
    unwrap_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ResFlow reproduction")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def cosine_lambda(step: int, total: int, minimum_ratio: float) -> float:
    progress = min(max(step / total, 0.0), 1.0)
    return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    process: ResFlowProcess,
    loader: DataLoader,
    device: torch.device,
    *,
    steps: int,
    seed: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {"PSNR": 0.0, "SSIM": 0.0, "MAE": 0.0}
    count = 0
    for batch in loader:
        lq = batch["lq"].to(device, non_blocking=True)
        hq = batch["hq"].to(device, non_blocking=True)
        y1 = torch.randn(lq.shape, device=device, dtype=lq.dtype, generator=generator)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with amp_context:
            restored = process.restore(model, lq, steps=steps, y1=y1)
        batch_size = lq.shape[0]
        totals["PSNR"] += psnr(restored, hq).sum().item()
        totals["SSIM"] += ssim(restored, hq).sum().item()
        totals["MAE"] += mae(restored, hq).sum().item()
        count += batch_size
    if was_training:
        model.train()
    return {name: value / count for name, value in totals.items()}


def load_best_records(output: Path) -> list[dict]:
    path = output / "best_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def update_best(
    output: Path,
    last_path: Path,
    records: list[dict],
    *,
    metrics: dict[str, float],
    step: int,
    metric: str,
    top_k: int,
) -> list[dict]:
    candidate = {
        "step": step,
        "metric": metric,
        "score": metrics[metric],
        "metrics": metrics,
        "file": f"best_step_{step:07d}_{metric.lower()}_{metrics[metric]:.4f}.pt",
    }
    ranked = sorted(records + [candidate], key=lambda item: item["score"], reverse=True)
    kept = ranked[:top_k]
    if candidate in kept:
        atomic_link_or_copy(last_path, output / candidate["file"])
    kept_files = {item["file"] for item in kept}
    for record in records:
        path = output / record["file"]
        if record["file"] not in kept_files and path.exists():
            path.unlink()
    metadata = output / "best_metrics.json"
    temporary = metadata.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata)
    return kept


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    rank, world_size, local_rank = distributed_setup()
    train_cfg = config["train"]
    seed_everything(int(train_cfg.get("seed", 0)), rank)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        save_config(config, output / "resolved_config.yaml")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", local_rank)

    global_batch = int(train_cfg["global_batch_size"])
    if global_batch % world_size:
        raise ValueError(f"global batch {global_batch} is not divisible by world size {world_size}")
    batch_size = global_batch // world_size
    dataset = build_dataset(config["data"]["train"], training=True)
    sampler = (
        DistributedSampler(dataset, world_size, rank, shuffle=True, seed=int(train_cfg.get("seed", 0)))
        if world_size > 1
        else RandomSampler(dataset)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(train_cfg.get("workers", 8)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=int(train_cfg.get("workers", 8)) > 0,
    )

    validation_cfg = config.get("validation", {})
    validation_loader = None
    if rank == 0 and validation_cfg.get("enabled", False):
        validation_dataset = build_dataset(config["data"]["test"], training=False)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(validation_cfg.get("batch_size", 16)),
            shuffle=False,
            num_workers=int(validation_cfg.get("workers", 4)),
            pin_memory=device.type == "cuda",
            persistent_workers=int(validation_cfg.get("workers", 4)) > 0,
        )

    model = build_model(config["model"]).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    process = ResFlowProcess(**config["flow"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg["betas"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    iterations = int(train_cfg["iterations"])
    minimum_ratio = float(train_cfg["minimum_learning_rate"]) / float(train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_lambda(step, iterations, minimum_ratio)
    )
    precision = train_cfg.get("precision", "fp32")
    use_amp = device.type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and precision == "fp16")
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])

    save_every = int(train_cfg.get("save_every", 10_000))
    log_every = int(train_cfg.get("log_every", 100))
    best_records = load_best_records(output) if rank == 0 else []
    epoch = 0
    data_iterator = iter(loader)
    model.train()
    last_time = time.time()
    for step in range(start_step, iterations):
        try:
            batch = next(data_iterator)
        except StopIteration:
            epoch += 1
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            data_iterator = iter(loader)
            batch = next(data_iterator)
        hq = batch["hq"].to(device, non_blocking=True)
        lq = batch["lq"].to(device, non_blocking=True)
        flow_batch = process.training_batch(hq, lq)
        optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
        )
        with amp_context:
            prediction = model(flow_batch.xt, flow_batch.yt, flow_batch.t)
            loss = process.loss(prediction, flow_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        completed = step + 1
        if rank == 0 and completed % log_every == 0:
            elapsed = time.time() - last_time
            print(
                f"step={completed}/{iterations} loss={loss.item():.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} time={elapsed:.2f}s",
                flush=True,
            )
            last_time = time.time()
        should_save = completed % save_every == 0 or completed == iterations
        if rank == 0 and should_save:
            validation_metrics = None
            if validation_loader is not None:
                validation_metrics = validate(
                    unwrap_model(model),
                    process,
                    validation_loader,
                    device,
                    steps=int(validation_cfg.get("steps", 4)),
                    seed=int(validation_cfg.get("seed", 0)),
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                print(
                    f"validation step={completed} "
                    + " ".join(f"{key}={value:.6f}" for key, value in validation_metrics.items()),
                    flush=True,
                )
            state = checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=completed,
                config=config,
            )
            state["validation"] = validation_metrics
            last_path = output / "last.pt"
            atomic_torch_save(state, last_path)
            if validation_metrics is not None:
                best_records = update_best(
                    output,
                    last_path,
                    best_records,
                    metrics=validation_metrics,
                    step=completed,
                    metric=str(validation_cfg.get("metric", "PSNR")),
                    top_k=int(validation_cfg.get("top_k", 2)),
                )
        if should_save and world_size > 1:
            dist.barrier()


if __name__ == "__main__":
    main()
