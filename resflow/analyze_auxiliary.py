from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .config import load_config
from .data import build_dataset, tensor_to_pil
from .flow import ResFlowProcess
from .metrics import LPIPSMetric, mae, psnr, ssim
from .model import build_model
from .utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intervene on ResFlow's auxiliary trajectory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20250920)
    parser.add_argument("--multi-y-samples", type=int, default=20)
    parser.add_argument("--multi-y-images", type=int, default=32)
    parser.add_argument("--velocity-images", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--save-examples", type=int, default=8)
    parser.add_argument("--skip-fid-all", action="store_true")
    parser.add_argument("--only-fid-all", action="store_true")
    return parser.parse_args()


def amp_context(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )


def randn(shape: tuple[int, ...], generator: torch.Generator, reference: Tensor) -> Tensor:
    return torch.randn(shape, generator=generator, device=reference.device, dtype=reference.dtype)


@torch.no_grad()
def rollout(
    model: nn.Module,
    process: ResFlowProcess,
    x1: Tensor,
    noises: list[Tensor],
    *,
    steps: int,
    return_states: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    if len(noises) != steps:
        raise ValueError(f"expected {steps} noise tensors, got {len(noises)}")
    x = x1.clone()
    states = []
    times = torch.linspace(1.0, 0.0, steps + 1, device=x.device, dtype=x.dtype)
    for index in range(steps):
        if return_states:
            states.append(x.clone())
        t_value, next_t = times[index], times[index + 1]
        t = t_value.expand(x.shape[0])
        sigma = process.sigma_y(t).reshape(x.shape[0], 1, 1, 1)
        velocity = model(x, sigma * noises[index], t)
        x = x + (next_t - t_value) * velocity[:, : x.shape[1]]
    restored = x.clamp(-1.0, 1.0)
    return (restored, states) if return_states else restored


def intervention_noises(
    reference: Tensor,
    *,
    steps: int,
    generator: torch.Generator,
    fixed_typical: Tensor,
) -> dict[str, list[Tensor]]:
    shape = tuple(reference.shape)
    noise_a = randn(shape, generator, reference)
    noise_b = randn(shape, generator, reference)
    independent = [noise_a] + [randn(shape, generator, reference) for _ in range(steps - 1)]
    modes: dict[str, list[Tensor]] = {
        "persistent": [noise_a] * steps,
        "step_resample": independent,
        "fixed_typical": [fixed_typical.expand_as(reference)] * steps,
    }
    for transition in range(1, steps):
        # transition=1 means A at t=1, then B from t=.75 onward.
        modes[f"half_swap_after_step_{transition}"] = [noise_a] * transition + [noise_b] * (
            steps - transition
        )
    return modes


class QualityAccumulator:
    def __init__(self, device: torch.device, mode_names: list[str]):
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.sums = {
            mode: {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0, "delta_LPIPS": 0.0}
            for mode in mode_names
        }
        self.count = 0
        self.lpips = LPIPSMetric(device)
        self.fids = {
            mode: FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            for mode in mode_names
        }

    @torch.no_grad()
    def update(self, outputs: dict[str, Tensor], target: Tensor) -> None:
        persistent = outputs["persistent"]
        real = target.float().add(1).div(2).clamp(0, 1)
        batch = target.shape[0]
        for name, output in outputs.items():
            self.sums[name]["PSNR"] += psnr(output, target).sum().item()
            self.sums[name]["SSIM"] += ssim(output, target).sum().item()
            self.sums[name]["LPIPS"] += self.lpips(output, target).sum().item()
            if name != "persistent":
                self.sums[name]["delta_LPIPS"] += self.lpips(output, persistent).sum().item()
            fake = output.float().add(1).div(2).clamp(0, 1)
            self.fids[name].update(real, real=True)
            self.fids[name].update(fake, real=False)
        self.count += batch

    def compute(self) -> dict[str, dict[str, float]]:
        result = {}
        for name, values in self.sums.items():
            result[name] = {key: value / self.count for key, value in values.items()}
            result[name]["FID_test"] = self.fids[name].compute().item()
        return result


def dct_matrix(size: int, device: torch.device) -> Tensor:
    positions = torch.arange(size, device=device, dtype=torch.float64)
    frequencies = torch.arange(size, device=device, dtype=torch.float64)[:, None]
    matrix = torch.cos(math.pi / size * (positions[None] + 0.5) * frequencies)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


def dct_variance_bands(outputs: Tensor) -> dict[str, float]:
    # outputs: [M,C,H,W], measured on [0,1] RGB with an orthonormal DCT-II.
    size = outputs.shape[-1]
    matrix = dct_matrix(size, outputs.device)
    values = outputs.double().add(1).div(2)
    coefficients = torch.einsum("uh,mchw,vw->mcuv", matrix, values, matrix)
    variance = coefficients.var(dim=0, unbiased=False).mean(dim=0)
    coordinate = torch.arange(size, device=outputs.device, dtype=torch.float64)
    radius = torch.sqrt(coordinate[:, None].square() + coordinate[None, :].square())
    radius = radius / (math.sqrt(2.0) * (size - 1))
    masks = {
        "low_0_0.25": radius < 0.25,
        "mid_0.25_0.5": (radius >= 0.25) & (radius < 0.5),
        "high_0.5_1.0": radius >= 0.5,
    }
    total = variance.sum().clamp_min(1e-30)
    result = {}
    for name, mask in masks.items():
        result[f"{name}_mean_variance"] = variance[mask].mean().item()
        result[f"{name}_energy_fraction"] = (variance[mask].sum() / total).item()
    return result


@torch.no_grad()
def pairwise_lpips(metric: LPIPSMetric, outputs: Tensor, chunk: int = 64) -> float:
    first, second = torch.triu_indices(outputs.shape[0], outputs.shape[0], offset=1)
    values = []
    for start in range(0, first.numel(), chunk):
        indices = slice(start, start + chunk)
        values.append(metric(outputs[first[indices]], outputs[second[indices]]).float().cpu())
    return torch.cat(values).mean().item()


@torch.no_grad()
def multi_y_analysis(
    model: nn.Module,
    process: ResFlowProcess,
    dataset,
    device: torch.device,
    *,
    steps: int,
    samples: int,
    images: int,
    seed: int,
    output_dir: Path,
) -> dict:
    metric = LPIPSMetric(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    per_image = []
    example_dir = output_dir / "multi_y_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    for image_index in range(min(images, len(dataset))):
        item = dataset[image_index]
        lq = item["lq"].unsqueeze(0).to(device)
        batched_lq = lq.expand(samples, -1, -1, -1)
        noise = randn(tuple(batched_lq.shape), generator, batched_lq)
        with amp_context(device):
            restored = rollout(model, process, batched_lq, [noise] * steps, steps=steps)
        pixel_values = restored.float().add(1).div(2)
        record = {
            "name": item["name"],
            "pixel_variance": pixel_values.var(dim=0, unbiased=False).mean().item(),
            "pairwise_LPIPS": pairwise_lpips(metric, restored),
            "dct": dct_variance_bands(restored),
        }
        per_image.append(record)
        if image_index < 4:
            for sample_index in range(min(samples, 8)):
                tensor_to_pil(restored[sample_index]).save(
                    example_dir / f"{image_index:02d}_{item['name']}_y{sample_index:02d}.png"
                )
        if (image_index + 1) % 8 == 0 or image_index + 1 == min(images, len(dataset)):
            print(f"multi-y {image_index + 1}/{min(images, len(dataset))}", flush=True)
    keys = per_image[0]["dct"].keys()
    return {
        "images": len(per_image),
        "samples_per_image": samples,
        "mean_pixel_variance": sum(item["pixel_variance"] for item in per_image) / len(per_image),
        "mean_pairwise_LPIPS": sum(item["pairwise_LPIPS"] for item in per_image) / len(per_image),
        "mean_dct": {
            key: sum(item["dct"][key] for item in per_image) / len(per_image) for key in keys
        },
        "per_image": per_image,
    }


@torch.no_grad()
def velocity_analysis(
    model: nn.Module,
    process: ResFlowProcess,
    dataset,
    device: torch.device,
    *,
    steps: int,
    samples: int,
    images: int,
    seed: int,
    epsilon: float,
) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed)
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)[:-1]
    accumulators = [
        {"velocity_variance": 0.0, "relative_variance": 0.0, "finite_difference": 0.0}
        for _ in range(steps)
    ]
    image_count = min(images, len(dataset))
    for image_index in range(image_count):
        lq = dataset[image_index]["lq"].unsqueeze(0).to(device)
        identity = randn(tuple(lq.shape), generator, lq)
        with amp_context(device):
            _, states = rollout(
                model, process, lq, [identity] * steps, steps=steps, return_states=True
            )
        for time_index, (t_value, state) in enumerate(zip(times, states)):
            state_batch = state.expand(samples, -1, -1, -1)
            t = t_value.expand(samples).to(state.dtype)
            sigma = process.sigma_y(t).reshape(samples, 1, 1, 1)
            noises = randn(tuple(state_batch.shape), generator, state_batch)
            with amp_context(device):
                velocities = model(state_batch, sigma * noises, t)[:, :3].float()
            variance = velocities.var(dim=0, unbiased=False).mean()
            signal = velocities.square().mean().clamp_min(1e-30)

            base_y = sigma[:1] * identity
            direction = randn(tuple(identity.shape), generator, identity)
            base_t = t[:1]
            with amp_context(device):
                base_velocity = model(state, base_y, base_t)[:, :3].float()
                shifted_velocity = model(state, base_y + epsilon * direction, base_t)[:, :3].float()
            numerator = (shifted_velocity - base_velocity).square().mean().sqrt()
            denominator = epsilon * direction.square().mean().sqrt().clamp_min(1e-30)

            accumulators[time_index]["velocity_variance"] += variance.item()
            accumulators[time_index]["relative_variance"] += (variance / signal).item()
            accumulators[time_index]["finite_difference"] += (numerator / denominator).item()
        if (image_index + 1) % 8 == 0 or image_index + 1 == image_count:
            print(f"velocity intervention {image_index + 1}/{image_count}", flush=True)
    return {
        "images": image_count,
        "y_samples_per_state": samples,
        "epsilon": epsilon,
        "per_timestep": [
            {
                "t": times[index].item(),
                **{key: value / image_count for key, value in accumulator.items()},
            }
            for index, accumulator in enumerate(accumulators)
        ],
    }


def save_json(result: dict, output: Path) -> None:
    path = output / "results.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def compute_fid_all(
    model: nn.Module,
    process: ResFlowProcess,
    config: dict,
    test_dataset,
    device: torch.device,
    *,
    batch_size: int,
    workers: int,
    steps: int,
    seed: int,
) -> tuple[float, int]:
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    train_dataset = build_dataset(config["data"]["train"], training=False)
    all_dataset = ConcatDataset((train_dataset, test_dataset))
    loader = DataLoader(
        all_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"FID(all) on {len(all_dataset)} images", flush=True)
    processed = 0
    for batch_index, batch in enumerate(loader, start=1):
        lq = batch["lq"].to(device, non_blocking=True)
        hq = batch["hq"].to(device, non_blocking=True)
        identity = randn(tuple(lq.shape), generator, lq)
        with amp_context(device):
            restored = rollout(model, process, lq, [identity] * steps, steps=steps)
        fid.update(hq.float().add(1).div(2).clamp(0, 1), real=True)
        fid.update(restored.float().add(1).div(2).clamp(0, 1), real=False)
        processed += lq.shape[0]
        if batch_index % 25 == 0 or processed == len(all_dataset):
            print(f"FID(all) {processed}/{len(all_dataset)}", flush=True)
    return fid.compute().item(), len(all_dataset)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    process = ResFlowProcess(**config["flow"])
    test_dataset = build_dataset(config["data"]["test"], training=False)
    if args.only_fid_all:
        score, count = compute_fid_all(
            model,
            process,
            config,
            test_dataset,
            device,
            batch_size=args.batch_size,
            workers=args.workers,
            steps=args.steps,
            seed=args.seed,
        )
        result = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_step": checkpoint.get("step"),
            "seed": args.seed,
            "steps": args.steps,
            "FID_all": score,
            "FID_all_images": count,
        }
        save_json(result, output)
        print(json.dumps(result, indent=2), flush=True)
        return
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    transition_times = [1.0 - index / args.steps for index in range(1, args.steps)]
    mode_names = ["persistent", "step_resample", "fixed_typical"] + [
        f"half_swap_after_step_{index}" for index in range(1, args.steps)
    ]
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "seed": args.seed,
        "steps": args.steps,
        "time_grid": [1.0 - index / args.steps for index in range(args.steps + 1)],
        "half_swap_definition": {
            f"half_swap_after_step_{index}": {
                "A_at": [1.0 - j / args.steps for j in range(index)],
                "B_from_t": transition_times[index - 1],
            }
            for index in range(1, args.steps)
        },
        "test_images": len(test_dataset),
    }

    quality = QualityAccumulator(device, mode_names)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    fixed_typical = randn((1, 3, 64, 64), generator, torch.empty((), device=device))
    examples_dir = output / "test_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    print(f"phase 1/4: test interventions on {len(test_dataset)} images", flush=True)
    for batch_index, batch in enumerate(test_loader, start=1):
        lq = batch["lq"].to(device, non_blocking=True)
        hq = batch["hq"].to(device, non_blocking=True)
        schedules = intervention_noises(
            lq,
            steps=args.steps,
            generator=generator,
            fixed_typical=fixed_typical,
        )
        outputs = {}
        for name, noises in schedules.items():
            with amp_context(device):
                outputs[name] = rollout(model, process, lq, noises, steps=args.steps)
        quality.update(outputs, hq)
        while saved < args.save_examples and saved < quality.count:
            local = saved - (quality.count - lq.shape[0])
            if local < 0 or local >= lq.shape[0]:
                break
            name = batch["name"][local]
            tensor_to_pil(lq[local]).save(examples_dir / f"{saved:02d}_{name}_lq.png")
            tensor_to_pil(hq[local]).save(examples_dir / f"{saved:02d}_{name}_hq.png")
            for mode, values in outputs.items():
                tensor_to_pil(values[local]).save(examples_dir / f"{saved:02d}_{name}_{mode}.png")
            saved += 1
        if batch_index % 5 == 0 or quality.count == len(test_dataset):
            print(f"test interventions {quality.count}/{len(test_dataset)}", flush=True)
    result["intervention_quality"] = quality.compute()
    result["metric_protocol"] = {
        "PSNR": "RGB [0,1], no border crop",
        "SSIM": "RGB, 11x11 Gaussian sigma=1.5, valid window",
        "LPIPS": "AlexNet LPIPS on [-1,1]",
        "FID": "torchmetrics/torch-fidelity Inception-v3 2048D; inputs resized internally",
    }
    save_json(result, output)

    if not args.skip_fid_all:
        score, count = compute_fid_all(
            model,
            process,
            config,
            test_dataset,
            device,
            batch_size=args.batch_size,
            workers=args.workers,
            steps=args.steps,
            seed=args.seed,
        )
        result["FID_all"] = score
        result["FID_all_images"] = count
        save_json(result, output)

    subset_count = max(args.multi_y_images, args.velocity_images)
    analysis_subset = Subset(test_dataset, range(min(subset_count, len(test_dataset))))
    print("phase 3/4: multi-y output diversity", flush=True)
    result["multi_y"] = multi_y_analysis(
        model,
        process,
        analysis_subset,
        device,
        steps=args.steps,
        samples=args.multi_y_samples,
        images=args.multi_y_images,
        seed=args.seed + 1,
        output_dir=output,
    )
    save_json(result, output)
    print("phase 4/4: fixed-x_t velocity intervention", flush=True)
    result["velocity_intervention"] = velocity_analysis(
        model,
        process,
        analysis_subset,
        device,
        steps=args.steps,
        samples=args.multi_y_samples,
        images=args.velocity_images,
        seed=args.seed + 2,
        epsilon=args.epsilon,
    )
    save_json(result, output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
