from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import apply_overrides, load_config
from .data import build_dataset, tensor_to_pil
from .flow import ResFlowProcess
from .model import build_model
from .utils import crop_original, pad_to_multiple, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore full-resolution images with ResFlow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    process = ResFlowProcess(**config["flow"])
    dataset = build_dataset(config["data"]["test"], training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    steps = args.steps or int(config["inference"]["steps"])
    multiple = 2 ** (len(config["model"]["channel_multipliers"]) - 1)
    for index, batch in enumerate(loader, start=1):
        lq = batch["lq"].to(device)
        padded, original_shape = pad_to_multiple(lq, multiple)
        restored = process.restore(model, padded, steps=steps)
        restored = crop_original(restored, original_shape)
        name = batch["name"][0]
        tensor_to_pil(restored[0]).save(output / f"{name}.png")
        print(f"[{index}/{len(dataset)}] {name}.png", flush=True)


if __name__ == "__main__":
    main()

