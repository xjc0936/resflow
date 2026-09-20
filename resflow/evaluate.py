from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader

from .config import apply_overrides, load_config
from .data import build_dataset, pil_to_tensor
from .metrics import LPIPSMetric, mae, psnr, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate restored PNGs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--restored", required=True)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    dataset = build_dataset(config["data"]["test"], training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_metric = LPIPSMetric(device) if args.lpips else None
    values: dict[str, list[float]] = {"PSNR": [], "SSIM": [], "MAE": []}
    if lpips_metric:
        values["LPIPS"] = []
    for batch in loader:
        prediction_path = Path(args.restored) / f"{batch['name'][0]}.png"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"missing restored image: {prediction_path}")
        with Image.open(prediction_path) as image:
            prediction = pil_to_tensor(image).unsqueeze(0).to(device)
        target = batch["hq"].to(device)
        if prediction.shape != target.shape:
            raise ValueError(
                f"shape mismatch for {prediction_path.name}: "
                f"prediction {tuple(prediction.shape)}, target {tuple(target.shape)}"
            )
        values["PSNR"].append(psnr(prediction, target).item())
        values["SSIM"].append(ssim(prediction, target).item())
        values["MAE"].append(mae(prediction, target).item())
        if lpips_metric:
            values["LPIPS"].append(lpips_metric(prediction, target).item())
    result = {name: sum(scores) / len(scores) for name, scores in values.items()}
    result["images"] = len(dataset)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
