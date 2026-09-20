from __future__ import annotations

from io import BytesIO
from pathlib import Path
import csv
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    array = (
        tensor.detach().float().clamp(-1, 1).add(1).mul(127.5).round()
        .byte().permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _reflect_pad_to_square(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    side = max(height, width)
    pad_y = side - height
    pad_x = side - width
    top, bottom = pad_y // 2, pad_y - pad_y // 2
    left, right = pad_x // 2, pad_x - pad_x // 2
    if pad_y or pad_x:
        mode = "reflect" if min(height, width) > 1 else "edge"
        array = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode=mode)
    return Image.fromarray(array, mode="RGB")


def _images(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.suffix.lower() in IMAGE_EXTENSIONS)


def _key(path: Path, root: Path, strip_suffixes: list[str]) -> str:
    relative = path.relative_to(root)
    stem = relative.stem
    changed = True
    while changed:
        changed = False
        for suffix in strip_suffixes:
            if suffix and stem.lower().endswith(suffix.lower()):
                stem = stem[: -len(suffix)]
                changed = True
    return str(relative.with_name(stem).with_suffix("")).lower()


def _read_manifest(path: Path) -> list[tuple[Path, Path]]:
    pairs = []
    root = path.parent
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"lq", "hq"}.issubset(reader.fieldnames):
            raise ValueError("manifest CSV must have lq,hq columns")
        for row in reader:
            lq, hq = Path(row["lq"]), Path(row["hq"])
            pairs.append((lq if lq.is_absolute() else root / lq, hq if hq.is_absolute() else root / hq))
    return pairs


class PairedImageDataset(Dataset[dict[str, Any]]):
    """Paired LQ/HQ images with synchronized random crops.

    A two-column manifest is the least ambiguous option.  Without one, images
    are paired by relative path/stem after removing configured suffixes.
    """

    def __init__(
        self,
        *,
        hq_dir: str | None = None,
        lq_dir: str | None = None,
        manifest: str | None = None,
        crop_size: int | None = None,
        training: bool = False,
        recursive: bool = True,
        strip_hq_suffixes: list[str] | None = None,
        strip_lq_suffixes: list[str] | None = None,
        horizontal_flip: bool = False,
        jpeg_quality: int | None = None,
        expected_count: int | None = None,
        reflect_pad_to_square: bool = False,
        resize: int | None = None,
    ) -> None:
        self.crop_size = crop_size
        self.training = training
        self.horizontal_flip = horizontal_flip
        self.jpeg_quality = jpeg_quality
        self.reflect_pad_to_square = reflect_pad_to_square
        self.resize = resize
        if manifest:
            self.pairs = _read_manifest(Path(manifest))
        else:
            if hq_dir is None:
                raise ValueError("hq_dir or manifest is required")
            hq_root = Path(hq_dir)
            if jpeg_quality is not None and lq_dir is None:
                self.pairs = [(path, path) for path in _images(hq_root, recursive)]
            else:
                if lq_dir is None:
                    raise ValueError("lq_dir is required for paired data")
                lq_root = Path(lq_dir)
                hq_suffixes = strip_hq_suffixes or []
                lq_suffixes = strip_lq_suffixes or []
                hq_map = {_key(p, hq_root, hq_suffixes): p for p in _images(hq_root, recursive)}
                lq_map = {_key(p, lq_root, lq_suffixes): p for p in _images(lq_root, recursive)}
                missing_hq = sorted(set(lq_map) - set(hq_map))
                missing_lq = sorted(set(hq_map) - set(lq_map))
                if missing_hq or missing_lq:
                    preview = f"missing HQ={missing_hq[:5]}, missing LQ={missing_lq[:5]}"
                    raise ValueError(f"unmatched image pairs: {preview}")
                self.pairs = [(lq_map[key], hq_map[key]) for key in sorted(hq_map)]
        if not self.pairs:
            raise ValueError("dataset contains no image pairs")
        if expected_count is not None and len(self.pairs) != expected_count:
            raise ValueError(
                f"dataset has {len(self.pairs)} pairs, paper split expects {expected_count}"
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def _jpeg(self, image: Image.Image) -> Image.Image:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality, subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")

    def __getitem__(self, index: int) -> dict[str, Any]:
        lq_path, hq_path = self.pairs[index]
        with Image.open(hq_path) as image:
            hq = image.convert("RGB")
        if self.jpeg_quality is None:
            with Image.open(lq_path) as image:
                lq = image.convert("RGB")
        else:
            lq = None

        if lq is not None and lq.size != hq.size:
            raise ValueError(f"pair dimensions differ: {lq_path} {lq.size}, {hq_path} {hq.size}")
        if self.reflect_pad_to_square:
            hq = _reflect_pad_to_square(hq)
            if lq is not None:
                lq = _reflect_pad_to_square(lq)
        if self.resize is not None:
            size = (self.resize, self.resize)
            hq = hq.resize(size, resample=Image.Resampling.BICUBIC)
            if lq is not None:
                lq = lq.resize(size, resample=Image.Resampling.BICUBIC)
        if self.crop_size is not None:
            width, height = hq.size
            crop = self.crop_size
            if width < crop or height < crop:
                raise ValueError(f"{hq_path} is {width}x{height}, smaller than crop {crop}")
            if self.training:
                left = random.randint(0, width - crop)
                top = random.randint(0, height - crop)
            else:
                left, top = (width - crop) // 2, (height - crop) // 2
            box = (left, top, left + crop, top + crop)
            hq = hq.crop(box)
            if lq is not None:
                lq = lq.crop(box)
        if lq is None:
            lq = self._jpeg(hq)
        if self.training and self.horizontal_flip and random.random() < 0.5:
            hq = hq.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            lq = lq.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return {
            "lq": pil_to_tensor(lq),
            "hq": pil_to_tensor(hq),
            "name": hq_path.stem,
            "lq_path": str(lq_path),
            "hq_path": str(hq_path),
        }


def build_dataset(config: dict[str, Any], *, training: bool) -> PairedImageDataset:
    values = dict(config)
    values["training"] = training
    return PairedImageDataset(**values)
