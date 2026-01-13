import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
from tifffile import imread

# Fixed palette (tab20-ish) to avoid extra deps; values in 0-255 RGB
PALETTE: List[Tuple[int, int, int]] = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]

FILENAME_RE = re.compile(r"^(c\d+_t\d+)_([^_]+)_([0-9]+)\.tif$", re.IGNORECASE)


def parse_filename(path: Path) -> Tuple[str, str, str]:
    """Extract (scene_time, competitor, cell_id) from filename."""
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")
    scene_time, competitor, cell_id = m.groups()
    return scene_time, competitor, cell_id


def load_mask(path: Path) -> np.ndarray:
    """Load TIFF and return the segmentation mask (channel index 1)."""
    arr = imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (C,H,W) in {path}, got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"Expected at least 2 channels in {path}, got shape {arr.shape}")
    mask = arr[1]
    return mask


def alpha_composite(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Composite fg over bg. Arrays are float in 0..1 with shape (H,W,4)."""
    alpha_f = fg[..., 3:4]
    alpha_b = bg[..., 3:4]
    out_rgb = fg[..., :3] * alpha_f + bg[..., :3] * alpha_b * (1 - alpha_f)
    out_alpha = alpha_f + alpha_b * (1 - alpha_f)
    out = np.concatenate([out_rgb, out_alpha], axis=-1)
    return out


def mask_outline(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Return boolean outline pixels for a binary mask."""
    eroded = binary_erosion(mask > 0, iterations=iterations, border_value=0)
    return (mask > 0) & (~eroded)


def shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a boolean mask without wrap-around."""
    out = np.zeros_like(mask, dtype=mask.dtype)
    y0 = max(dy, 0)
    x0 = max(dx, 0)
    y1 = mask.shape[0] - max(-dy, 0)
    x1 = mask.shape[1] - max(-dx, 0)
    if y1 <= y0 or x1 <= x0:
        return out
    out[y0:y1, x0:x1] = mask[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return out


def soft_shadow(mask: np.ndarray, sigma: float, dy: int, dx: int) -> np.ndarray:
    """Return a soft shadow alpha map (float 0-1) shifted by (dy, dx)."""
    blurred = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return shift_mask(blurred, dy, dx)


def build_overlay(
    masks: Dict[str, np.ndarray],
    alpha: float = 1.0,
    background: Tuple[int, int, int] = (0, 0, 0),
    color_overrides: Dict[str, Tuple[int, int, int]] | None = None,
    variant: str = "fill",
    order: List[str] | None = None,
) -> np.ndarray:
    """Return RGBA uint8 overlay for given competitor masks.

    variant: "fill" | "fill_outline" | "outline" | "stacked".
    order: optional explicit draw order (back-to-front).
    """
    if not masks:
        raise ValueError("No masks to overlay")
    color_overrides = color_overrides or {}
    sample_mask = next(iter(masks.values()))
    h, w = sample_mask.shape
    canvas = np.zeros((h, w, 4), dtype=np.float32)
    canvas[..., :3] = np.array(background, dtype=np.float32) / 255.0
    canvas[..., 3] = 1.0

    keys = order if order else sorted(masks.keys())

    for idx, competitor in enumerate(keys):
        if competitor not in masks:
            continue
        mask = masks[competitor]
        color_rgb = color_overrides.get(competitor, PALETTE[idx % len(PALETTE)])
        color = np.array(color_rgb, dtype=np.float32) / 255.0
        fg = np.zeros((h, w, 4), dtype=np.float32)
        on = mask > 0
        if not np.any(on):
            continue
        if variant == "fill":
            fg[on, :3] = color
            fg[on, 3] = alpha
        elif variant == "fill_outline":
            outline = mask_outline(mask)
            fg[on, :3] = color
            fg[on, 3] = alpha
            fg[outline, :3] = color
            fg[outline, 3] = 1.0
        elif variant == "outline":
            outline = mask_outline(mask)
            fg[outline, :3] = color
            fg[outline, 3] = 1.0
        elif variant == "stacked":
            # Soft shadow behind layer
            shadow_alpha = soft_shadow(mask > 0, sigma=1.2, dy=3, dx=3) * 0.35
            shadow = np.zeros((h, w, 4), dtype=np.float32)
            shadow[..., :3] = 0.0
            shadow[..., 3] = shadow_alpha
            canvas = alpha_composite(canvas, shadow)
            # Fill
            fg[on, :3] = color
            fg[on, 3] = alpha
            # Bevel: light on top-left, dark on bottom-right
            outline = mask_outline(mask)
            light = shift_mask(outline, dy=-1, dx=-1)
            dark = shift_mask(outline, dy=1, dx=1)
            fg[light, :3] = np.clip(color * 1.25, 0, 1)
            fg[light, 3] = 1.0
            fg[dark, :3] = color * 0.35
            fg[dark, 3] = 1.0
        else:
            raise ValueError(f"Unknown variant '{variant}'")
        canvas = alpha_composite(canvas, fg)

    out = np.clip(canvas * 255, 0, 255).astype(np.uint8)
    return out


def gather_masks(input_dir: Path) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """Group masks by (scene_time, cell_id)."""
    groups: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for path in sorted(input_dir.glob("*.tif")):
        scene_time, competitor, cell_id = parse_filename(path)
        mask = load_mask(path)
        key = (scene_time, cell_id)
        groups.setdefault(key, {})[competitor] = mask
    return groups


def save_overlay(output_dir: Path, scene_time: str, cell_id: str, overlay: np.ndarray, suffix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scene_time}_{cell_id}_overlay{suffix}.png"
    from tifffile import imwrite

    imwrite(out_path, overlay, photometric="rgb")
    return out_path


def run(
    input_dir: Path,
    output_dir: Path,
    alpha: float,
    background: Tuple[int, int, int],
    color_overrides: Dict[str, Tuple[int, int, int]],
    variants: List[str],
    only_scene_time: str | None,
    only_cell_id: str | None,
    order: List[str] | None,
) -> None:
    groups = gather_masks(input_dir)
    if not groups:
        raise SystemExit(f"No TIFF files found in {input_dir}")

    for (scene_time, cell_id), masks in groups.items():
        if only_scene_time and scene_time != only_scene_time:
            continue
        if only_cell_id and cell_id != only_cell_id:
            continue
        for variant in variants:
            overlay = build_overlay(
                masks,
                alpha=alpha,
                background=background,
                color_overrides=color_overrides,
                variant=variant,
                order=order,
            )
            suffix = "" if variant == "fill" else f"_{variant}"
            out_path = save_overlay(output_dir, scene_time, cell_id, overlay, suffix)
            print(f"wrote {out_path} ({len(masks)} competitors, {variant})")


def parse_color_override(values: List[str] | None) -> Dict[str, Tuple[int, int, int]]:
    overrides: Dict[str, Tuple[int, int, int]] = {}
    if not values:
        return overrides
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --color format '{item}', expected name=R,G,B")
        name, rgb_str = item.split("=", 1)
        parts = rgb_str.split(",")
        if len(parts) != 3:
            raise ValueError(f"Invalid RGB '{rgb_str}', expected R,G,B")
        overrides[name] = tuple(int(p) for p in parts)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay competitor segmentations for each cell crop")
    parser.add_argument("input_dir", type=Path, help="Directory with crop TIFFs (channel-first, segmentation in channel 1)")
    parser.add_argument("output_dir", type=Path, help="Where to write overlay PNGs")
    parser.add_argument("--alpha", type=float, default=0.8, help="Per-mask opacity (0-1). Default: 0.8")
    parser.add_argument(
        "--background",
        type=str,
        default="0,0,0",
        help="Background color as R,G,B (0-255). Default: 0,0,0 (black)",
    )
    parser.add_argument(
        "--color",
        action="append",
        help="Color override per competitor: name=R,G,B. Repeatable.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="fill,fill_outline,outline,stacked",
        help="Comma-separated variants: fill, fill_outline, outline, stacked",
    )
    parser.add_argument("--scene", type=str, default=None, help="Optional scene_time filter (e.g., c01_t0126)")
    parser.add_argument("--cell", type=str, default=None, help="Optional cell_id filter (e.g., 1)")
    parser.add_argument(
        "--order",
        type=str,
        default=None,
        help="Optional comma-separated competitor draw order (back-to-front)",
    )
    args = parser.parse_args()

    background = tuple(int(p) for p in args.background.split(","))
    color_overrides = parse_color_override(args.color)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    order = [v.strip() for v in args.order.split(",") if v.strip()] if args.order else None

    run(
        args.input_dir,
        args.output_dir,
        alpha=args.alpha,
        background=background,
        color_overrides=color_overrides,
        variants=variants,
        only_scene_time=args.scene,
        only_cell_id=args.cell,
        order=order,
    )


if __name__ == "__main__":
    main()

